"""Evaluation of an auto-labeling run against hidden ground truth (benchmark use only).

Answers four questions:
1. How good are the pseudo-labels?                     -> metrics on the unlabeled pool U
2. Does the recall guarantee hold?                       -> realised defect recall at the conformal threshold
3. Does the self-assessment rank / estimate quality?      -> correlations, AUROCs, plug-in estimates
4. How precise is an audit with PPI vs. audit alone?     -> CI coverage and width over random audits
5. How much does human review of the riskiest labels buy? -> review-budget curves
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

from . import metrics as M
from . import ppi
from .postprocess import hysteresis, select_thresholds
from .quality import true_quality


def _clean(d: Dict) -> Dict:
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in d.items()}


def eval_round(rr: Dict, gts_U: Sequence[np.ndarray], gts_X: Optional[Sequence[np.ndarray]] = None,
               tol: int = 2, merge_radius: int = 2) -> Dict:
    probs = [as_prob(p) for p in rr["prob_U"]]
    out = {"U": _clean(M.summarize(rr["mask_U"], gts_U, tol, merge_radius, probs=probs)),
           "U_safe": _clean(M.summarize(rr["mask_safe_U"], gts_U, tol, merge_radius))}
    out["U"]["px_ap"] = M.average_precision_px(probs, gts_U)
    out["U"]["px_auroc"] = M.pixel_auroc(probs, gts_U)
    # oracle thresholds on U: upper bound of what threshold selection could achieve
    th_or, m_or = select_thresholds(probs, gts_U, tol, merge_radius)
    out["U_oracle_threshold"] = _clean(m_or)
    out["U_oracle_threshold"]["thresholds"] = th_or
    if gts_X is not None and len(rr.get("prob_X", [])):
        th = rr["thresholds"]
        px = [as_prob(p) for p in rr["prob_X"]]
        mx = [hysteresis(p, th["t_seed"], th["t_ext"], int(th["min_area"])) for p in px]
        out["X"] = _clean(M.summarize(mx, gts_X, tol, merge_radius, probs=px))
        out["X"]["px_ap"] = M.average_precision_px(px, gts_X)
    out["thresholds"] = rr["thresholds"]
    out["t_conformal"], out["t_pac"], out["t_safe"] = rr["t_conformal"], rr["t_pac"], rr["t_safe"]
    out["n_labeled_defects"] = int(len(rr["defect_scores_oof"]))
    out["certified_recall_at_t_seed"] = list(rr["certified_recall_at_t_seed"])
    out["oof"] = _clean(rr["oof_metrics"])
    out["oof_ece"] = rr["oof_ece"]
    return out


def eval_single_member(rr: Dict, gts_U, tol: int = 2, merge_radius: int = 2, k: int = 0) -> Dict:
    th = rr["thresholds"]
    probs = [m[k].astype(np.float32) for m in rr["member_U"]]
    masks = [hysteresis(p, th["t_seed"], th["t_ext"], int(th["min_area"])) for p in probs]
    out = _clean(M.summarize(masks, gts_U, tol, merge_radius, probs=probs))
    out["px_ap"] = M.average_precision_px(probs, gts_U)
    return out


def _q8(p):
    return np.round(np.clip(np.asarray(p, np.float32), 0, 1) * 255).astype(np.uint8)


def compact_results(res: Dict) -> Dict:
    """Shrink a result dict for storage: probabilities -> uint8, drop per-member maps."""
    for rr in res["rounds"]:
        rr.pop("member_U", None)
        for key in ("prob_U", "prob_X", "oof_prob"):
            if key in rr:
                rr[key] = [_q8(p) for p in rr[key]]
    for key in ("anomaly_U", "anomaly_L"):
        if key in res:
            res[key] = [np.asarray(a, np.float16) for a in res[key]]
    return res


def as_prob(p) -> np.ndarray:
    p = np.asarray(p)
    return p.astype(np.float32) / 255.0 if p.dtype == np.uint8 else p.astype(np.float32)


def conformal_check(rr: Dict, gts_U, alphas=(0.05, 0.1, 0.2), tol=2, merge_radius=2) -> Dict:
    """Realised defect recall on U when the seed threshold is set by the conformal rule."""
    from . import conformal as C
    probs = [as_prob(p) for p in rr["prob_U"]]
    true_scores = C.defect_scores(probs, gts_U, tol, merge_radius)
    out = {}
    for a in alphas:
        t = C.conformal_threshold(rr["defect_scores_oof"], a)
        tp = C.pac_threshold(rr["defect_scores_oof"], a, 0.1)
        out[f"alpha={a}"] = {
            "t_conformal": t, "realised_recall_conformal": float(np.mean(true_scores >= t)),
            "t_pac": tp, "realised_recall_pac": float(np.mean(true_scores >= tp)),
            "target_recall": 1 - a,
        }
    return out


def self_assessment(res: Dict, gts_U, tol=2, merge_radius=2) -> Dict:
    last = res["rounds"][-1]
    feats = res["features_U"]
    truth = [true_quality(m, g, tol, merge_radius) for m, g in zip(last["mask_U"], gts_U)]
    out = {}
    dice_true = np.array([t["dice_tol"] for t in truth])
    defective = np.array([t["img_gt"] > 0 for t in truth])
    for k in ("exp_dice", "member_dice", "mean_conf_in_mask"):
        v = np.array([f.get(k, np.nan) for f in feats], np.float64)
        ok = ~np.isnan(v)
        if ok.sum() > 5:
            out[f"spearman_{k}_vs_dice_tol"] = float(spearmanr(v[ok], dice_true[ok]).correlation)
            sel = ok & (defective | np.array([f["n_pred"] > 0 for f in feats]))
            if sel.sum() > 5:
                out[f"spearman_{k}_vs_dice_tol_nonempty"] = float(spearmanr(v[sel], dice_true[sel]).correlation)
    missed = np.array([t["missed_defect"] for t in truth])
    bad = np.array([t["bad_label"] for t in truth])
    signals = {
        "p_missed_defect": np.array([f["p_missed_defect"] for f in feats]),
        "anomaly_out": np.array([f.get("anomaly_out", np.nan) for f in feats]),
        "inv_member_dice": -np.array([f.get("member_dice", np.nan) for f in feats]),
        "inv_exp_dice": -np.array([f["exp_dice"] for f in feats]),
        "risk": np.asarray(res["risk_U"]),
    }
    for name, s in signals.items():
        if np.all(np.isnan(s)):
            continue
        out[f"auroc_{name}_for_missed_defect"] = M.auroc(np.nan_to_num(s, nan=np.nanmin(s)), missed)
        out[f"auroc_{name}_for_bad_label"] = M.auroc(np.nan_to_num(s, nan=np.nanmin(s)), bad)
    # plug-in (no audit) estimates of dataset-level quality vs truth
    S = {k: sum(t[k] for t in truth) for k in ("tp", "fp", "fn", "gt_comp", "gt_comp_hit", "pred_comp",
                                                "pred_comp_hit")}
    E = {k: sum(f[k] for f in feats) for k in ("exp_tp", "exp_fp", "exp_fn", "exp_hits", "exp_defects",
                                                "exp_false_comps", "n_pred_comps")}
    out["px_precision_true"] = S["tp"] / max(1, S["tp"] + S["fp"])
    out["px_precision_selfest"] = E["exp_tp"] / max(1e-9, E["exp_tp"] + E["exp_fp"])
    out["px_recall_true"] = S["tp"] / max(1, S["tp"] + S["fn"])
    out["px_recall_selfest"] = E["exp_tp"] / max(1e-9, E["exp_tp"] + E["exp_fn"])
    out["defect_recall_true"] = S["gt_comp_hit"] / max(1, S["gt_comp"])
    out["defect_recall_selfest"] = E["exp_hits"] / max(1e-9, E["exp_defects"])
    out["defect_precision_true"] = S["pred_comp_hit"] / max(1, S["pred_comp"])
    out["defect_precision_selfest"] = 1 - E["exp_false_comps"] / max(1e-9, E["n_pred_comps"])
    out["mean_dice_tol_true"] = float(dice_true.mean())
    out["mean_dice_tol_selfest"] = float(np.mean([f["exp_dice"] for f in feats]))
    return out


def audit_simulation(res: Dict, gts_U, ns=(10, 20, 40, 80), reps: int = 500, alpha: float = 0.1,
                     tol: int = 2, merge_radius: int = 2, seed: int = 0) -> Dict:
    """Random audits of n images: coverage and width of PPI vs classical confidence intervals."""
    last = res["rounds"][-1]
    feats = res["features_U"]
    truth = [true_quality(m, g, tol, merge_radius) for m, g in zip(last["mask_U"], gts_U)]
    N = len(truth)
    # (name, true numerator, true denominator, predicted numerator, predicted denominator) per image
    T = lambda k: np.array([t[k] for t in truth], np.float64)
    F = lambda k: np.array([f[k] for f in feats], np.float64)
    ratio_metrics = {
        "px_recall": (T("tp"), T("tp") + T("fn"), F("exp_tp"), F("exp_tp") + F("exp_fn")),
        "px_precision": (T("tp"), T("tp") + T("fp"), F("exp_tp"), F("exp_tp") + F("exp_fp")),
        "defect_recall": (T("gt_comp_hit"), T("gt_comp"), F("exp_hits"), F("exp_defects")),
        "defect_precision": (T("pred_comp_hit"), T("pred_comp"), F("n_pred_comps") - F("exp_false_comps"),
                             F("n_pred_comps")),
    }
    mean_metrics = {"mean_dice_tol": (T("dice_tol"), F("exp_dice")),
                    "share_missed": (np.array([t["missed_defect"] for t in truth]), F("p_missed_defect"))}
    rng = np.random.RandomState(seed)
    out = {}
    for name, (a, b, fa, fb) in ratio_metrics.items():
        true = a.sum() / max(1e-9, b.sum())
        for n in ns:
            if n >= N:
                continue
            rows = {"ppi": [], "classical": []}
            for _ in range(reps):
                s = rng.choice(N, n, replace=False)
                rest = np.setdiff1d(np.arange(N), s)
                rows["ppi"].append(ppi.estimate_ratio(a[s], b[s], fa[s], fb[s], fa[rest], fb[rest], alpha))
                rows["classical"].append(ppi.classical_ratio(a[s], b[s], alpha, N))
            for meth, v in rows.items():
                v = np.array(v, np.float64)
                ok = ~np.isnan(v[:, 0])
                lo, hi = np.clip(v[ok, 1], 0, 1), np.clip(v[ok, 2], 0, 1)
                out[f"{name}/n={n}/{meth}"] = {
                    "true": float(true), "coverage": float(np.mean((lo <= true) & (true <= hi))),
                    "width": float(np.mean(hi - lo)), "abs_err": float(np.mean(np.abs(v[ok, 0] - true))),
                    "valid_frac": float(ok.mean())}
    for name, (y, f) in mean_metrics.items():
        true = y.mean()
        for n in ns:
            if n >= N:
                continue
            rows = {"ppi": [], "classical": []}
            for _ in range(reps):
                s = rng.choice(N, n, replace=False)
                rest = np.setdiff1d(np.arange(N), s)
                rows["ppi"].append(ppi.estimate_mean(y[s], f[s], f[rest], alpha))
                rows["classical"].append(ppi.classical_mean(y[s], alpha, N))
            for meth, v in rows.items():
                v = np.array(v, np.float64)
                lo, hi = np.clip(v[:, 1], 0, 1), np.clip(v[:, 2], 0, 1)
                out[f"{name}/n={n}/{meth}"] = {
                    "true": float(true), "coverage": float(np.mean((lo <= true) & (true <= hi))),
                    "width": float(np.mean(hi - lo)), "abs_err": float(np.mean(np.abs(v[:, 0] - true)))}
    return out


def review_curve(res: Dict, gts_U, fracs=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5), tol=2, merge_radius=2,
                 seed: int = 0) -> Dict:
    """Quality of the label pool after a human corrects the top-x% images of a review ordering."""
    last = res["rounds"][-1]
    masks = last["mask_U"]
    truth = [true_quality(m, g, tol, merge_radius) for m, g in zip(masks, gts_U)]
    gt_stats = [M.image_stats(g, g, tol, merge_radius) for g in gts_U]
    N = len(masks)
    rng = np.random.RandomState(seed)
    bad = np.array([t["bad_label"] for t in truth], np.float64)
    orders = {
        "risk": np.argsort(-np.asarray(res["risk_U"])),
        "random": rng.permutation(N),
        "oracle": np.argsort(-(bad * 10 + (1 - np.array([t["dice_tol"] for t in truth])))),
    }
    out = {}
    for name, order in orders.items():
        for fr in fracs:
            k = int(round(fr * N))
            fixed = set(order[:k].tolist())
            stats = [gt_stats[i] if i in fixed else truth[i] for i in range(N)]
            agg = M.aggregate(stats)
            out[f"{name}/{fr:.2f}"] = {k2: float(agg[k2]) for k2 in
                                       ("defect_recall", "defect_precision", "tol_f1", "px_f1", "img_recall",
                                        "mean_dice_tol")}
            out[f"{name}/{fr:.2f}"]["bad_labels_left"] = float(sum(bad[i] for i in range(N) if i not in fixed))
    return out


def anomaly_baseline(res: Dict, gts_L, gts_U, tol=2, merge_radius=2) -> Optional[Dict]:
    """Unsupervised normality model alone; only its threshold is chosen on the labeled set."""
    if "anomaly_U" not in res:
        return None
    aU = [a.astype(np.float32) for a in res["anomaly_U"]]
    lo = float(np.percentile([a.min() for a in aU], 5))
    hi = float(np.percentile([a.max() for a in aU], 99.5))
    norm = lambda a: np.clip((np.asarray(a, np.float32) - lo) / (hi - lo + 1e-9), 0, 1)
    pL, pU = [norm(a) for a in res["anomaly_L"]], [norm(a) for a in aU]
    grid = tuple(np.linspace(0.05, 0.95, 19))
    th, _ = select_thresholds(pL, gts_L, tol, merge_radius, seed_grid=grid, ext_ratios=(1.0, 0.8))
    masks = [hysteresis(p, th["t_seed"], th["t_ext"], int(th["min_area"])) for p in pU]
    out = _clean(M.summarize(masks, gts_U, tol, merge_radius, probs=pU))
    out["px_ap"] = M.average_precision_px(pU, gts_U)
    out["thresholds"] = th
    return out


# ============================================================================ Phase-1 metrics (00 §6)
# Pool quantities of CertAL-lite (00 §4.3) measured against the hidden GT of U:
#   b_i  = 1 if image i contains a GT defect component that the label does not hit (tol, any-hit)
#   F_i  = number of predicted components without any GT pixel within tol (false components)
#   zeta = mean b_i (share of "bad images"), phi = mean F_i (false components per image)
# K'(k) = number of bad images left after a perfect reviewer repaired the k highest-risk images.

SIZE_CLASSES = (("S", 0, 100), ("M", 100, 1000), ("L", 1000, float("inf")))   # GT component area in px


def pool_errors(masks: Sequence[np.ndarray], gts: Sequence[np.ndarray], tol: int = 2,
                merge_radius: int = 2) -> Dict[str, np.ndarray]:
    """Per-image b_i, F_i, defective flag and per-component (area, hit) for size classes."""
    from scipy import ndimage as ndi
    b, F, defective, areas, hits = [], [], [], [], []
    for m, g in zip(masks, gts):
        m, g = np.asarray(m, bool), np.asarray(g, bool)
        glab, ng = M.label_components(g, merge_radius)
        pr_d = M.dilate(m, tol)
        miss = 0
        if ng:
            idx = np.arange(1, ng + 1)
            hit = np.asarray(ndi.maximum(pr_d.astype(np.uint8), glab, index=idx)) > 0
            area = np.asarray(ndi.sum(np.ones(g.shape, np.float32), glab, index=idx))
            areas.extend(area.tolist())
            hits.extend(hit.tolist())
            miss = int((~hit).any())
        plab, npr = M.label_components(m, merge_radius)
        fc = 0
        if npr:
            g_d = M.dilate(g, tol)
            ph = np.asarray(ndi.maximum(g_d.astype(np.uint8), plab, index=np.arange(1, npr + 1))) > 0
            fc = int((~ph).sum())
        b.append(miss)
        F.append(fc)
        defective.append(int(ng > 0))
    return {"b": np.array(b, np.int64), "F": np.array(F, np.int64), "defective": np.array(defective, bool),
            "comp_area": np.array(areas, np.float64), "comp_hit": np.array(hits, bool)}


def size_class_recall(pe: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    out = {}
    for name, lo, hi in SIZE_CLASSES:
        sel = (pe["comp_area"] >= lo) & (pe["comp_area"] < hi)
        out[name] = {"n": int(sel.sum()),
                     "recall": float(pe["comp_hit"][sel].mean()) if sel.any() else float("nan")}
    return out


def pool_summary(pe: Dict[str, np.ndarray]) -> Dict[str, float]:
    N = len(pe["b"])
    clean = ~pe["defective"]
    return {"N": N, "K": int(pe["b"].sum()), "zeta": float(pe["b"].mean()),
            "phi": float(pe["F"].mean()), "phi_clean": float(pe["F"][clean].mean()) if clean.any() else float("nan"),
            "prevalence": float(pe["defective"].mean()),
            "img_recall_rho_b": float(1 - pe["b"][pe["defective"]].mean()) if pe["defective"].any() else float("nan"),
            "size_classes": size_class_recall(pe)}


def bad_ranks(bad: np.ndarray, order: np.ndarray) -> np.ndarray:
    """0-based positions (in review order) of the bad images; K'(k) = K - #(ranks < k)."""
    pos = np.empty(len(order), np.int64)
    pos[np.asarray(order)] = np.arange(len(order))
    return np.sort(pos[np.flatnonzero(bad)])


def kprime_at(ranks: np.ndarray, ks: Sequence[int]) -> List[int]:
    ranks = np.asarray(ranks)
    return [int(len(ranks) - np.searchsorted(ranks, k, side="left")) for k in ks]


class N90Cache:
    """n90(K') for one (N, K_target, delta, power): exact fixed-n audit size (certify.plan_n)."""

    def __init__(self, N: int, K_target: int, delta: float = 0.05, power: float = 0.9):
        self.N, self.K_target, self.delta, self.power = N, K_target, delta, power
        self.c: Dict[int, int] = {}

    def __call__(self, kp: int) -> int:
        from .certify import plan_n
        kp = int(kp)
        if kp not in self.c:
            n, _ = plan_n(self.N, kp, self.K_target, self.delta, self.power)
            self.c[kp] = self.N if n is None else n
        return self.c[kp]


def cost_curve(ranks: np.ndarray, N: int, n90: N90Cache, ratio: float) -> Dict[str, float]:
    """min_k Cost(k) = k*c_review + n90(K'(k))*c_audit in units of c_audit (c_review = ratio*c_audit).

    K'(k) only drops right after a bad image is reviewed, so the optimum is at k = 0 or k = rank+1.
    """
    K = len(ranks)
    ks = np.concatenate([[0], np.asarray(ranks) + 1])
    kp = K - np.arange(K + 1)                       # K'(ks[j]) = K - j
    costs = np.array([k * ratio + n90(p) for k, p in zip(ks, kp)], np.float64)
    j = int(np.argmin(costs))
    return {"k_star": int(ks[j]), "K_prime_at_k_star": int(kp[j]), "n90_at_k_star": int(n90(kp[j])),
            "cost_min": float(costs[j]), "cost_k0": float(costs[0]), "n90_k0": int(n90(K)),
            "cost_full_review": float(N * ratio)}


def kprime_report(b: np.ndarray, risk: np.ndarray, zeta_targets=(0.02, 0.05), ratios=(1.0, 0.25),
                  fracs=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5), delta: float = 0.05, n_random: int = 20,
                  seed: int = 0) -> Dict:
    """K'(k) for risk / random / oracle review order and the closed-form cost (00 §4.5, perfect reviewer)."""
    N, K = len(b), int(b.sum())
    rng = np.random.RandomState(seed)
    orders = {"risk": np.argsort(-np.asarray(risk), kind="stable"), "oracle": np.argsort(-b, kind="stable")}
    ranks = {k: bad_ranks(b, o) for k, o in orders.items()}
    rand_ranks = [bad_ranks(b, rng.permutation(N)) for _ in range(n_random)]
    ks = [int(round(f * N)) for f in fracs]
    out = {"N": N, "K": K, "fracs": list(fracs), "risk_bad_ranks": ranks["risk"].tolist(),
           "K_prime": {"risk": kprime_at(ranks["risk"], ks), "oracle": kprime_at(ranks["oracle"], ks),
                       "random_mean": np.mean([kprime_at(r, ks) for r in rand_ranks], 0).tolist()},
           # area under the normalised K'(k)/K curve over k/N in [0, 1] (lower = misses found earlier)
           "auc_K_prime": {"risk": float(np.mean(ranks["risk"]) / N) if K else float("nan"),
                           "random": float(np.mean([np.mean(r) for r in rand_ranks]) / N) if K else float("nan"),
                           "oracle": float(np.mean(ranks["oracle"]) / N) if K else float("nan")},
           "cost": {}}
    for zt in zeta_targets:
        n90 = N90Cache(N, int(np.floor(zt * N)), delta)
        for ratio in ratios:
            key = f"zeta*={zt}/c_review={ratio}"
            out["cost"][key] = {name: cost_curve(ranks[name], N, n90, ratio) for name in ("risk", "oracle")}
            rc = [cost_curve(r, N, n90, ratio) for r in rand_ranks]
            out["cost"][key]["random"] = {"cost_min": float(np.mean([c["cost_min"] for c in rc])),
                                          "k_star": float(np.mean([c["k_star"] for c in rc])),
                                          "cost_k0": rc[0]["cost_k0"], "n90_k0": rc[0]["n90_k0"]}
    return out


def strata_report(b: np.ndarray, pe: Dict[str, np.ndarray], risk_ref: np.ndarray, n_strata: int = 5) -> Dict:
    """Image-level recall (1 - share of defective images with a miss) per risk quintile of ``risk_ref``.

    Stratum 0 = lowest risk. Used for gate G1 with the round-0 risk as fixed reference.
    """
    order = np.argsort(np.asarray(risk_ref), kind="stable")
    strata = np.empty(len(order), np.int64)
    strata[order] = (np.arange(len(order)) * n_strata) // len(order)
    out = []
    for s in range(n_strata):
        sel = strata == s
        dsel = sel & pe["defective"]
        out.append({"n": int(sel.sum()), "n_defective": int(dsel.sum()), "bad": int(b[sel].sum()),
                    "img_recall": float(1 - b[dsel].mean()) if dsel.any() else float("nan")})
    return {"strata": out}


def phase1_round(rr: Dict, gts_U, risk_ref: Optional[np.ndarray] = None, tol: int = 2,
                 merge_radius: int = 2) -> Dict:
    pe = pool_errors(rr["mask_U"], gts_U, tol, merge_radius)
    out = {"pool": pool_summary(pe)}
    risk = rr.get("risk_U")
    if risk is not None:
        risk = np.asarray(risk, np.float64)
        out["kprime"] = kprime_report(pe["b"], risk)
        # where do the misses sit in this round's own risk ranking? (hidden = below median risk)
        r01 = np.argsort(np.argsort(risk)) / max(1, len(risk) - 1)
        out["share_bad_below_median_risk"] = float(np.mean(r01[pe["b"] > 0] < 0.5)) if pe["b"].any() else float("nan")
    if risk_ref is not None:
        out["strata_ref"] = strata_report(pe["b"], pe, risk_ref)
    return out


def threshold_gap(round_eval: Dict) -> Dict:
    """Gate G2: OOF thresholds vs. thresholds chosen on U with GT (oracle), and the metric gap."""
    th, tho = round_eval["thresholds"], round_eval["U_oracle_threshold"]["thresholds"]
    u, uo = round_eval["U"], round_eval["U_oracle_threshold"]
    return {"t_seed_oof": th["t_seed"], "t_seed_oracle": tho["t_seed"],
            "t_ext_oof": th["t_ext"], "t_ext_oracle": tho["t_ext"],
            "min_area_oof": th["min_area"], "min_area_oracle": tho["min_area"],
            "tol_f1_gap": float(uo["tol_f1"] - u["tol_f1"]),
            "defect_recall_gap": float(uo["defect_recall"] - u["defect_recall"]),
            "defect_precision_gap": float(uo["defect_precision"] - u["defect_precision"])}


def phase1_report(res: Dict, round_evals: Sequence[Dict], gts_U, tol: int = 2, merge_radius: int = 2) -> Dict:
    """All Phase-1 quantities for every round: pool zeta/phi/size classes, K'(k) and cost, G1 strata
    (fixed round-0 risk quintiles), G2 threshold gap."""
    rounds = res["rounds"]
    for j, rr in enumerate(rounds):        # runs from before per-round risk: only the last round has it
        if "risk_U" not in rr and j == len(rounds) - 1 and "risk_U" in res:
            rr["risk_U"] = res["risk_U"]
    ref = rounds[0].get("risk_U")
    ref = None if ref is None else np.asarray(ref, np.float64)
    out = {"rounds": [], "size_classes_px": [list(c) for c in SIZE_CLASSES]}
    for rr, ev in zip(rounds, round_evals):
        r = phase1_round(rr, gts_U, ref, tol, merge_radius)
        r["g2_threshold_gap"] = threshold_gap(ev)
        r["seg_f1"] = ev["U"]["px_f1"]     # SegF1 = pooled pixel F1 at the selected threshold
        out["rounds"].append(r)
    return out
