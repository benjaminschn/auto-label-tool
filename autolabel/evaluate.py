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
