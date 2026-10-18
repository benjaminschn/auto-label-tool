"""Label-free quality estimation of individual pseudo-labels and of the whole pseudo-labeled pool.

All quantities are computed from (calibrated) probabilities, ensemble disagreement and the
normality model - no ground truth needed. They are

* used directly as the tool's *self-assessment* (expected precision / recall / Dice),
* used to rank images for human review (triage), and
* used as the "predictions" in the audit-based estimators of :mod:`autolabel.ppi`, which turn a small
  random human audit into confidence intervals for the true label quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import ndimage as ndi

from . import metrics as M
from .calibration import ScoreCalibrator
from .postprocess import candidates


@dataclass
class CandidateModel:
    """P(candidate region is a real defect | its max probability), fitted on out-of-fold predictions."""
    t_low: float
    tol: int = 2
    merge_radius: int = 2
    cal: ScoreCalibrator = field(default_factory=ScoreCalibrator)
    no_candidate_rate: float = 0.0     # share of real defects without any candidate (invisible to the model)

    def fit(self, probs: Sequence[np.ndarray], gts: Sequence[np.ndarray]):
        scores, events, n_gt, n_gt_nocand = [], [], 0, 0
        for p, g in zip(probs, gts):
            lab, mx, _ = candidates(p, self.t_low)
            g_d = M.dilate(g, self.tol)
            if len(mx):
                real = np.asarray(ndi.maximum(g_d.astype(np.uint8), lab, index=np.arange(1, len(mx) + 1))) > 0
                scores.extend(mx.tolist())
                events.extend(real.astype(float).tolist())
            glab, ng = M.label_components(g, self.merge_radius)
            if ng:
                cand_d = M.dilate(lab > 0, self.tol)
                hit = np.asarray(ndi.maximum(cand_d.astype(np.uint8), glab, index=np.arange(1, ng + 1))) > 0
                n_gt += ng
                n_gt_nocand += int((~hit).sum())
        self.cal.fit(scores, events)
        self.no_candidate_rate = n_gt_nocand / n_gt if n_gt else 0.0
        return self


def image_quality(prob: np.ndarray, qprob: np.ndarray, mask: np.ndarray, cand: CandidateModel,
                  member_probs: Optional[np.ndarray] = None, anomaly: Optional[np.ndarray] = None,
                  t_ext: float = 0.5) -> Dict[str, float]:
    """Self-assessment features of one pseudo-label.

    prob: ensemble mean probability, qprob: calibrated probability, mask: final label.
    """
    f: Dict[str, float] = {}
    y = mask.astype(bool)
    q = qprob.astype(np.float64)
    tp = float((q * y).sum())
    f["exp_tp"] = tp
    f["exp_fp"] = float(y.sum()) - tp
    f["exp_fn"] = float((q * ~y).sum())
    f["n_pred"] = float(y.sum())
    den = q.sum() + y.sum()
    f["exp_dice"] = float(2 * tp / den) if den > 1e-6 else 1.0
    f["p_max"] = float(prob.max())
    f["mean_conf_in_mask"] = float(prob[y].mean()) if y.any() else float("nan")
    f["uncertain_mass"] = float((q * (1 - q)).sum())

    # defect-level self-assessment from candidate regions
    lab, mx, _ = candidates(prob, cand.t_low)
    if len(mx):
        p_real = cand.cal(mx)
        selected = np.asarray(ndi.maximum(y.astype(np.uint8), lab, index=np.arange(1, len(mx) + 1))) > 0
    else:
        p_real, selected = np.zeros(0), np.zeros(0, bool)
    f["exp_hits"] = float(p_real[selected].sum())
    f["exp_missed"] = float(p_real[~selected].sum())
    k = 1.0 / max(1e-6, 1.0 - cand.no_candidate_rate)
    f["exp_defects"] = (f["exp_hits"] + f["exp_missed"]) * k
    f["exp_false_comps"] = float((1 - p_real[selected]).sum())
    f["n_pred_comps"] = float(selected.sum())
    f["p_missed_defect"] = float(1 - np.prod(1 - p_real[~selected])) if (~selected).any() else 0.0
    f["p_has_defect"] = float(1 - np.prod(1 - p_real)) if len(p_real) else 0.0

    if member_probs is not None and len(member_probs) > 1:
        mp = member_probs.astype(np.float32)
        ms = mp >= t_ext
        dices = []
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                dices.append(M.per_image_dice(ms[i], ms[j]))
        f["member_dice"] = float(np.mean(dices))
        f["member_std_max"] = float(mp.std(0).max())
        f["member_pmax_std"] = float(mp.reshape(len(mp), -1).max(1).std())
    if anomaly is not None:
        a = anomaly.astype(np.float32)
        outside = ~M.dilate(y, 6)
        f["anomaly_max"] = float(a.max())
        f["anomaly_out"] = float(a[outside].max()) if outside.any() else 0.0
    return f


def true_quality(pred: np.ndarray, gt: np.ndarray, tol: int = 2, merge_radius: int = 2) -> Dict[str, float]:
    s = M.image_stats(pred, gt, tol, merge_radius)
    s["missed_defect"] = float(s["gt_comp"] - s["gt_comp_hit"] > 0)
    s["bad_label"] = float(s["missed_defect"] > 0 or s["dice_tol"] < 0.5)
    return s


def rank01(x: np.ndarray) -> np.ndarray:
    """Percentile rank in [0,1] (ties averaged)."""
    from scipy.stats import rankdata
    x = np.asarray(x, np.float64)
    x = np.where(np.isnan(x), np.nanmin(x) if np.isfinite(x).any() else 0.0, x)
    return (rankdata(x) - 1) / max(1, len(x) - 1)


def review_risk(features: List[Dict[str, float]], weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Combine self-assessment signals into one review priority (higher = review first).

    Rank averaging keeps the combination scale-free and robust; weights can be tuned per application.
    """
    w = weights or {"inv_exp_dice": 1.0, "p_missed_defect": 1.0, "anomaly_out": 1.0, "inv_member_dice": 1.0}
    cols = {
        "inv_exp_dice": [-f["exp_dice"] for f in features],
        "p_missed_defect": [f["p_missed_defect"] for f in features],
        "anomaly_out": [f.get("anomaly_out", np.nan) for f in features],
        "inv_member_dice": [-f.get("member_dice", np.nan) for f in features],
        "exp_false_comps": [f["exp_false_comps"] for f in features],
    }
    tot, wsum = np.zeros(len(features)), 0.0
    for k, wk in w.items():
        v = np.asarray(cols[k], np.float64)
        if wk == 0 or np.all(np.isnan(v)):
            continue
        tot += wk * rank01(v)
        wsum += wk
    return tot / max(wsum, 1e-9)
