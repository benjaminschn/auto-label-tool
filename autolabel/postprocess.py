"""From probability maps to label masks: hysteresis thresholding and threshold selection."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

from . import metrics as M


def hysteresis(prob: np.ndarray, t_seed: float, t_ext: float, min_area: int = 0) -> np.ndarray:
    """Keep regions ``prob >= t_ext`` that contain at least one pixel ``>= t_seed`` (and ``>= min_area`` px).

    Faint parts of a scratch connected to a clearly visible part are kept, isolated faint responses
    (texture, reflections) are dropped.
    """
    t_ext = min(t_ext, t_seed)
    ext = prob >= t_ext
    if not ext.any():
        return ext
    lab, n = ndi.label(ext, structure=np.ones((3, 3)))
    mx = ndi.maximum(prob, lab, index=np.arange(1, n + 1))
    keep = np.asarray(mx) >= t_seed
    if min_area > 0:
        area = ndi.sum(np.ones_like(prob), lab, index=np.arange(1, n + 1))
        keep &= np.asarray(area) >= min_area
    lut = np.concatenate([[False], keep])
    return lut[lab]


def candidates(prob: np.ndarray, t_low: float):
    """Connected components of ``prob >= t_low`` with their max probability."""
    lab, n = ndi.label(prob >= t_low, structure=np.ones((3, 3)))
    if n == 0:
        return lab, np.zeros(0, np.float32), np.zeros(0, np.float32)
    idx = np.arange(1, n + 1)
    mx = np.asarray(ndi.maximum(prob, lab, index=idx), np.float32)
    area = np.asarray(ndi.sum(np.ones_like(prob, dtype=np.float32), lab, index=idx), np.float32)
    return lab, mx, area


def select_thresholds(probs: Sequence[np.ndarray], gts: Sequence[np.ndarray], tol: int = 2,
                      merge_radius: int = 2, seed_grid=None, ext_ratios=(1.0, 0.7, 0.5, 0.35),
                      min_areas=(0, 4, 12), objective: str = "tol_f1",
                      min_defect_recall: float = 0.0) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Grid search of (t_seed, t_ext, min_area) on out-of-fold predictions.

    ``objective`` is maximised subject to ``defect_recall >= min_defect_recall`` (if feasible).
    """
    if seed_grid is None:
        seed_grid = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    best, best_val, best_m = None, -1.0, None
    fallback, fb_rec = None, -1.0
    infos = [M.GTInfo(g, tol, merge_radius) for g in gts]
    for ts in seed_grid:
        for r in ext_ratios:
            for a in min_areas:
                preds = [hysteresis(p, ts, ts * r, a) for p in probs]
                m = M.summarize(preds, gts, tol, merge_radius, infos=infos)
                val = m.get(objective, 0.0)
                val = 0.0 if val != val else val
                rec = m["defect_recall"] if m["defect_recall"] == m["defect_recall"] else 1.0
                if rec >= min_defect_recall and val > best_val:
                    best, best_val, best_m = dict(t_seed=ts, t_ext=ts * r, min_area=a), val, m
                if rec > fb_rec or (rec == fb_rec and fallback is not None and val > fallback[1]):
                    fallback, fb_rec = ((dict(t_seed=ts, t_ext=ts * r, min_area=a), val, m)), rec
    if best is None:
        best, best_val, best_m = fallback
    return best, best_m
