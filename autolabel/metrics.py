"""Segmentation quality metrics tailored to small / thin / rare defects.

Three levels are reported because they answer different questions:

* pixel level  - how accurate are the label masks (exact and with a boundary tolerance of ``tol`` px,
  the usual protocol for thin structures where the defect border is subjective)
* defect level - how many real defects (connected components) are found / how many predicted defects
  are spurious. Missed defects are the most critical error for us.
* image level  - does the label say "defective" for images that contain a defect.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import cv2
import numpy as np
from scipy import ndimage as ndi

_DISKS: Dict[int, np.ndarray] = {}


def disk(r: int) -> np.ndarray:
    if r not in _DISKS:
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        _DISKS[r] = (x * x + y * y) <= r * r
    return _DISKS[r]


def dilate(m: np.ndarray, r: int) -> np.ndarray:
    if r <= 0 or not m.any():
        return m.astype(bool)
    return cv2.dilate(m.astype(np.uint8), disk(r).astype(np.uint8)) > 0


def label_components(m: np.ndarray, merge_radius: int = 0):
    """8-connected components. With ``merge_radius`` > 0, fragments closer than that are one defect."""
    if merge_radius > 0 and m.any():
        lab, n = ndi.label(dilate(m, merge_radius), structure=np.ones((3, 3)))
        return lab * m, n
    return ndi.label(m, structure=np.ones((3, 3)))


class GTInfo:
    """Prediction-independent ground-truth quantities, cached when many predictions are scored."""

    def __init__(self, gt: np.ndarray, tol: int = 2, merge_radius: int = 2):
        self.gt = gt.astype(bool)
        self.gt_d = dilate(self.gt, tol)
        self.glab, self.ng = label_components(self.gt, merge_radius)


def image_stats(pred: np.ndarray, gt: np.ndarray, tol: int = 2, merge_radius: int = 2,
                info: Optional[GTInfo] = None) -> Dict[str, float]:
    """All counts needed for dataset-level aggregation of one image."""
    pred = pred.astype(bool)
    info = info or GTInfo(gt, tol, merge_radius)
    gt = info.gt
    tp = float(np.logical_and(pred, gt).sum())
    s = {
        "tp": tp, "fp": float(pred.sum()) - tp, "fn": float(gt.sum()) - tp,
        "n_pred": float(pred.sum()), "n_gt": float(gt.sum()),
    }
    gt_d = info.gt_d
    pr_d = dilate(pred, tol)
    s["tp_pred_tol"] = float(np.logical_and(pred, gt_d).sum())   # predicted pixels close to a defect
    s["tp_gt_tol"] = float(np.logical_and(gt, pr_d).sum())       # defect pixels close to a prediction
    glab, ng = info.glab, info.ng
    plab, npr = label_components(pred, merge_radius)
    if ng:
        hit = ndi.maximum(pr_d.astype(np.uint8), glab, index=np.arange(1, ng + 1))
        s["gt_comp"], s["gt_comp_hit"] = float(ng), float(np.sum(hit))
    else:
        s["gt_comp"], s["gt_comp_hit"] = 0.0, 0.0
    if npr:
        hit = ndi.maximum(gt_d.astype(np.uint8), plab, index=np.arange(1, npr + 1))
        s["pred_comp"], s["pred_comp_hit"] = float(npr), float(np.sum(hit))
    else:
        s["pred_comp"], s["pred_comp_hit"] = 0.0, 0.0
    s["img_gt"] = float(gt.any())
    s["img_pred"] = float(pred.any())
    s["dice"] = per_image_dice(pred, gt)
    s["dice_tol"] = per_image_dice_tol(s)
    return s


def per_image_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    a, b = pred.sum(), gt.sum()
    if a + b == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / (a + b))


def per_image_dice_tol(s: Dict[str, float]) -> float:
    if s["n_pred"] + s["n_gt"] == 0:
        return 1.0
    p = s["tp_pred_tol"] / s["n_pred"] if s["n_pred"] else 0.0
    r = s["tp_gt_tol"] / s["n_gt"] if s["n_gt"] else 0.0
    return float(2 * p * r / (p + r)) if p + r > 0 else 0.0


def _div(a: float, b: float, empty: float = float("nan")) -> float:
    return float(a / b) if b > 0 else empty


_KEYS = ("tp", "fp", "fn", "n_pred", "n_gt", "tp_pred_tol", "tp_gt_tol", "gt_comp", "gt_comp_hit",
         "pred_comp", "pred_comp_hit")


def aggregate(stats: Iterable[Dict[str, float]]) -> Dict[str, float]:
    stats = list(stats)
    if not stats:
        return {}
    S = {k: float(sum(s[k] for s in stats)) for k in _KEYS}
    out = {}
    out["px_precision"] = _div(S["tp"], S["tp"] + S["fp"])
    out["px_recall"] = _div(S["tp"], S["tp"] + S["fn"])
    out["px_f1"] = _div(2 * S["tp"], 2 * S["tp"] + S["fp"] + S["fn"])
    out["px_iou"] = _div(S["tp"], S["tp"] + S["fp"] + S["fn"])
    p = _div(S["tp_pred_tol"], S["n_pred"])
    r = _div(S["tp_gt_tol"], S["n_gt"])
    out["tol_precision"], out["tol_recall"] = p, r
    out["tol_f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    out["defect_recall"] = _div(S["gt_comp_hit"], S["gt_comp"])
    out["defect_precision"] = _div(S["pred_comp_hit"], S["pred_comp"], empty=1.0)
    out["defect_f1"] = (2 * out["defect_recall"] * out["defect_precision"] /
                        (out["defect_recall"] + out["defect_precision"])
                        if out["defect_recall"] + out["defect_precision"] > 0 else 0.0)
    pos = [s for s in stats if s["img_gt"] > 0]
    neg = [s for s in stats if s["img_gt"] == 0]
    out["img_recall"] = _div(sum(s["img_pred"] for s in pos), len(pos))
    out["img_specificity"] = _div(sum(1 - s["img_pred"] for s in neg), len(neg))
    out["mean_dice"] = float(np.mean([s["dice"] for s in stats]))
    out["mean_dice_tol"] = float(np.mean([s["dice_tol"] for s in stats]))
    out["mean_dice_tol_defective"] = float(np.mean([s["dice_tol"] for s in pos])) if pos else float("nan")
    out["n_images"] = float(len(stats))
    out["n_defective"] = float(len(pos))
    return out


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels).astype(int)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _pixel_histograms(probs, gts, bins: int = 2000):
    pos = np.zeros(bins, np.float64)
    neg = np.zeros(bins, np.float64)
    for p, g in zip(probs, gts):
        idx = np.clip((np.asarray(p, np.float32).ravel() * bins).astype(np.int64), 0, bins - 1)
        g = np.asarray(g, bool).ravel()
        pos += np.bincount(idx[g], minlength=bins)
        neg += np.bincount(idx[~g], minlength=bins)
    return pos, neg


def pixel_auroc(probs, gts, bins: int = 2000) -> float:
    """Pixel-level AUROC over all pixels of all images (histogram based, exact up to binning)."""
    pos, neg = _pixel_histograms(probs, gts, bins)
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    # P(score_pos > score_neg) + 0.5 P(tie)
    neg_below = np.cumsum(neg) - neg
    return float((pos * (neg_below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def average_precision_px(probs, gts, bins: int = 2000) -> float:
    """Pixel-level average precision (area under the PR curve), histogram based."""
    pos, neg = _pixel_histograms(probs, gts, bins)
    if pos.sum() == 0:
        return float("nan")
    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / pos.sum()
    drec = np.diff(np.concatenate([[0.0], rec]))
    return float((prec * drec).sum())


def summarize(preds, gts, tol: int = 2, merge_radius: int = 2, probs: Optional[list] = None,
              infos: Optional[list] = None) -> Dict[str, float]:
    infos = infos or [None] * len(gts)
    stats = [image_stats(p, g, tol, merge_radius, inf) for p, g, inf in zip(preds, gts, infos)]
    out = aggregate(stats)
    if probs is not None:
        out["img_auroc"] = auroc(np.array([p.max() for p in probs]), np.array([g.any() for g in gts]))
    return out
