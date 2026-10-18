"""Distribution-free guarantees on missed defects.

With cross-fitted (out-of-fold) predictions on the labeled set, every labeled defect gets a detection
score s = max probability within its (tolerance-dilated) area. A defect is detected by hysteresis
thresholding iff s >= t_seed. Under exchangeability of labeled and unlabeled images, choosing
t_seed from the empirical distribution of these scores gives

* marginal (split-conformal) guarantee:  P(new defect detected) >= 1 - alpha
* PAC guarantee (Clopper-Pearson):        P(miss rate on the unlabeled pool <= alpha) >= 1 - delta

Both get honest about tiny label sets: with too few labeled defects the only certifiable threshold is
"flag everything" (t=0), which is reported instead of a fake guarantee.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import beta as beta_dist

from . import metrics as M


def defect_scores(probs: Sequence[np.ndarray], gts: Sequence[np.ndarray], tol: int = 2,
                  merge_radius: int = 2) -> np.ndarray:
    """Max predicted probability inside each ground-truth defect (dilated by ``tol``)."""
    out: List[float] = []
    for p, g in zip(probs, gts):
        lab, n = M.label_components(g, merge_radius)
        if n == 0:
            continue
        if tol > 0:
            # grow labels into a tolerance band (nearest label wins)
            dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
            grown = lab[iy, ix]
            grown[dist > tol] = 0
            lab = grown
        out.extend(np.asarray(ndi.maximum(p, lab, index=np.arange(1, n + 1)), np.float64).tolist())
    return np.array(out)


def image_scores(probs: Sequence[np.ndarray], gts: Sequence[np.ndarray]) -> np.ndarray:
    return np.array([float(np.max(p)) for p, g in zip(probs, gts) if g.any()])


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """Largest t with P(s_new >= t) >= 1 - alpha (split conformal, lower tail)."""
    s = np.sort(np.asarray(scores))
    n = len(s)
    k = int(math.floor(alpha * (n + 1)))
    if k < 1:
        return 0.0
    return float(s[k - 1])


def clopper_pearson_upper(m: int, n: int, delta: float) -> float:
    if n == 0:
        return 1.0
    if m >= n:
        return 1.0
    return float(beta_dist.ppf(1 - delta, m + 1, n - m))


def pac_threshold(scores: np.ndarray, alpha: float, delta: float = 0.1) -> float:
    """Largest t such that the (1-delta) upper confidence bound of the miss rate is <= alpha."""
    s = np.sort(np.asarray(scores))
    n = len(s)
    best = 0.0
    for m in range(n + 1):
        t = s[m] if m < n else 1.0 + 1e-9
        # m = number of labeled defects with score < t (ties at s[m] are detected)
        if clopper_pearson_upper(m, n, delta) <= alpha:
            best = float(t) if m < n else best
        else:
            break
    return best


def certified_recall(scores: np.ndarray, t: float, delta: float = 0.1):
    """Recall guarantees implied by threshold t: (marginal conformal bound, PAC lower bound)."""
    s = np.asarray(scores)
    n = len(s)
    m = int(np.sum(s < t))
    marginal = 1.0 - (m + 1) / (n + 1) if n else 0.0
    pac = 1.0 - clopper_pearson_upper(m, n, delta)
    return marginal, pac
