"""Audit-based, statistically valid estimates of pseudo-label quality.

Setting: a pool of M auto-labeled images. For every image the tool predicts its own error counts
(e.g. expected true positives / false negatives from calibrated probabilities). A small *random*
audit sample of n images is checked by a human. The estimators below combine both:

    total(y) ~= sum_{all} lam*f_i  +  M * mean_{audit}(y_i - lam*f_i)

i.e. prediction-powered inference (Angelopoulos et al., Science 2023) in its finite-population form
(the model-assisted difference / regression estimator of survey sampling). The audit removes any bias of
the self-assessment, the predictions on all images remove most of the variance. With ``lam`` tuned on
the audit (PPI++ / regression estimator) the interval is never meaningfully wider than the classical
audit-only interval, even if the self-assessment is useless; with small audits the untuned version
(``lam=1``, default) has better coverage.

The target is the quality of *this* pool of labels, hence finite-population corrections.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import t as t_dist


def _q(alpha: float, n: int) -> float:
    return float(t_dist.ppf(1 - alpha / 2, max(1, n - 1)))


def _tuned_lambda(y: np.ndarray, f: np.ndarray) -> float:
    if len(y) < 3 or np.var(f) <= 0:
        return 1.0
    return float(np.clip(np.cov(y, f, ddof=1)[0, 1] / np.var(f, ddof=1), 0.0, 1.5))


def estimate_mean(y_audit, f_audit, f_rest, alpha: float = 0.1, lam: Optional[float] = 1.0
                  ) -> Tuple[float, float, float]:
    """Mean of y over the pool (audit + rest). ``lam=1`` -> plain PPI (default), ``None`` -> tuned."""
    y, fa, fr = (np.asarray(v, np.float64) for v in (y_audit, f_audit, f_rest))
    n, M = len(y), len(y) + len(fr)
    if lam is None:
        lam = _tuned_lambda(y, fa)
    f_all = np.concatenate([fa, fr])
    d = y - lam * fa
    est = lam * f_all.mean() + d.mean()
    se = np.sqrt(np.var(d, ddof=1) / n * (1 - n / M)) if n > 1 else np.inf
    h = _q(alpha, n) * se
    return float(est), float(est - h), float(est + h)


def classical_mean(y_audit, alpha: float = 0.1, n_total: Optional[int] = None):
    y = np.asarray(y_audit, np.float64)
    n = len(y)
    fpc = 1.0 if not n_total else (1 - n / n_total)
    h = _q(alpha, n) * np.sqrt(np.var(y, ddof=1) / n * fpc)
    return float(y.mean()), float(y.mean() - h), float(y.mean() + h)


def estimate_ratio(a_audit, b_audit, fa_audit, fb_audit, fa_rest, fb_rest, alpha: float = 0.1,
                   lam: Optional[float] = 1.0) -> Tuple[float, float, float]:
    """sum(a)/sum(b) over the pool, e.g. recall = sum TP / sum (TP + FN).

    a, b: true per-image counts on the audit; fa, fb: the tool's predicted counts (audit and rest).
    ``lam=1`` is plain PPI (default: best coverage with small audits); ``lam=None`` tunes it on the
    linearised ratio residual (PPI++, narrower but under-covers for n < ~50 because lambda is fitted on
    the same audit); ``lam=0`` reproduces the classical audit-only ratio estimator.
    """
    A, B, FA, FB, RA, RB = (np.asarray(v, np.float64) for v in (a_audit, b_audit, fa_audit, fb_audit,
                                                              fa_rest, fb_rest))
    n, M = len(A), len(A) + len(RA)
    if B.sum() <= 0 and FB.sum() + RB.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    if lam is None:
        r0 = A.sum() / B.sum() if B.sum() > 0 else (FA.sum() + RA.sum()) / (FB.sum() + RB.sum())
        lam = _tuned_lambda(A - r0 * B, FA - r0 * FB)
        lam = float(np.clip(lam, 0.0, 1.0))
    ta = lam * (FA.sum() + RA.sum()) + M * (A - lam * FA).mean()
    tb = lam * (FB.sum() + RB.sum()) + M * (B - lam * FB).mean()
    if tb <= 0:
        return float("nan"), float("nan"), float("nan")
    r = ta / tb
    if n < 2:
        return float(r), float("-inf"), float("inf")
    d = (A - lam * FA) - r * (B - lam * FB)
    se = np.sqrt(np.var(d, ddof=1) / n * (1 - n / M)) * M / tb
    h = _q(alpha, n) * se
    return float(r), float(r - h), float(r + h)


def classical_ratio(a_audit, b_audit, alpha: float = 0.1, n_total: Optional[int] = None):
    A, B = np.asarray(a_audit, np.float64), np.asarray(b_audit, np.float64)
    n = len(A)
    if B.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    r = A.sum() / B.sum()
    fpc = 1.0 if not n_total else (1 - n / n_total)
    se = np.sqrt(np.var(A - r * B, ddof=1) / n * fpc) / B.mean()
    h = _q(alpha, n) * se
    return float(r), float(r - h), float(r + h)
