"""Audit-based, APPROXIMATE estimates of pseudo-label quality (not valid for certificates).

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

APPROXIMATE ONLY. All intervals in this module (``estimate_*`` and ``classical_*``) are asymptotic:
t quantile, delta method for ratios, and the finite-population correction of simple random sampling
without replacement (SRS) with a sample size n fixed in advance. They are NOT valid under PPS / weighted
sampling, under optional stopping or monitoring of a growing audit, or when lambda or the self-assessment
are recalibrated on the same audit, and they carry no finite-sample guarantee. With rare defects they
under-cover. In the simulations of docs/research/04_guarantees_measurement.md §3 the one-sided 95%
bounds covered 0.88-1.00 (PPI, lam=1) and 0.66-0.94 (audit-only) for the defect recall (S1), and
0.83-0.98 (PPI) and 0.23-0.89 (audit-only) for the missed-defect image rate (S2); a bad (heavy-tailed)
predictor gave 0.88 at n=50 (docs/research/07_red_team.md, F5). Use them for
monitoring and rough estimates only, never as a certificate. Design-based certificates are in
autolabel/certify.py (exact hypergeometric bound for a registered fixed n and uniform sampling without
replacement, anytime-valid PPR confidence sequence); IPW betting bounds with predictable bet caps for
non-uniform designs exist only as prototypes in docs/research/sims/.

When no interval can be formed (empty denominator), the ratio functions return the trivial interval
``(nan, lo, hi)`` with ``(lo, hi) = bounds`` (default ``(0.0, 1.0)``, i.e. ratios such as recall or
precision with 0 <= a <= b) instead of NaN bounds: "no information", not an error.
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
    """Mean of y over the pool (audit + rest). ``lam=1`` -> plain PPI (default), ``None`` -> tuned.

    APPROXIMATE: t interval with SRS finite-population correction, fixed n; asymptotic, under-covers with
    rare events (04 S2) and is not valid for certificates (see module docstring).
    """
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
    """Audit-only mean with t interval (SRS, fixed n). APPROXIMATE, not valid for certificates."""
    y = np.asarray(y_audit, np.float64)
    n = len(y)
    fpc = 1.0 if not n_total else (1 - n / n_total)
    h = _q(alpha, n) * np.sqrt(np.var(y, ddof=1) / n * fpc)
    return float(y.mean()), float(y.mean() - h), float(y.mean() + h)


def estimate_ratio(a_audit, b_audit, fa_audit, fb_audit, fa_rest, fb_rest, alpha: float = 0.1,
                   lam: Optional[float] = 1.0, *, bounds: Tuple[float, float] = (0.0, 1.0)
                   ) -> Tuple[float, float, float]:
    """sum(a)/sum(b) over the pool, e.g. recall = sum TP / sum (TP + FN).

    a, b: true per-image counts on the audit; fa, fb: the tool's predicted counts (audit and rest).
    ``lam=1`` is plain PPI (default: best coverage with small audits); ``lam=None`` tunes it on the
    linearised ratio residual (PPI++, narrower but under-covers for n < ~50 because lambda is fitted on
    the same audit); ``lam=0`` reproduces the classical audit-only ratio estimator.

    APPROXIMATE: delta method + t quantile + SRS finite-population correction with fixed n. Asymptotic
    only; with rare defects and small audits it under-covers (04 S1: 0.88-1.00 for lam=1, 0.66-0.94
    audit-only; 07 F5: 0.88 at n=50 with a heavy-tailed predictor). Under PPS sampling the unweighted
    estimator is biased (07 F5 (ii)); there is no IPW variant here. Not valid for certificates.

    If the estimated denominator total is <= 0, no interval can be formed and the trivial interval
    ``(nan, bounds[0], bounds[1])`` is returned (default ``(nan, 0.0, 1.0)``, for ratios in [0, 1]).
    """
    A, B, FA, FB, RA, RB = (np.asarray(v, np.float64) for v in (a_audit, b_audit, fa_audit, fb_audit,
                                                              fa_rest, fb_rest))
    n, M = len(A), len(A) + len(RA)
    trivial = (float("nan"), float(bounds[0]), float(bounds[1]))
    if B.sum() <= 0 and FB.sum() + RB.sum() <= 0:
        return trivial
    if lam is None:
        r0 = A.sum() / B.sum() if B.sum() > 0 else (FA.sum() + RA.sum()) / (FB.sum() + RB.sum())
        lam = _tuned_lambda(A - r0 * B, FA - r0 * FB)
        lam = float(np.clip(lam, 0.0, 1.0))
    ta = lam * (FA.sum() + RA.sum()) + M * (A - lam * FA).mean()
    tb = lam * (FB.sum() + RB.sum()) + M * (B - lam * FB).mean()
    if tb <= 0:
        return trivial
    r = ta / tb
    if n < 2:
        return float(r), float("-inf"), float("inf")
    d = (A - lam * FA) - r * (B - lam * FB)
    se = np.sqrt(np.var(d, ddof=1) / n * (1 - n / M)) * M / tb
    h = _q(alpha, n) * se
    return float(r), float(r - h), float(r + h)


def classical_ratio(a_audit, b_audit, alpha: float = 0.1, n_total: Optional[int] = None, *,
                    bounds: Tuple[float, float] = (0.0, 1.0)):
    """Audit-only ratio sum(a)/sum(b) with delta-method t interval (SRS, fixed n).

    APPROXIMATE, not valid for certificates (04 S1: 0.66-0.94 coverage of the 95% LCB with rare
    defects). If sum(b) <= 0 on the audit, returns the trivial interval ``(nan, bounds[0], bounds[1])``.
    """
    A, B = np.asarray(a_audit, np.float64), np.asarray(b_audit, np.float64)
    n = len(A)
    if B.sum() <= 0:
        return float("nan"), float(bounds[0]), float(bounds[1])
    r = A.sum() / B.sum()
    fpc = 1.0 if not n_total else (1 - n / n_total)
    se = np.sqrt(np.var(A - r * B, ddof=1) / n * fpc) / B.mean()
    h = _q(alpha, n) * se
    return float(r), float(r - h), float(r + h)
