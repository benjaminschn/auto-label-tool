"""Probability calibration on out-of-fold predictions of the labeled set."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

_EPS = 1e-5


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), _EPS, 1 - _EPS)
    return np.log(p) - np.log1p(-p)


class IsotonicCalibrator:
    """Histogram-binned isotonic calibration (memory-friendly for millions of pixels).

    Bins live on the logit scale so that the rare high-probability region gets enough resolution.
    """

    def __init__(self, n_bins: int = 200):
        self.n_bins = n_bins
        self.x = None
        self.y = None

    def fit(self, probs: Sequence[np.ndarray], labels: Sequence[np.ndarray], weights=None):
        edges = np.linspace(_logit(_EPS * 2), _logit(1 - _EPS * 2), self.n_bins + 1)
        n = np.zeros(self.n_bins)
        k = np.zeros(self.n_bins)
        for i, (p, y) in enumerate(zip(probs, labels)):
            w = 1.0 if weights is None else weights[i]
            b = np.clip(np.searchsorted(edges, _logit(np.asarray(p).ravel())) - 1, 0, self.n_bins - 1)
            yb = np.asarray(y, bool).ravel()
            n += w * np.bincount(b, minlength=self.n_bins)
            k += w * np.bincount(b[yb], minlength=self.n_bins)
        ok = n > 0
        centers = 0.5 * (edges[1:] + edges[:-1])
        # weak Beta(0.5,0.5)-style smoothing towards the raw probability avoids 0/1 plateaus with few samples
        prior = 1 / (1 + np.exp(-centers))
        frac = (k + 2 * prior) / (n + 2)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(centers[ok], frac[ok], sample_weight=n[ok])
        self.x = centers[ok]
        self.y = iso.predict(centers[ok])
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        if self.x is None:
            return np.asarray(p, np.float32)
        return np.interp(_logit(p), self.x, self.y).astype(np.float32)


def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    """Pixel ECE restricted to the region that matters: bins over p in [0,1] weighted by counts."""
    edges = np.linspace(0, 1, n_bins + 1)
    n = np.zeros(n_bins)
    s = np.zeros(n_bins)
    k = np.zeros(n_bins)
    for p, y in zip(probs, labels):
        p = np.asarray(p, np.float64).ravel()
        b = np.clip(np.searchsorted(edges, p) - 1, 0, n_bins - 1)
        n += np.bincount(b, minlength=n_bins)
        s += np.bincount(b, weights=p, minlength=n_bins)
        k += np.bincount(b[np.asarray(y, bool).ravel()], minlength=n_bins)
    ok = n > 0
    return float(np.sum(np.abs(s[ok] - k[ok])) / n.sum())


class ScoreCalibrator:
    """1-D isotonic map score -> P(event), e.g. P(candidate component is a real defect | max prob)."""

    def __init__(self):
        self.iso = None
        self.const = None

    def fit(self, scores, events):
        scores = np.asarray(scores, np.float64)
        events = np.asarray(events, np.float64)
        if len(scores) == 0:
            self.const = 0.5
            return self
        # add pseudo-observations at the extremes so the map is defined everywhere
        s = np.concatenate([scores, [0.0, 1.0]])
        e = np.concatenate([events, [0.0, 1.0]])
        w = np.concatenate([np.ones(len(scores)), [0.5, 0.5]])
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s, e, sample_weight=w)
        return self

    def __call__(self, s):
        if self.iso is None:
            return np.full(np.shape(s), self.const, np.float32)
        return self.iso.predict(np.asarray(s, np.float64)).astype(np.float32)
