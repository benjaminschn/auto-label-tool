"""Defect-type assignment for segmented defects (few-shot) with conformal prediction sets.

The segmentation stage answers "where is a defect"; this stage answers "which type". Every connected
component of a pseudo-label is described by
  * frozen CNN features pooled inside the (dilated) component and in its surrounding box, and
  * shape / contrast descriptors (size, elongation, contrast to the surrounding ring, border contact),
and classified by a nearest-class-mean rule in a standardised feature space fitted on the few labeled
components. Conformal prediction sets (leave-one-image-out calibration on the labeled set) return *all*
plausible types when the evidence is ambiguous, so such defects can be routed to review instead of
silently getting a wrong class.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from .model import create_encoder, normalize


class ComponentFeaturizer:
    def __init__(self, backbone: str = "resnet18d", out_indices=(2, 3), threads: int = 4):
        torch.set_num_threads(threads)
        self.enc = create_encoder(backbone, True, out_indices=out_indices).eval()

    @torch.no_grad()
    def feature_maps(self, img: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None]
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="replicate")
        fs = self.enc(normalize(x))
        size = (h, w)
        return torch.cat([F.interpolate(f, size=fs[0].shape[-2:], mode="bilinear", align_corners=False)
                          for f in fs], 1)[0], fs[0].shape[-2:]

    def describe(self, img: np.ndarray, mask: np.ndarray, merge_radius: int = 2
                 ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Features of every connected component of ``mask``; returns (features, component masks)."""
        lab, n = ndi.label(cv2.dilate(mask.astype(np.uint8), np.ones((2 * merge_radius + 1,) * 2, np.uint8))
                           if merge_radius else mask, structure=np.ones((3, 3)))
        lab = lab * mask
        if n == 0:
            return np.zeros((0, 1)), []
        fmap, (fh, fw) = self.feature_maps(img)
        H, W = mask.shape
        gray = img.astype(np.float32).mean(-1)
        feats, comps = [], []
        for i, sl in enumerate(ndi.find_objects(lab)):
            if sl is None:
                continue
            cm = lab == i + 1
            if cm.sum() == 0:
                continue
            # CNN features pooled inside the component and in its context box
            small = cv2.resize(cv2.dilate(cm.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32),
                               (fw * 1, fh * 1), interpolation=cv2.INTER_AREA)
            small = small[: fmap.shape[1], : fmap.shape[2]]
            if small.sum() < 1e-6:
                yy, xx = np.argwhere(cm).mean(0)
                small = np.zeros_like(small)
                small[min(int(yy / H * fh), fh - 1), min(int(xx / W * fw), fw - 1)] = 1
            wgt = torch.from_numpy(small / small.sum())
            inside = (fmap[:, : wgt.shape[0], : wgt.shape[1]] * wgt).sum((1, 2)).numpy()
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            by0, by1 = int(y0 / H * fh), max(int(np.ceil(y1 / H * fh)), int(y0 / H * fh) + 1)
            bx0, bx1 = int(x0 / W * fw), max(int(np.ceil(x1 / W * fw)), int(x0 / W * fw) + 1)
            ctx = fmap[:, max(0, by0 - 1): by1 + 1, max(0, bx0 - 1): bx1 + 1].mean((1, 2)).numpy()
            # shape / contrast descriptors
            area = cm.sum()
            ys, xs = np.nonzero(cm)
            cov = np.cov(np.vstack([ys, xs])) if area > 2 else np.eye(2)
            ev = np.sort(np.linalg.eigvalsh(cov + 1e-6 * np.eye(2)))
            ring = cv2.dilate(cm.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~cm
            contrast = gray[cm].mean() - (gray[ring].mean() if ring.any() else gray.mean())
            border = float(y0 == 0 or x0 == 0 or y1 == H or x1 == W)
            shape = np.array([np.log1p(area), np.log1p(area / (H * W) * 1e4), 0.5 * np.log((ev[1] + 1) / (ev[0] + 1)),
                              contrast / 32.0, border, (y1 - y0) / H, (x1 - x0) / W])
            feats.append(np.concatenate([inside, ctx, shape * 4.0]))
            comps.append(cm)
        return np.array(feats, np.float32), comps


class NearestMeanClassifier:
    """Nearest class mean in a balanced feature space.

    CNN features are reduced with PCA (fitted on *all* components, labeled and unlabeled - the
    unlabeled ones are free and stabilise the projection) and whitened; shape descriptors are
    standardised. Both blocks get equal weight so that 768 CNN dimensions do not drown 7 shape cues.
    """

    def __init__(self, n_shape: int = 7, n_pca: int = 16):
        self.n_shape, self.n_pca = n_shape, n_pca

    def _embed(self, X):
        cnn, shp = X[:, :-self.n_shape], X[:, -self.n_shape:]
        zc = self.pca.transform((cnn - self.mu_c) / self.sd_c) / self.pca_sd
        zs = (shp - self.mu_s) / self.sd_s
        return np.hstack([zc / np.sqrt(zc.shape[1]), zs / np.sqrt(zs.shape[1])])

    def fit(self, X: np.ndarray, y: Sequence[str], X_ref: Optional[np.ndarray] = None):
        from sklearn.decomposition import PCA
        self.classes = sorted(set(y))
        y = np.asarray(y)
        R = X if X_ref is None or len(X_ref) == 0 else np.vstack([X, X_ref])
        cnn, shp = R[:, :-self.n_shape], R[:, -self.n_shape:]
        self.mu_c, self.sd_c = cnn.mean(0), cnn.std(0) + 1e-6
        self.mu_s, self.sd_s = shp.mean(0), shp.std(0) + 1e-6
        k = int(min(self.n_pca, len(R) - 1, cnn.shape[1]))
        self.pca = PCA(k, random_state=0).fit((cnn - self.mu_c) / self.sd_c)
        self.pca_sd = np.sqrt(self.pca.explained_variance_) + 1e-6
        Z = self._embed(X)
        self.means = np.stack([Z[y == c].mean(0) for c in self.classes])
        d = np.linalg.norm(Z[:, None] - self.means[None], axis=-1)
        self.temp = float(np.median(d)) + 1e-6
        self.X_ref = X_ref
        return self

    def proba(self, X: np.ndarray) -> np.ndarray:
        Z = self._embed(X)
        d = np.linalg.norm(Z[:, None] - self.means[None], axis=-1)
        logits = -d / self.temp * 4.0
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(1, keepdims=True)


class DefectTypeAssigner:
    def __init__(self, alpha: float = 0.1, backbone: str = "resnet18d"):
        self.alpha = alpha
        self.feat = ComponentFeaturizer(backbone)
        self.clf: Optional[NearestMeanClassifier] = None
        self.qhat = 1.0

    def fit(self, images: Sequence[np.ndarray], masks: Sequence[np.ndarray], kinds: Sequence[str],
            X_ref: Optional[np.ndarray] = None):
        """kinds: defect type per labeled image (all its components share it).
        X_ref: optional features of unlabeled components (only used for the PCA projection)."""
        X, y, g = [], [], []
        for j, (img, m, k) in enumerate(zip(images, masks, kinds)):
            if m is None or not m.any():
                continue
            f, _ = self.feat.describe(img, m)
            X.extend(f)
            y.extend([k] * len(f))
            g.extend([j] * len(f))
        X, y, g = np.array(X), np.array(y), np.array(g)
        self.clf = NearestMeanClassifier().fit(X, y, X_ref)
        # leave-one-image-out nonconformity scores for conformal sets
        scores = []
        for j in np.unique(g):
            tr = g != j
            if len(set(y[tr])) < len(self.clf.classes):
                # class would vanish: score with full model (slightly optimistic, rare)
                clf = self.clf
            else:
                clf = NearestMeanClassifier().fit(X[tr], y[tr], X_ref)
            p = clf.proba(X[~tr])
            for row, lab in zip(p, y[~tr]):
                scores.append(1 - row[clf.classes.index(lab)])
        scores = np.sort(scores)
        n = len(scores)
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        self.qhat = float(scores[k - 1]) if k <= n else 1.0
        self.n_cal = n
        return self

    def predict(self, img: np.ndarray, mask: np.ndarray, feats=None):
        f, comps = feats if feats is not None else self.feat.describe(img, mask)
        if not comps:
            return []
        P = self.clf.proba(f)
        out = []
        for row, cm in zip(P, comps):
            sets = [c for c, p in zip(self.clf.classes, row) if 1 - p <= self.qhat]
            out.append({"mask": cm, "type": self.clf.classes[int(row.argmax())], "proba": dict(zip(self.clf.classes, row.tolist())),
                        "prediction_set": sets or [self.clf.classes[int(row.argmax())]]})
        return out
