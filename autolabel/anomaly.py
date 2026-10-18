"""Normality model learned from the *unlabeled* images (PatchCore-style, contamination-robust).

Defects are rare, so the unlabeled pool is a good description of "normal" surface appearance, including
reflections, textures and machining marks. Patch features of a frozen ImageNet backbone are stored in a
memory bank; the anomaly score of a patch is its distance to the k-th nearest bank patch coming from a
*different* image. Excluding the own image and using k>1 makes the model robust to the few real defects
contained in the unlabeled pool: a defect is only "explained" if at least k similar patches exist
elsewhere.

The map is used (i) to select clean backgrounds for defect synthesis and (ii) as an independent
second opinion for quality estimation: anomalous regions not covered by the predicted label are a
strong indicator for a missed defect.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import create_encoder, get_device, normalize, set_threads


class NormalityModel:
    def __init__(self, backbone: str = "wide_resnet50_2", out_indices=(2, 3), proj_dim: int = 256,
                 patches_per_image: int = 64, k: int = 3, blur_sigma: float = 4.0, seed: int = 0,
                 threads: int = 0):
        set_threads(threads)
        self.dev = get_device()
        self.encoder = create_encoder(backbone, True, out_indices=out_indices).eval().to(self.dev)
        self.proj_dim, self.ppi, self.k, self.sigma = proj_dim, patches_per_image, k, blur_sigma
        self.rng = np.random.RandomState(seed)
        dims = sum(self.encoder.feature_info.channels())
        g = torch.Generator().manual_seed(seed)
        self.proj = (torch.randn(dims, proj_dim, generator=g) / np.sqrt(proj_dim)).to(self.dev)
        self.bank: Optional[torch.Tensor] = None
        self.bank_ids: Optional[torch.Tensor] = None
        self.offsets = {}   # image id -> (start, end) of its patches in the bank (contiguous)

    @torch.no_grad()
    def features(self, img: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None].to(self.dev)
        h, w = x.shape[-2:]
        ph, pw = (-h) % 16, (-w) % 16
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        feats = self.encoder(normalize(x))
        size = feats[0].shape[-2:]
        fs = [F.interpolate(F.avg_pool2d(f, 3, 1, 1), size=size, mode="bilinear", align_corners=False)
              for f in feats]
        f = torch.cat(fs, 1)[0].permute(1, 2, 0)            # h' x w' x D
        hs, ws = int(np.ceil(h / 8)), int(np.ceil(w / 8))
        return (f[:hs, :ws] @ self.proj).contiguous()         # crop padding, project

    def fit(self, images: Sequence[np.ndarray], log=None):
        bank, ids, pos = [], [], 0
        for i, img in enumerate(images):
            f = self.features(img).reshape(-1, self.proj_dim)
            sel = self.rng.choice(len(f), min(self.ppi, len(f)), replace=False)
            bank.append(f[torch.from_numpy(sel).to(f.device)])
            ids.append(torch.full((len(sel),), i, dtype=torch.long))
            self.offsets[i] = (pos, pos + len(sel))
            pos += len(sel)
            if log and i % 200 == 0:
                log(f"  normality bank {i}/{len(images)}")
        self.bank = torch.cat(bank)
        self.bank_ids = torch.cat(ids)
        self.bank_sq = (self.bank ** 2).sum(1)
        return self

    @torch.no_grad()
    def knn_distance(self, q: torch.Tensor, image_id: Optional[int] = None) -> torch.Tensor:
        """Distance of each query patch to its k-th nearest bank patch from another image."""
        if not hasattr(self, "offsets"):
            self.offsets = {}
        if image_id is not None and image_id not in self.offsets:
            # bank built elsewhere (e.g. tests): derive contiguous ranges from the id vector
            ids = self.bank_ids.cpu().numpy()
            for j in np.unique(ids):
                w = np.flatnonzero(ids == j)
                self.offsets[int(j)] = (int(w[0]), int(w[-1]) + 1)
        kth = torch.empty(len(q), device=q.device)
        for c in range(0, len(q), 2048):   # chunked to bound memory
            qc = q[c:c + 2048]
            dist = (qc ** 2).sum(1, keepdim=True) - 2 * qc @ self.bank.T + self.bank_sq[None]
            if image_id is not None:
                s, e = self.offsets[image_id]
                dist[:, s:e] = float("inf")     # never explain a patch by its own image
            vals = torch.topk(dist, self.k, dim=1, largest=False).values
            kth[c:c + 2048] = vals[:, self.k - 1].clamp(min=0).sqrt()
        return kth

    @torch.no_grad()
    def score(self, img: np.ndarray, image_id: Optional[int] = None) -> np.ndarray:
        f = self.features(img)
        hs, ws, d = f.shape
        amap = self.knn_distance(f.reshape(-1, d), image_id).reshape(1, 1, hs, ws)
        H, W = img.shape[:2]
        amap = F.interpolate(amap, size=(hs * 8, ws * 8), mode="bilinear", align_corners=False)[0, 0, :H, :W]
        amap = amap.float().cpu().numpy()
        if self.sigma > 0:
            amap = cv2.GaussianBlur(amap, (0, 0), self.sigma)
        return amap.astype(np.float32)

    def score_many(self, images: Sequence[np.ndarray], image_ids: Optional[Sequence[Optional[int]]] = None,
                   log=None) -> List[np.ndarray]:
        out = []
        for i, img in enumerate(images):
            out.append(self.score(img, None if image_ids is None else image_ids[i]).astype(np.float16))
            if log and i % 200 == 0:
                log(f"  anomaly maps {i}/{len(images)}")
        return out
