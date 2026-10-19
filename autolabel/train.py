"""Training of a single segmentation member on labeled + (pseudo-labeled / synthetic) data."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import augment as A
from .model import UNet, get_device, memory_format, normalize, set_threads


@dataclass
class TrainConfig:
    encoder: str = "resnet18d"
    decoder_channels: Tuple[int, ...] = (128, 64, 32, 16)
    stem_channels: int = 8
    crop: int = 192
    batch: int = 8
    iters: int = 600
    lr: float = 1e-3
    enc_lr_mult: float = 0.3
    weight_decay: float = 1e-4
    warmup: int = 30
    p_center: float = 0.6            # labeled defective image: prob. that the crop is centred on a defect
    p_synth: float = 0.35            # prob. that a sample is a synthetic defect on an unlabeled background
    p_procedural: float = 0.0        # share of procedural (instead of copy-paste) defects in synthetic samples
    max_paste: int = 3
    paste_large: bool = False        # also paste large area defects (alpha blending; rarely realistic)
    p_pseudo: float = 0.0            # prob. that a sample is a pseudo-labeled unlabeled image (self-training)
    rot90: bool = True
    scale_range: Tuple[float, float] = (0.85, 1.2)
    dice_weight: float = 1.0
    boundary_ignore: int = 0         # ignore a ring of this width outside GT defects (subjective borders)
    dropout: float = 0.1
    seed: int = 0
    threads: int = 0                 # CPU threads for torch (0 = torch default = all cores)

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainData:
    """Everything a member can learn from.

    labeled:     list of (image uint8 HxWx3, mask bool HxW)
    backgrounds: unlabeled images used as canvases for synthetic defects; optional per-image
                 (positive, weight) maps from a teacher so that unlabeled real defects are not
                 taught as background.
    pseudo:      list of (image, positive mask, weight map) from a teacher (weight 0 = ignore)
    """
    labeled: List[Tuple[np.ndarray, np.ndarray]]
    backgrounds: List[np.ndarray] = field(default_factory=list)
    background_labels: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    pseudo: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)


class SampleGenerator:
    def __init__(self, data: TrainData, cfg: TrainConfig, rng: np.random.RandomState):
        self.d, self.cfg, self.rng = data, cfg, rng
        bank = A.build_defect_bank([im for im, _ in data.labeled], [m for _, m in data.labeled])
        self.contrasts = A.defect_contrast_stats(bank)
        self.bank = bank if cfg.paste_large else [p for p in bank if p.mode == "residual"]
        self.pos_idx = [i for i, (_, m) in enumerate(data.labeled) if m.any()]

    def _finish(self, img, tgt, w):
        img = A.photometric(self.rng, img.astype(np.float32))
        return img, tgt.astype(np.float32), w.astype(np.float32)

    def _labeled(self):
        c, rng = self.cfg, self.rng
        if self.pos_idx and rng.rand() < 0.7:
            i = self.pos_idx[rng.randint(len(self.pos_idx))]
        else:
            i = rng.randint(len(self.d.labeled))
        img, m = self.d.labeled[i]
        w = np.ones(m.shape, np.float32)
        if c.boundary_ignore > 0 and m.any():
            ring = A._grow(m, c.boundary_ignore) & ~m
            w[ring] = 0.0
        img, m, w = A.rescale(rng, [img, m, w], c.scale_range)
        img, m, w = A.random_crop(rng, [img, m, w], c.crop, center_on=m, p_center=c.p_center)
        img, m, w = A.geometric(rng, [img, m, w], c.rot90)
        return img.astype(np.float32), m.copy(), w

    def _synthetic(self):
        c, rng = self.cfg, self.rng
        j = rng.randint(len(self.d.backgrounds))
        img = self.d.backgrounds[j]
        if self.d.background_labels is not None:
            pos, w = self.d.background_labels[j]
        else:
            pos, w = np.zeros(img.shape[:2], bool), np.ones(img.shape[:2], np.float32)
        img, pos, w = A.rescale(rng, [img, pos, w], c.scale_range)
        img, pos, w = A.random_crop(rng, [img, pos, w], c.crop)
        img, pos, w = A.geometric(rng, [img, pos, w], c.rot90)
        canvas = img.astype(np.float32).copy()
        tgt = pos.copy()
        w = w.copy()
        ign = np.zeros_like(tgt)
        for _ in range(rng.randint(1, c.max_paste + 1)):
            if self.bank and rng.rand() >= c.p_procedural:
                m = A.paste_defect(rng, canvas, tgt, self.bank[rng.randint(len(self.bank))], ignore=ign)
            else:
                m = A.procedural_defect(rng, canvas, tgt, self.contrasts, ignore=ign)
            w[m] = 1.0  # visibly pasted defect is certain even if the teacher was unsure there
        # invisible parts of pasted defects and their changed rim: neither positive nor negative
        w[ign & ~tgt] = 0.0
        return canvas, tgt, w

    def _pseudo(self):
        c, rng = self.cfg, self.rng
        j = rng.randint(len(self.d.pseudo))
        img, pos, w = self.d.pseudo[j]
        img, pos, w = A.rescale(rng, [img, pos, w], c.scale_range)
        img, pos, w = A.random_crop(rng, [img, pos, w], c.crop, center_on=pos, p_center=0.5)
        img, pos, w = A.geometric(rng, [img, pos, w], c.rot90)
        return img.astype(np.float32), pos.copy(), w

    def sample(self):
        c, rng = self.cfg, self.rng
        r = rng.rand()
        if self.d.pseudo and r < c.p_pseudo:
            out = self._pseudo()
        elif self.d.backgrounds and (self.bank or c.p_procedural > 0) and r < c.p_pseudo + c.p_synth:
            out = self._synthetic()
        else:
            out = self._labeled()
        return self._finish(*out)

    def batch(self, n):
        xs, ts, ws = zip(*(self.sample() for _ in range(n)))
        x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous()
        return x, torch.from_numpy(np.stack(ts))[:, None], torch.from_numpy(np.stack(ws))[:, None]


def seg_loss(logits, target, weight, dice_weight=1.0):
    wsum = weight.sum().clamp(min=1.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="sum") / wsum
    p = torch.sigmoid(logits) * weight
    t = target * weight
    dice = 1 - (2 * (p * t).sum() + 1.0) / (p.sum() + t.sum() + 1.0)
    return bce + dice_weight * dice


def build_model(cfg: TrainConfig, pretrained: bool = True) -> UNet:
    return UNet(cfg.encoder, pretrained=pretrained, decoder_channels=cfg.decoder_channels,
                stem_channels=cfg.stem_channels, dropout=cfg.dropout)


def train_member(data: TrainData, cfg: TrainConfig, log: Optional[Callable[[str], None]] = None) -> UNet:
    torch.manual_seed(cfg.seed)
    set_threads(cfg.threads)
    dev = get_device()
    rng = np.random.RandomState(cfg.seed)
    gen = SampleGenerator(data, cfg, rng)
    mf = memory_format(dev)
    model = build_model(cfg).to(dev, memory_format=mf)
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": cfg.lr * cfg.enc_lr_mult},
        {"params": dec_params, "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)
    base = [g["lr"] for g in opt.param_groups]
    model.train()
    t0, run = time.time(), 0.0
    for it in range(cfg.iters):
        f = (it + 1) / cfg.warmup if it < cfg.warmup else \
            0.5 * (1 + math.cos(math.pi * (it - cfg.warmup) / max(1, cfg.iters - cfg.warmup)))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * f
        x, t, w = (v.to(dev) for v in gen.batch(cfg.batch))
        x = normalize(x).contiguous(memory_format=mf)
        loss = seg_loss(model(x), t, w, cfg.dice_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run = 0.95 * run + 0.05 * loss.item() if it else loss.item()
        if log and (it % 100 == 0 or it == cfg.iters - 1):
            log(f"  it {it:4d}  loss {run:.4f}  {time.time() - t0:.0f}s  [{dev.type}]")
    model.eval()
    return model
