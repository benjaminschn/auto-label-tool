"""Full-image inference with test-time augmentation and ensembles."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .model import memory_format, normalize


def _pad32(x: torch.Tensor, m: int = 32):
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        mode = "reflect" if (ph < h and pw < w) else "replicate"
        x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode=mode)
    return x, (ph // 2, pw // 2, h, w)


@torch.no_grad()
def predict_image(model, img: np.ndarray, tta: str = "hflip") -> np.ndarray:
    """img HxWx3 uint8 -> defect probability HxW float32."""
    dev = next(model.parameters()).device
    x = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None].to(dev)
    x, (oy, ox, h, w) = _pad32(x)
    xs = [x]
    if tta in ("hflip", "flips"):
        xs.append(torch.flip(x, [3]))
    if tta == "flips":
        xs.append(torch.flip(x, [2]))
    batch = normalize(torch.cat(xs)).contiguous(memory_format=memory_format(dev))
    p = torch.sigmoid(model(batch))[:, 0]
    outs = [p[0]]
    if len(xs) > 1:
        outs.append(torch.flip(p[1], [1]))
    if len(xs) > 2:
        outs.append(torch.flip(p[2], [0]))
    prob = torch.stack(outs).mean(0)
    return prob[oy:oy + h, ox:ox + w].float().cpu().numpy()


def predict_members(models: Sequence, images: Sequence[np.ndarray], tta: str = "hflip") -> List[np.ndarray]:
    """Returns, per image, an array KxHxW (float16) with the probability of every ensemble member."""
    out = []
    for img in images:
        out.append(np.stack([predict_image(m, img, tta) for m in models]).astype(np.float16))
    return out
