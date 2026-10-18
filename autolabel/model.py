"""U-Net with an ImageNet-pretrained timm encoder and a full-resolution detail path.

Thin scratches are often 1-3 px wide. Standard U-Nets predict at stride 2 or 4 and upsample, which
blurs such structures away; we add a light full-resolution stem whose features are concatenated in the
last decoder stage so that the logits are produced at stride 1.
"""
from __future__ import annotations

import os
import urllib.request
from typing import List, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pretrained weights are fetched from timm's GitHub release assets (works where huggingface.co is blocked).
_GH = "https://github.com/rwightman/pytorch-image-models/releases/download/"
WEIGHT_URLS = {
    "resnet18d": _GH + "v0.1-weights/resnet18d_ra2-48a79e06.pth",
    "resnet34": _GH + "v0.1-weights/resnet34-43635321.pth",
    "efficientnet_b0": _GH + "v0.1-weights/efficientnet_b0_ra-3dd342df.pth",
    "efficientnet_b3": _GH + "v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth",
    "wide_resnet50_2": _GH + "v0.1-weights/wide_resnet50_racm-8234f177.pth",
}


def get_device() -> torch.device:
    """AUTOLABEL_DEVICE (cpu|cuda|mps) if set, else CUDA > Apple MPS > CPU."""
    want = os.environ.get("AUTOLABEL_DEVICE", "").strip().lower()
    if want:
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_threads(n: int):
    """n > 0 limits torch CPU threads; 0 keeps torch's default (all cores)."""
    if n and n > 0:
        torch.set_num_threads(n)


def weights_dir() -> str:
    return os.environ.get("AUTOLABEL_WEIGHTS", os.path.expanduser("~/.cache/autolabel/weights"))


def pretrained_state_dict(name: str) -> Optional[dict]:
    url = WEIGHT_URLS.get(name)
    if url is None:
        return None
    path = os.path.join(weights_dir(), os.path.basename(url))
    if not os.path.exists(path):
        os.makedirs(weights_dir(), exist_ok=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return torch.load(path, map_location="cpu")


def create_encoder(name: str, pretrained: bool = True, in_chans: int = 3, **kw):
    enc = timm.create_model(name, features_only=True, pretrained=False, in_chans=in_chans, **kw)
    if pretrained:
        sd = pretrained_state_dict(name)
        if sd is not None:
            if in_chans != 3:
                # adapt first conv: copy RGB filters, initialise extra channels with their mean
                for k, v in list(sd.items()):
                    if v.ndim == 4 and v.shape[1] == 3:
                        extra = v.mean(1, keepdim=True).repeat(1, in_chans - 3, 1, 1)
                        sd[k] = torch.cat([v, extra], 1)
                        break
            enc.load_state_dict(sd, strict=False)
    return enc


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3):
        super().__init__(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(ConvBNAct(cin + cskip, cout), ConvBNAct(cout, cout))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class UNet(nn.Module):
    def __init__(self, encoder: str = "resnet18d", pretrained: bool = True, in_chans: int = 3,
                 decoder_channels=(256, 128, 64, 32), stem_channels: int = 16, n_classes: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = create_encoder(encoder, pretrained, in_chans)
        chs: List[int] = self.encoder.feature_info.channels()
        self.stem = nn.Sequential(ConvBNAct(in_chans, stem_channels), ConvBNAct(stem_channels, stem_channels))
        blocks, cin = [], chs[-1]
        skips = chs[:-1][::-1]  # deepest skip first
        for cskip, cout in zip(skips, decoder_channels):
            blocks.append(DecoderBlock(cin, cskip, cout))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.final = DecoderBlock(cin, stem_channels, stem_channels * 2)
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(stem_channels * 2, n_classes, 1)
        self.stride = max(self.encoder.feature_info.reduction())

    def forward(self, x):
        feats = self.encoder(x)
        y = feats[-1]
        for blk, skip in zip(self.blocks, feats[:-1][::-1]):
            y = blk(y, skip)
        y = self.final(y, self.stem(x))
        return self.head(self.drop(y))


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """uint8-range float tensor Bx3xHxW (or Bx(3+k)xHxW, extra channels already in [0,1]) -> network input."""
    rgb = (x[:, :3] / 255.0 - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
    if x.shape[1] > 3:
        return torch.cat([rgb, x[:, 3:]], 1)
    return rgb
