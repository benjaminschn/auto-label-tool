"""Training-sample generation: defect-aware cropping, photometric augmentation and defect synthesis.

Defect synthesis uses the *unlabeled* images as backgrounds. This is the main lever to teach the model
the large variety of normal appearance (reflections, textures, dust, machining marks) with only a
handful of labeled images:

* residual copy-paste - a labeled defect is transferred as its *deviation from the local background*
  (defect minus inpainted background) and added to a new background with random gain. This keeps the
  target surface's shading/reflection and lets us create fainter (harder) variants of a defect.
* alpha copy-paste - for large defects where a residual is ill-defined, pixels are blended with a soft
  mask after matching the mean intensity.
* procedural defects (optional) - thin Bezier "scratches" and small blobs whose contrast distribution is
  matched to the labeled defects.

Label contract (02 N2 / 07 C6): the label of a synthetic defect comes from the signal that was
*actually produced* (canvas after pasting and clipping minus canvas before), not from the source mask.
Source-mask pixels whose realised change is below a visibility threshold (clipping at 0/255, small gain,
faint residual) become *ignore*, never positive; a pasted component without enough visible pixels is
ignored entirely. A band around the source mask that changed noticeably is ignore as well. See
:func:`realised_label`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

_K3 = np.ones((3, 3), np.uint8)


def _grow(m: np.ndarray, it: int) -> np.ndarray:
    return cv2.dilate(m.astype(np.uint8), _K3, iterations=it) > 0


@dataclass
class DefectPatch:
    rgb: np.ndarray        # hxwx3 float32, source pixels
    residual: np.ndarray   # hxwx3 float32, rgb - inpainted background
    mask: np.ndarray       # hxw bool
    mode: str              # "residual" or "alpha"


def build_defect_bank(images: Sequence[np.ndarray], masks: Sequence[np.ndarray], margin: int = 6,
                      max_residual_area: int = 6000) -> List[DefectPatch]:
    bank = []
    for img, m in zip(images, masks):
        if m is None or not m.any():
            continue
        lab, n = ndi.label(_grow(m, 2), structure=np.ones((3, 3)))
        for sl_i, sl in enumerate(ndi.find_objects(lab)):
            if sl is None:
                continue
            y0 = max(0, sl[0].start - margin)
            y1 = min(m.shape[0], sl[0].stop + margin)
            x0 = max(0, sl[1].start - margin)
            x1 = min(m.shape[1], sl[1].stop + margin)
            pm = m[y0:y1, x0:x1] & (lab[y0:y1, x0:x1] == sl_i + 1)
            if pm.sum() < 3:
                continue
            rgb = img[y0:y1, x0:x1].astype(np.float32)
            big = pm.sum() > max_residual_area or (y1 - y0) > 0.8 * m.shape[0] or (x1 - x0) > 0.8 * m.shape[1]
            if big:
                res = np.zeros_like(rgb)
                mode = "alpha"
            else:
                inp_mask = _grow(pm, 2).astype(np.uint8)
                bg = cv2.inpaint(img[y0:y1, x0:x1].copy(), inp_mask, 5, cv2.INPAINT_TELEA).astype(np.float32)
                res = (rgb - bg) * _grow(pm, 2)[..., None]
                mode = "residual"
            bank.append(DefectPatch(rgb, res, pm, mode))
    return bank


def defect_contrast_stats(bank: Sequence[DefectPatch]) -> np.ndarray:
    """Mean residual (gray) per residual-mode defect -> empirical contrast distribution."""
    vals = [p.residual.mean(-1)[p.mask].mean() for p in bank if p.mode == "residual" and p.mask.any()]
    return np.array(vals if vals else [-25.0, 25.0], np.float32)


def _rand_geom(rng, arr_list, allow_rot90=True):
    k = rng.randint(4) if allow_rot90 else 0
    fh, fv = rng.rand() < 0.5, rng.rand() < 0.5
    out = []
    for a in arr_list:
        if k:
            a = np.rot90(a, k)
        if fh:
            a = a[:, ::-1]
        if fv:
            a = a[::-1]
        out.append(np.ascontiguousarray(a))
    return out


def _scale(arrs, s):
    out = []
    for a in arrs:
        h, w = a.shape[:2]
        size = (max(2, int(round(w * s))), max(2, int(round(h * s))))
        if a.dtype == bool:
            out.append(cv2.resize(a.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) > 0.3)
        else:
            out.append(cv2.resize(a, size, interpolation=cv2.INTER_LINEAR))
    return out


def local_noise_sigma(gray: np.ndarray) -> float:
    """Robust noise level of a (background) gray patch: MAD of a high-pass residual."""
    g = gray.astype(np.float32)
    if g.size < 9:
        return 1.0
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    return float(1.4826 * np.median(np.abs(hp - np.median(hp))))


@dataclass
class VisibilityRule:
    """When does a pasted pixel count as a visible (positive) defect pixel?

    |delta| >= max(kappa * sigma_local, min_abs) with delta the realised change in gray level. kappa = 1
    keeps faint but real-looking defects as positives (02 QC 2); min_abs guards flat backgrounds.
    A pasted component needs at least ``min_visible_px`` visible pixels (or ``min_visible_frac`` of its
    area, whichever is smaller) to be positive at all; otherwise it is ignore.
    """
    kappa: float = 1.0
    min_abs: float = 2.0
    min_visible_px: int = 3
    min_visible_frac: float = 0.2
    band: int = 3              # band around the source mask that may change (soft paste edges)


def realised_label(before: np.ndarray, after: np.ndarray, src_mask: np.ndarray,
                   rule: Optional[VisibilityRule] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Positive and ignore masks of a synthetic defect from the realised change.

    before / after: HxWx3 (or HxW) float canvases around the paste location, ``after`` already clipped.
    src_mask: the pasted source mask (bool HxW). Returns (positive, ignore, threshold):
      * positive  = visible pixels of src_mask components that are visible enough as a whole;
      * ignore    = src_mask minus positive, plus the band around src_mask that changed by >= threshold/2.
    Pixels outside both stay whatever the caller had (normally negative).
    """
    rule = rule or VisibilityRule()
    m = src_mask.astype(bool)
    b = before.astype(np.float32)
    a = after.astype(np.float32)
    if b.ndim == 3:
        delta = np.abs(a - b).max(-1)       # a change in any channel is visible
        gray_bg = b.mean(-1)
    else:
        delta, gray_bg = np.abs(a - b), b
    band = _grow(m, rule.band) if rule.band > 0 else m
    sigma = local_noise_sigma(gray_bg[~band]) if (~band).any() else local_noise_sigma(gray_bg)
    thr = max(rule.kappa * sigma, rule.min_abs)
    vis = (delta >= thr) & m
    pos = np.zeros_like(m)
    if vis.any():
        lab, n = ndi.label(m, structure=np.ones((3, 3)))
        idx = np.arange(1, n + 1)
        area = np.asarray(ndi.sum(np.ones_like(delta), lab, idx))
        nvis = np.asarray(ndi.sum(vis.astype(np.float32), lab, idx))
        need = np.minimum(rule.min_visible_px, np.maximum(1.0, np.ceil(rule.min_visible_frac * area)))
        ok = np.concatenate([[False], nvis >= need])
        pos = vis & ok[lab]
    ign = (m & ~pos) | (band & ~m & (delta >= 0.5 * thr))
    return pos, ign, thr


def paste_defect(rng, canvas: np.ndarray, target: np.ndarray, patch: DefectPatch,
                 gain_range=(0.6, 1.3), scale_range=(0.75, 1.3), center: Optional[Tuple[int, int]] = None,
                 ignore: Optional[np.ndarray] = None, rule: Optional[VisibilityRule] = None):
    """Paste one defect into ``canvas`` (float32 HxWx3, modified in place).

    The label follows the realised change (:func:`realised_label`): visible pixels are OR-ed into
    ``target``; invisible source pixels and the changed band go to ``ignore`` (if given) and are removed
    from ``target`` where they were not positive before. Returns the realised positive mask (full size).
    """
    rgb, res, m = _rand_geom(rng, [patch.rgb, patch.residual, patch.mask])
    s = rng.uniform(*scale_range)
    rgb, res, m = _scale([rgb, res, m], s)
    h, w = m.shape
    H, W = canvas.shape[:2]
    if h >= H or w >= W:
        ys, xs = (h - H + 1) // 2 if h >= H else 0, (w - W + 1) // 2 if w >= W else 0
        rgb, res, m = rgb[ys:ys + min(h, H), xs:xs + min(w, W)], res[ys:ys + min(h, H), xs:xs + min(w, W)], \
            m[ys:ys + min(h, H), xs:xs + min(w, W)]
        h, w = m.shape
    if center is None:
        y0, x0 = rng.randint(0, H - h + 1), rng.randint(0, W - w + 1)
    else:
        y0 = int(np.clip(center[0] - h // 2, 0, H - h))
        x0 = int(np.clip(center[1] - w // 2, 0, W - w))
    rule = rule or VisibilityRule()
    pad = rule.band + 1
    wy0, wy1, wx0, wx1 = max(0, y0 - pad), min(H, y0 + h + pad), max(0, x0 - pad), min(W, x0 + w + pad)
    before = canvas[wy0:wy1, wx0:wx1].copy()
    region = canvas[y0:y0 + h, x0:x0 + w]
    if patch.mode == "residual":
        g = rng.uniform(*gain_range)
        soft = cv2.GaussianBlur(_grow(m, 2).astype(np.float32), (0, 0), 1.0)
        region += g * res * soft[..., None]
    else:
        soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)[..., None]
        src = rgb - rgb[~m].mean(0) + region[~m].mean(0) if (~m).any() else rgb
        region[:] = soft * src + (1 - soft) * region
    np.clip(region, 0, 255, out=region)
    src = np.zeros((H, W), bool)
    src[y0:y0 + h, x0:x0 + w] = m
    return _apply_realised(canvas, before, (wy0, wy1, wx0, wx1), src, target, ignore, rule)


def _apply_realised(canvas, before, win, src, target, ignore, rule):
    wy0, wy1, wx0, wx1 = win
    pos_w, ign_w, _ = realised_label(before, canvas[wy0:wy1, wx0:wx1], src[wy0:wy1, wx0:wx1], rule)
    pos = np.zeros_like(src)
    ign = np.zeros_like(src)
    pos[wy0:wy1, wx0:wx1] = pos_w
    ign[wy0:wy1, wx0:wx1] = ign_w
    target |= pos
    if ignore is not None:
        ignore |= ign & ~target
    return pos


def _bezier(rng, H, W, length):
    p0 = np.array([rng.uniform(0, H), rng.uniform(0, W)])
    ang = rng.uniform(0, np.pi)
    d = np.array([np.sin(ang), np.cos(ang)]) * length
    bend = rng.normal(0, 0.25, size=2) * length
    pts = np.array([p0, p0 + d / 2 + bend, p0 + d])
    t = np.linspace(0, 1, max(8, int(length * 2)))[:, None]
    return ((1 - t) ** 2) * pts[0] + 2 * (1 - t) * t * pts[1] + (t ** 2) * pts[2]


def procedural_defect(rng, canvas: np.ndarray, target: np.ndarray, contrasts: np.ndarray, kind: str = "auto",
                      ignore: Optional[np.ndarray] = None, rule: Optional[VisibilityRule] = None):
    """Draw a synthetic thin scratch or small blob with a contrast sampled from the labeled defects.

    Labelled from the realised change like :func:`paste_defect`; returns the realised positive mask.
    """
    H, W = canvas.shape[:2]
    ss = 4  # supersampling for anti-aliasing
    layer = np.zeros((H * ss, W * ss), np.uint8)
    kind = kind if kind != "auto" else ("scratch" if rng.rand() < 0.7 else "blob")
    if kind == "scratch":
        pts = _bezier(rng, H, W, rng.uniform(0.15, 0.6) * max(H, W)) * ss
        width = int(round(rng.uniform(1.0, 3.0) * ss))
        cv2.polylines(layer, [pts[:, ::-1].astype(np.int32).reshape(-1, 1, 2)], False, 255, width, cv2.LINE_AA)
    else:
        c = (int(rng.uniform(0, W) * ss), int(rng.uniform(0, H) * ss))
        ax = (int(rng.uniform(1.5, 8) * ss), int(rng.uniform(1.5, 8) * ss))
        cv2.ellipse(layer, c, ax, rng.uniform(0, 180), 0, 360, 255, -1, cv2.LINE_AA)
    alpha = cv2.resize(layer, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), rng.uniform(0.3, 0.8))
    delta = float(rng.choice(contrasts)) * rng.uniform(0.6, 1.3)
    before = canvas.copy()
    canvas += alpha[..., None] * delta
    np.clip(canvas, 0, 255, out=canvas)
    m = alpha > 0.35
    return _apply_realised(canvas, before, (0, H, 0, W), m, target, ignore, rule or VisibilityRule())


def photometric(rng, img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """img float32 0..255 HxWx3 -> augmented copy."""
    x = img
    if rng.rand() < 0.8:
        c = np.exp(rng.uniform(-0.25, 0.25) * strength)
        b = rng.uniform(-20, 20) * strength
        mu = x.mean()
        x = (x - mu) * c + mu + b
    if rng.rand() < 0.3:
        g = np.exp(rng.uniform(-0.25, 0.25) * strength)
        x = 255.0 * (np.clip(x, 0, 255) / 255.0) ** g
    if rng.rand() < 0.3:
        x = x * rng.uniform(0.92, 1.08, size=(1, 1, 3)).astype(np.float32)
    if rng.rand() < 0.15:
        x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), rng.uniform(0.3, 0.9))
    if rng.rand() < 0.3:
        x = x + rng.normal(0, rng.uniform(1, 6) * strength, size=x.shape[:2])[..., None]
    return np.clip(x, 0, 255).astype(np.float32)


def random_crop(rng, arrays: List[np.ndarray], size: int, center_on: Optional[np.ndarray] = None,
                p_center: float = 0.0):
    """Crop the same window from all arrays (reflect-pad if smaller than ``size``)."""
    H, W = arrays[0].shape[:2]
    if H < size or W < size:
        py, px = max(0, size - H), max(0, size - W)
        padded = []
        for a in arrays:
            pad = ((py // 2, py - py // 2), (px // 2, px - px // 2)) + ((0, 0),) * (a.ndim - 2)
            padded.append(np.pad(a, pad, mode="reflect"))
        arrays = padded
        if center_on is not None:
            center_on = np.pad(center_on, ((py // 2, py - py // 2), (px // 2, px - px // 2)))
        H, W = arrays[0].shape[:2]
    if center_on is not None and rng.rand() < p_center and center_on.any():
        ys, xs = np.nonzero(center_on)
        j = rng.randint(len(ys))
        cy = ys[j] + rng.randint(-size // 3, size // 3 + 1)
        cx = xs[j] + rng.randint(-size // 3, size // 3 + 1)
        y0 = int(np.clip(cy - size // 2, 0, H - size))
        x0 = int(np.clip(cx - size // 2, 0, W - size))
    else:
        y0, x0 = rng.randint(0, H - size + 1), rng.randint(0, W - size + 1)
    return [a[y0:y0 + size, x0:x0 + size] for a in arrays]


def geometric(rng, arrays: List[np.ndarray], rot90: bool = True) -> List[np.ndarray]:
    return _rand_geom(rng, arrays, allow_rot90=rot90)


def rescale(rng, arrays: List[np.ndarray], scale_range=(0.85, 1.2)) -> List[np.ndarray]:
    s = rng.uniform(*scale_range)
    if abs(s - 1) < 0.03:
        return arrays
    out = []
    for a in arrays:
        h, w = a.shape[:2]
        size = (max(8, int(round(w * s))), max(8, int(round(h * s))))
        if a.dtype == bool:
            if s < 1:  # keep thin structures alive when shrinking
                out.append(cv2.resize(a.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.3)
            else:
                out.append(cv2.resize(a.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0)
        elif a.ndim == 2:
            out.append(cv2.resize(a, size, interpolation=cv2.INTER_NEAREST))
        else:
            out.append(cv2.resize(a, size, interpolation=cv2.INTER_LINEAR))
    return out
