"""Dataset loading, caching and few-label split protocol.

Every dataset is turned into a list of :class:`Sample` objects holding the image
(HxWx3 uint8), the binary defect mask (HxW bool, or ``None`` if unknown), a
``kind`` (defect class name or ``"free"``) and a ``group`` key used for
leakage-free splitting (e.g. the same physical location photographed over time).
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Sample:
    id: str
    image: np.ndarray                 # HxWx3 uint8
    mask: Optional[np.ndarray]        # HxW bool, None = unlabeled / unknown
    kind: str = "unknown"             # defect class or "free"
    group: str = ""                   # samples sharing a group never cross split boundaries
    meta: Dict = field(default_factory=dict)

    @property
    def has_defect(self) -> bool:
        return self.mask is not None and bool(self.mask.any())


def _rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _resize(img: np.ndarray, scale: float, nearest: bool = False) -> np.ndarray:
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if img.dtype == bool:
        # area-resize the mask and keep pixels covered >= 30% so thin structures survive downscaling
        m = Image.fromarray(img.astype(np.uint8) * 255).resize(size, Image.BOX)
        return np.array(m) >= 77
    return np.array(Image.fromarray(img).resize(size, Image.NEAREST if nearest else Image.BILINEAR))


# ----------------------------------------------------------------------------- loaders

def load_magnetic_tile(root: str, scale: float = 1.0) -> List[Sample]:
    """Magnetic-tile defect dataset (Huang et al. 2018), github.com/abin24/Magnetic-tile-defect-datasets."""
    out = []
    for d in sorted(glob.glob(os.path.join(root, "MT_*"))):
        kind = os.path.basename(d)[3:].lower()
        for f in sorted(glob.glob(os.path.join(d, "Imgs", "*.jpg"))):
            m = np.array(Image.open(f[:-4] + ".png").convert("L")) > 127
            sid = os.path.basename(f)[:-4]
            out.append(Sample(f"mt/{kind}/{sid}", _resize(_rgb(f), scale), _resize(m, scale),
                              kind="free" if kind == "free" else kind, group=f"{kind}/{sid}"))
    return out


def load_bsdata(root: str, scale: float = 0.5, target_width: Optional[int] = 1130) -> List[Sample]:
    """BSData ball-screw-drive pitting dataset (Schlagenhauf & Landwehr 2021), github.com/2Obe/BSData.

    Images come in several native resolutions; all are first rescaled to ``target_width`` and then by
    ``scale``. Images without annotation file are pitting-free according to the dataset authors.
    Group key = spindle position + angle, so wear sequences of the same spot stay in one split.
    """
    meta = json.load(open(os.path.join(root, "image_data.json")))["images"]
    out = []
    for it in meta:
        fn = it["filename"]
        img = Image.open(os.path.join(root, "data", fn)).convert("RGB")
        w, h = img.size
        s = (target_width / w if target_width else 1.0) * scale
        size = (round(w * s), round(h * s))
        mask = Image.new("L", size, 0)
        if it.get("annotations"):
            js = json.load(open(os.path.join(root, "label", it["annotations"])))
            sx, sy = size[0] / js["imageWidth"], size[1] / js["imageHeight"]
            for shp in js["shapes"]:
                pts = [(x * sx, y * sy) for x, y in shp["points"]]
                ImageDraw.Draw(mask).polygon(pts, fill=1, outline=1)
        m = np.array(mask) > 0
        parts = fn.split("_")
        group = f"t{it.get('type')}_{parts[2]}_{parts[3]}" if len(parts) > 3 else fn
        out.append(Sample(f"bsdata/{fn[:-4]}", np.array(img.resize(size, Image.BILINEAR)), m,
                          kind="pitting" if m.any() else "free", group=group))
    return out


def load_deepcrack(root: str, scale: float = 1.0) -> List[Sample]:
    """DeepCrack (Liu et al. 2019), github.com/yhlleo/DeepCrack (dataset/DeepCrack.zip, extracted)."""
    out = []
    for split in ("train", "test"):
        for f in sorted(glob.glob(os.path.join(root, f"{split}_img", "*.jpg"))):
            sid = os.path.basename(f)[:-4]
            lab = os.path.join(root, f"{split}_lab", sid + ".png")
            m = np.array(Image.open(lab).convert("L")) > 127
            out.append(Sample(f"deepcrack/{sid}", _resize(_rgb(f), scale), _resize(m, scale),
                              kind="crack" if m.any() else "free", group=sid.split("-")[0]))
    return out


def load_crackforest(root: str, scale: float = 1.0) -> List[Sample]:
    """CrackForest CFD (Shi et al. 2016), github.com/cuilimeng/CrackForest-dataset."""
    import scipy.io as sio
    out = []
    for g in sorted(glob.glob(os.path.join(root, "groundTruth", "*.mat"))):
        sid = os.path.basename(g)[:-4]
        seg = sio.loadmat(g)["groundTruth"][0, 0]["Segmentation"]
        m = seg == 2
        out.append(Sample(f"cfd/{sid}", _resize(_rgb(os.path.join(root, "image", sid + ".jpg")), scale),
                          _resize(m, scale), kind="crack" if m.any() else "free", group=sid))
    return out


def load_ksdd2(root: str, scale: float = 1.0) -> List[Sample]:
    """KolektorSDD2 (Božič, Tabernik, Skočaj 2021), www.vicos.si/resources/kolektorsdd2/.

    Expected layout (as distributed): ``root/train`` and ``root/test`` with ``<id>.png`` images and
    ``<id>_GT.png`` masks. The official train/test split is ignored; our own few-label protocol
    (``make_split``) is applied to the union. Every image is its own group (no sequence info).
    """
    out = []
    for split in ("train", "test"):
        for f in sorted(glob.glob(os.path.join(root, split, "*.png"))):
            if f.endswith("_GT.png"):
                continue
            sid = os.path.basename(f)[:-4]
            gtf = os.path.join(root, split, sid + "_GT.png")
            m = np.array(Image.open(gtf).convert("L")) > 0 if os.path.exists(gtf) else None
            img = _rgb(f)
            if m is not None and m.shape != img.shape[:2]:
                m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(img.shape[1::-1], Image.NEAREST)) > 0
            out.append(Sample(f"ksdd2/{split}/{sid}", _resize(img, scale),
                              None if m is None else _resize(m, scale),
                              kind="defect" if m is not None and m.any() else "free", group=f"{split}/{sid}"))
    return out


def load_folder(images_dir: str, masks_dir: Optional[str] = None, mask_suffix: str = "",
                exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"), scale: float = 1.0) -> List[Sample]:
    """Generic loader for custom data: ``images_dir/<name>.<ext>`` and optional ``masks_dir/<name><suffix>.png``.

    Images without a matching mask file are returned with ``mask=None`` (unlabeled).
    """
    out = []
    for f in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in exts:
            continue
        m = None
        if masks_dir:
            for e in (".png", ".bmp", ".tif", ext):
                p = os.path.join(masks_dir, stem + mask_suffix + e)
                if os.path.exists(p):
                    m = _resize(np.array(Image.open(p).convert("L")) > 0, scale)
                    break
        kind = "unknown" if m is None else ("defect" if m.any() else "free")
        out.append(Sample(stem, _resize(_rgb(os.path.join(images_dir, f)), scale), m, kind=kind, group=stem))
    return out


LOADERS: Dict[str, Callable[..., List[Sample]]] = {
    "mt": load_magnetic_tile,
    "bsdata": load_bsdata,
    "deepcrack": load_deepcrack,
    "cfd": load_crackforest,
    "ksdd2": load_ksdd2,
}

DEFAULT_ROOTS = {
    "mt": "Magnetic-tile-defect-datasets.",
    "bsdata": "BSData",
    "deepcrack": "DeepCrack",
    "cfd": "CrackForest-dataset",
    "ksdd2": "KolektorSDD2",
}


def load_dataset(name: str, data_root: str, cache_dir: Optional[str] = None, **kw) -> List[Sample]:
    """Load a benchmark dataset by name, using a pickle cache (decoding + polygon rasterisation is slow)."""
    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        tag = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
        cache = os.path.join(cache_dir, f"{name}{'_' + tag if tag else ''}.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as fh:
                return pickle.load(fh)
    samples = LOADERS[name](os.path.join(data_root, DEFAULT_ROOTS[name]), **kw)
    if cache:
        with open(cache, "wb") as fh:
            pickle.dump(samples, fh, protocol=4)
    return samples


# ----------------------------------------------------------------------------- splits

@dataclass
class Split:
    labeled: List[int]
    unlabeled: List[int]
    test: List[int]

    def describe(self, samples: Sequence[Sample]) -> str:
        def c(ix):
            d = sum(samples[i].has_defect for i in ix)
            return f"{len(ix)} ({d} defective)"
        return f"L={c(self.labeled)}  U={c(self.unlabeled)}  T={c(self.test)}"


def make_split(samples: Sequence[Sample], n_labeled_defect: int, n_labeled_free: int,
               test_frac: float = 0.2, seed: int = 0, strict_groups: bool = False) -> Split:
    """Few-label protocol.

    1. Hold out ``test_frac`` of the *groups* as test set T (only used for downstream evaluation), so
       T never shares a physical part / wear sequence / source image with the rest.
    2. From the rest pick ``n_labeled_defect`` defective + ``n_labeled_free`` defect-free images as the
       labeled set L, stratified over defect kinds (round robin), at most one image per group.
    3. Everything else is the unlabeled pool U that has to be auto-labeled - like in practice, L is just
       a small random part of the same pool. Its ground truth is hidden from the method and only used
       to *measure* pseudo-label quality. ``strict_groups`` additionally removes L's groups from U.
    """
    rng = np.random.RandomState(seed)
    groups = sorted({s.group for s in samples})
    rng.shuffle(groups)
    n_test = int(round(test_frac * len(groups)))
    test_groups = set(groups[:n_test])
    test = [i for i, s in enumerate(samples) if s.group in test_groups]
    pool = [i for i, s in enumerate(samples) if s.group not in test_groups]

    by_kind: Dict[str, List[int]] = {}
    for i in pool:
        if samples[i].has_defect:
            by_kind.setdefault(samples[i].kind, []).append(i)
    for v in by_kind.values():
        rng.shuffle(v)
    kinds = sorted(by_kind)
    rng.shuffle(kinds)
    labeled, used_groups = [], set()
    ptr = {k: 0 for k in kinds}
    while len(labeled) < n_labeled_defect and any(ptr[k] < len(by_kind[k]) for k in kinds):
        for k in kinds:
            if len(labeled) >= n_labeled_defect:
                break
            while ptr[k] < len(by_kind[k]):
                i = by_kind[k][ptr[k]]
                ptr[k] += 1
                if samples[i].group not in used_groups:
                    labeled.append(i)
                    used_groups.add(samples[i].group)
                    break
    free = [i for i in pool if not samples[i].has_defect and samples[i].mask is not None]
    rng.shuffle(free)
    for i in free:
        if sum(1 for j in labeled if not samples[j].has_defect) >= n_labeled_free:
            break
        if samples[i].group not in used_groups:
            labeled.append(i)
            used_groups.add(samples[i].group)
    lab_set = set(labeled)
    if strict_groups:
        unlabeled = [i for i in pool if samples[i].group not in used_groups]
    else:
        unlabeled = [i for i in pool if i not in lab_set]
    return Split(sorted(labeled), sorted(unlabeled), sorted(test))


def kfold_indices(labeled: Sequence[int], samples: Sequence[Sample], k: int, seed: int = 0) -> List[List[int]]:
    """Split the labeled set into k folds, stratified by defective / defect-free."""
    rng = np.random.RandomState(seed + 991)
    pos = [i for i in labeled if samples[i].has_defect]
    neg = [i for i in labeled if not samples[i].has_defect]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds: List[List[int]] = [[] for _ in range(k)]
    for j, i in enumerate(pos + neg):
        folds[j % k].append(i)
    return folds


_safe = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(s: str) -> str:
    return _safe.sub("_", s)
