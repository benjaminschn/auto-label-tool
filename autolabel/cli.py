"""Command line interface for real projects.

    # 1) auto-label: images/ holds all images, masks/ the few existing pixel masks (same file stem)
    autolabel run --images images/ --masks masks/ --out run1/

    # 2) a human checks the images listed in run1/audit_sample.txt and stores corrected masks in audit/
    # 3) statistically valid quality estimate of all auto-labels
    autolabel estimate --run run1/ --audit-masks audit/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time

import numpy as np
from PIL import Image

from . import metrics as M
from . import ppi
from .data import load_folder
from .pipeline import AutoLabelConfig, AutoLabeler
from .quality import true_quality
from .train import TrainConfig


def _save_mask(path, m):
    Image.fromarray(m.astype(np.uint8) * 255).save(path)


def _overlay(img, mask, prob):
    import cv2
    vis = img.copy()
    heat = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    vis = (0.75 * vis + 0.25 * np.dstack([heat, np.zeros_like(heat), np.zeros_like(heat)])).astype(np.uint8)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 1)
    return np.hstack([img, vis])


def cmd_run(a):
    samples = load_folder(a.images, a.masks, a.mask_suffix, scale=a.scale)
    L = [s for s in samples if s.mask is not None]
    U = [s for s in samples if s.mask is None]
    if not L:
        sys.exit("no labeled images found (masks must share the image file stem)")
    print(f"{len(L)} labeled ({sum(s.has_defect for s in L)} with defects), {len(U)} unlabeled")
    tcfg = TrainConfig(encoder=a.encoder, iters=a.iters, crop=a.crop, rot90=not a.no_rot90)
    cfg = AutoLabelConfig(train=tcfg, k_folds=min(a.k_folds, len(L)), rounds=a.rounds, alpha=a.alpha,
                          use_anomaly=not a.no_anomaly, seed=a.seed)
    os.makedirs(a.out, exist_ok=True)
    res = AutoLabeler(cfg, workdir=os.path.join(a.out, "models")).run(L, U)
    last = res["rounds"][-1]
    for sub in ("labels", "labels_recall_safe", "ignore", "prob", "overlays"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    th = last["thresholds"]
    for i, s in enumerate(U):
        p = last["prob_U"][i].astype(np.float32)
        m = last["mask_U"][i]
        _save_mask(os.path.join(a.out, "labels", s.id + ".png"), m)
        _save_mask(os.path.join(a.out, "labels_recall_safe", s.id + ".png"), last["mask_safe_U"][i])
        ign = (p >= cfg.ignore_ratio * th["t_ext"]) & ~m
        _save_mask(os.path.join(a.out, "ignore", s.id + ".png"), ign)
        Image.fromarray((p * 255).astype(np.uint8)).save(os.path.join(a.out, "prob", s.id + ".png"))
        Image.fromarray(_overlay(s.image, m, p)).save(os.path.join(a.out, "overlays", s.id + ".jpg"), quality=90)
    feats = res["features_U"]
    order = np.argsort(-np.asarray(res["risk_U"]))
    with open(os.path.join(a.out, "review_queue.csv"), "w", newline="") as fh:
        cols = ["rank", "image", "risk", "exp_dice", "p_missed_defect", "p_has_defect", "n_pred_comps",
                "exp_false_comps", "member_dice", "anomaly_out", "p_max"]
        w = csv.writer(fh)
        w.writerow(cols)
        for r, i in enumerate(order):
            f = feats[i]
            w.writerow([r + 1, U[i].id, f"{res['risk_U'][i]:.4f}"] + [f"{f.get(c, float('nan')):.4f}" for c in cols[3:]])
    rng = np.random.RandomState(a.seed + 12345)
    audit = sorted(rng.choice(len(U), min(a.audit_size, len(U)), replace=False).tolist())
    with open(os.path.join(a.out, "audit_sample.txt"), "w") as fh:
        fh.write("\n".join(U[i].id for i in audit) + "\n")
    E = {k: sum(f[k] for f in feats) for k in ("exp_tp", "exp_fp", "exp_fn", "exp_hits", "exp_defects",
                                                "exp_false_comps", "n_pred_comps")}
    report = {
        "n_labeled": len(L), "n_unlabeled": len(U), "thresholds": th,
        "cross_validated_metrics_on_labeled": last["oof_metrics"],
        "recall_guarantee": {
            "alpha": cfg.alpha, "n_labeled_defects": int(len(last["defect_scores_oof"])),
            "t_conformal": last["t_conformal"], "t_pac": last["t_pac"],
            "certified_recall_at_t_seed (marginal, PAC)": list(last["certified_recall_at_t_seed"]),
            "note": "labels_recall_safe/ uses seed threshold min(t_seed, t_conformal); "
                    "t=0 means too few labeled defects to certify the requested recall",
        },
        "self_assessment (no ground truth, plug-in)": {
            "px_precision": E["exp_tp"] / max(1e-9, E["exp_tp"] + E["exp_fp"]),
            "px_recall": E["exp_tp"] / max(1e-9, E["exp_tp"] + E["exp_fn"]),
            "defect_recall": E["exp_hits"] / max(1e-9, E["exp_defects"]),
            "defect_precision": 1 - E["exp_false_comps"] / max(1e-9, E["n_pred_comps"]),
            "mean_expected_dice": float(np.mean([f["exp_dice"] for f in feats])),
        },
        "next_step": "label the images in audit_sample.txt (random sample!) and run `autolabel estimate`",
    }
    with open(os.path.join(a.out, "quality_report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    state = {"ids": [s.id for s in U], "features": feats, "masks": last["mask_U"], "audit": audit,
             "tol": cfg.tol, "merge_radius": cfg.merge_radius, "config": cfg.to_dict(), "time": time.time()}
    with open(os.path.join(a.out, "state.pkl"), "wb") as fh:
        pickle.dump(state, fh, protocol=4)
    print(json.dumps(report, indent=1, default=float))


def cmd_estimate(a):
    with open(os.path.join(a.run, "state.pkl"), "rb") as fh:
        st = pickle.load(fh)
    ids, feats, masks = st["ids"], st["features"], st["masks"]
    idx = {k: i for i, k in enumerate(ids)}
    audited, truth = [], []
    for f in sorted(os.listdir(a.audit_masks)):
        stem = os.path.splitext(f)[0]
        if stem in idx:
            g = np.array(Image.open(os.path.join(a.audit_masks, f)).convert("L")) > 0
            i = idx[stem]
            audited.append(i)
            truth.append(true_quality(masks[i], g, st["tol"], st["merge_radius"]))
    if len(audited) < 5:
        sys.exit("need at least 5 audited images")
    planned = set(st["audit"])
    if not set(audited) <= planned:
        print("WARNING: audited images are not (only) the random audit sample - estimates may be biased")
    rest = np.setdiff1d(np.arange(len(ids)), audited)
    F = lambda k, ix: np.array([feats[i][k] for i in ix], np.float64)
    T = lambda k: np.array([t[k] for t in truth], np.float64)
    out = {"n_audited": len(audited), "n_pool": len(ids), "confidence": 1 - a.alpha}
    defs = {
        "px_recall": (T("tp"), T("tp") + T("fn"), "exp_tp", ("exp_tp", "exp_fn")),
        "px_precision": (T("tp"), T("tp") + T("fp"), "exp_tp", ("exp_tp", "exp_fp")),
        "defect_recall": (T("gt_comp_hit"), T("gt_comp"), "exp_hits", ("exp_defects",)),
    }
    for name, (num, den, fnum, fden) in defs.items():
        fa = F(fnum, audited)
        fb = sum(F(k, audited) for k in fden)
        ra = F(fnum, rest)
        rb = sum(F(k, rest) for k in fden)
        e, lo, hi = ppi.estimate_ratio(num, den, fa, fb, ra, rb, a.alpha)
        ec, loc, hic = ppi.classical_ratio(num, den, a.alpha, len(ids))
        out[name] = {"ppi": [e, max(0, lo), min(1, hi)], "audit_only": [ec, max(0, loc), min(1, hic)]}
    e, lo, hi = ppi.estimate_mean(T("dice_tol"), F("exp_dice", audited), F("exp_dice", rest), a.alpha)
    out["mean_dice_tol"] = {"ppi": [e, max(0, lo), min(1, hi)],
                            "audit_only": list(ppi.classical_mean(T("dice_tol"), a.alpha, len(ids)))}
    print(json.dumps(out, indent=1))
    with open(os.path.join(a.run, "quality_estimate.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="autolabel")
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run", help="auto-label all unlabeled images")
    r.add_argument("--images", required=True)
    r.add_argument("--masks", required=True, help="folder with the existing masks (file stem = image stem)")
    r.add_argument("--mask-suffix", default="")
    r.add_argument("--out", required=True)
    r.add_argument("--scale", type=float, default=1.0)
    r.add_argument("--encoder", default="resnet18d")
    r.add_argument("--iters", type=int, default=600)
    r.add_argument("--crop", type=int, default=192)
    r.add_argument("--k-folds", type=int, default=4)
    r.add_argument("--rounds", type=int, default=2)
    r.add_argument("--alpha", type=float, default=0.1)
    r.add_argument("--audit-size", type=int, default=30)
    r.add_argument("--no-anomaly", action="store_true")
    r.add_argument("--no-rot90", action="store_true")
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(fn=cmd_run)
    e = sp.add_parser("estimate", help="audit-based confidence intervals for the label quality")
    e.add_argument("--run", required=True)
    e.add_argument("--audit-masks", required=True)
    e.add_argument("--alpha", type=float, default=0.1)
    e.set_defaults(fn=cmd_estimate)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
