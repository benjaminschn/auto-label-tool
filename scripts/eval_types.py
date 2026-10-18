#!/usr/bin/env python
"""Evaluate few-shot defect-type assignment on the pseudo-labels of a finished benchmark run.

Every predicted defect component on U that overlaps a ground-truth defect gets the true type of its
image (datasets like Magnetic-tile have one defect type per image). Reports accuracy, macro-F1 and the
coverage / size of the conformal prediction sets.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autolabel import metrics as M  # noqa: E402
from autolabel.classify import DefectTypeAssigner  # noqa: E402
from autolabel.data import load_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data-root", default=os.environ.get("AUTOLABEL_DATA", "data"))
    ap.add_argument("--cache", default=os.environ.get("AUTOLABEL_CACHE", "cache"))
    ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    with gzip.open(os.path.join(a.run, "results.pkl.gz"), "rb") as fh:
        d = pickle.load(fh)
    res, split = d["res"], d["split"]
    samples = load_dataset(d["dataset"], a.data_root, a.cache, **d["dataset_kw"])
    L = [samples[i] for i in split["labeled"]]
    U = [samples[i] for i in split["unlabeled"]]
    masks = res["rounds"][-1]["mask_U"]
    ta = DefectTypeAssigner(alpha=a.alpha)
    feats_U = [ta.feat.describe(s.image, m) for s, m in zip(U, masks)]
    X_ref = np.vstack([f for f, c in feats_U if len(c)]) if any(len(c) for _, c in feats_U) else None
    ta.fit([s.image for s in L], [s.mask for s in L], [s.kind for s in L], X_ref)
    y_true, y_pred, covered, set_sizes = [], [], [], []
    for s, m, fc in zip(U, masks, feats_U):
        if not s.has_defect or not m.any():
            continue
        gt_d = M.dilate(s.mask, 2)
        for comp in ta.predict(s.image, m, feats=fc):
            if not (comp["mask"] & gt_d).any():
                continue  # false alarm: no true type
            y_true.append(s.kind)
            y_pred.append(comp["type"])
            covered.append(s.kind in comp["prediction_set"])
            set_sizes.append(len(comp["prediction_set"]))
    from sklearn.metrics import confusion_matrix, f1_score
    classes = ta.clf.classes
    out = {
        "n_components": len(y_true), "classes": classes,
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else None,
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro")) if y_true else None,
        "conformal_alpha": a.alpha, "set_coverage": float(np.mean(covered)) if covered else None,
        "mean_set_size": float(np.mean(set_sizes)) if set_sizes else None,
        "singleton_rate": float(np.mean(np.array(set_sizes) == 1)) if set_sizes else None,
        "singleton_accuracy": float(np.mean([t == p for t, p, z in zip(y_true, y_pred, set_sizes) if z == 1]))
        if set_sizes and any(z == 1 for z in set_sizes) else None,
        "confusion": confusion_matrix(y_true, y_pred, labels=classes).tolist() if y_true else None,
        "n_calibration_components": ta.n_cal,
    }
    print(json.dumps(out, indent=1))
    with open(os.path.join(a.run, "types.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
