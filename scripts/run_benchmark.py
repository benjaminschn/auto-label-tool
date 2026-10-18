#!/usr/bin/env python
"""Simulate the few-label auto-labeling scenario on a public dataset and evaluate everything.

Example:
    python scripts/run_benchmark.py --dataset mt --n-defect 10 --n-free 5 --seed 0 --out runs/mt_s0
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autolabel import evaluate as E  # noqa: E402
from autolabel.data import load_dataset, make_split  # noqa: E402
from autolabel.pipeline import AutoLabelConfig, AutoLabeler  # noqa: E402
from autolabel.train import TrainConfig  # noqa: E402

DATASET_KW = {"mt": {}, "bsdata": {"scale": 0.5}, "deepcrack": {}, "cfd": {}}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_KW))
    ap.add_argument("--data-root", default=os.environ.get("AUTOLABEL_DATA", "data"))
    ap.add_argument("--cache", default=os.environ.get("AUTOLABEL_CACHE", "cache"))
    ap.add_argument("--n-defect", type=int, default=10)
    ap.add_argument("--n-free", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--k-folds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--st-iters", type=int, default=None)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--encoder", default="resnet18d")
    ap.add_argument("--p-synth", type=float, default=0.35)
    ap.add_argument("--p-procedural", type=float, default=0.0)
    ap.add_argument("--p-pseudo", type=float, default=0.4)
    ap.add_argument("--no-anomaly", action="store_true")
    ap.add_argument("--bg-ignore-quantile", type=float, default=0.98)
    ap.add_argument("--k-anomaly-patches", type=int, default=64)
    ap.add_argument("--teacher", default="fold", choices=["fold", "ensemble"])
    ap.add_argument("--no-rot90", action="store_true")
    ap.add_argument("--boundary-ignore", type=int, default=0)
    ap.add_argument("--max-unlabeled", type=int, default=None, help="subsample U (speed)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-maps", action="store_true", help="also keep the anomaly maps in results.pkl.gz")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "log.txt"), "a")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    samples = load_dataset(a.dataset, a.data_root, a.cache, **DATASET_KW[a.dataset])
    split = make_split(samples, a.n_defect, a.n_free, a.test_frac, a.seed)
    if a.max_unlabeled and len(split.unlabeled) > a.max_unlabeled:
        rng = np.random.RandomState(a.seed + 7)
        split.unlabeled = sorted(rng.choice(split.unlabeled, a.max_unlabeled, replace=False).tolist())
    log(f"{a.dataset}: {len(samples)} images; split {split.describe(samples)}")
    L = [samples[i] for i in split.labeled]
    U = [samples[i] for i in split.unlabeled]
    X = [samples[i] for i in split.test]

    tcfg = TrainConfig(encoder=a.encoder, crop=a.crop, iters=a.iters, p_synth=a.p_synth,
                       p_procedural=a.p_procedural, rot90=not a.no_rot90, boundary_ignore=a.boundary_ignore)
    cfg = AutoLabelConfig(train=tcfg, k_folds=a.k_folds, rounds=a.rounds, st_iters=a.st_iters,
                          p_pseudo=a.p_pseudo, use_anomaly=not a.no_anomaly, teacher=a.teacher, seed=a.seed,
                          bg_ignore_quantile=a.bg_ignore_quantile,
                          anomaly_patches_per_image=a.k_anomaly_patches)
    log("config: " + json.dumps(jsonable(cfg.to_dict())))
    res = AutoLabeler(cfg, log=log, workdir=None).run(L, U, X)

    gts_U = [s.mask for s in U]
    gts_X = [s.mask for s in X]
    gts_L = [s.mask for s in L]
    report = {"dataset": a.dataset, "args": vars(a), "split": split.describe(samples),
              "seconds": res["seconds"], "rounds": []}
    for rr in res["rounds"]:
        report["rounds"].append(E.eval_round(rr, gts_U, gts_X))
        if "member_U" in rr:   # a single cross-fitted member instead of the ensemble
            report["rounds"][-1]["U_single_member"] = E.eval_single_member(rr, gts_U)
        log(f"round {rr['round']}: U " + " ".join(f"{k}={report['rounds'][-1]['U'][k]:.3f}" for k in
                                                  ("tol_f1", "px_f1", "defect_recall", "defect_precision",
                                                   "img_recall", "img_specificity", "px_ap")))
    report["conformal"] = E.conformal_check(res["rounds"][-1], gts_U)
    report["self_assessment"] = E.self_assessment(res, gts_U)
    report["audit"] = E.audit_simulation(res, gts_U)
    report["review"] = E.review_curve(res, gts_U)
    report["anomaly_baseline"] = E.anomaly_baseline(res, gts_L, gts_U)
    with open(os.path.join(a.out, "report.json"), "w") as fh:
        json.dump(jsonable(report), fh, indent=1)
    if not a.save_maps:   # anomaly maps are large and only needed for the baseline computed above
        res.pop("anomaly_U", None)
        res.pop("anomaly_L", None)
    E.compact_results(res)
    with gzip.open(os.path.join(a.out, "results.pkl.gz"), "wb", compresslevel=3) as fh:
        pickle.dump({"res": res, "split": dataclasses.asdict(split), "dataset": a.dataset,
                     "dataset_kw": DATASET_KW[a.dataset]}, fh, protocol=4)
    log("saved " + a.out)


if __name__ == "__main__":
    main()
