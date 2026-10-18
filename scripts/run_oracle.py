#!/usr/bin/env python
"""Upper bound: a model trained on the *full* ground truth of the pool (L ∪ U), evaluated on the test set T.

Uses the same split, architecture and training recipe as the benchmark; thresholds are chosen on a
10 % hold-out of the pool. Comparing T-metrics of the auto-labeler (rounds.*.X in report.json) with
this run shows how much of the fully supervised performance is reached with ~10 masks.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autolabel import metrics as M  # noqa: E402
from autolabel.data import load_dataset, make_split  # noqa: E402
from autolabel.infer import predict_image  # noqa: E402
from autolabel.postprocess import hysteresis, select_thresholds  # noqa: E402
from autolabel.train import TrainConfig, TrainData, train_member  # noqa: E402
from scripts.run_benchmark import DATASET_KW, jsonable  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", default=os.environ.get("AUTOLABEL_DATA", "data"))
    ap.add_argument("--cache", default=os.environ.get("AUTOLABEL_CACHE", "cache"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    samples = load_dataset(a.dataset, a.data_root, a.cache, **DATASET_KW[a.dataset])
    split = make_split(samples, 10, 5, 0.2, a.seed)
    pool = split.labeled + split.unlabeled
    rng = np.random.RandomState(a.seed)
    rng.shuffle(pool)
    n_val = max(10, len(pool) // 10)
    val, train = pool[:n_val], pool[n_val:]
    cfg = TrainConfig(iters=a.iters, p_synth=0.0, seed=a.seed)
    model = train_member(TrainData(labeled=[(samples[i].image, samples[i].mask) for i in train]), cfg,
                         log=lambda m: print(m, flush=True))
    pv = [predict_image(model, samples[i].image) for i in val]
    th, _ = select_thresholds(pv, [samples[i].mask for i in val])
    px = [predict_image(model, samples[i].image) for i in split.test]
    gx = [samples[i].mask for i in split.test]
    mx = [hysteresis(p, th["t_seed"], th["t_ext"], int(th["min_area"])) for p in px]
    out = M.summarize(mx, gx, probs=px)
    out["px_ap"] = M.average_precision_px(px, gx)
    rep = {"dataset": a.dataset, "seed": a.seed, "n_train": len(train), "thresholds": th, "X": out,
           "config": dataclasses.asdict(cfg)}
    json.dump(jsonable(rep), open(os.path.join(a.out, "oracle.json"), "w"), indent=1)
    print(json.dumps(jsonable(rep["X"]), indent=1))


if __name__ == "__main__":
    main()
