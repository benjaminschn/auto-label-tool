#!/usr/bin/env python
"""Post-hoc control for gate G1 (NOT pre-registered): separate the threshold change from the model change.

The pre-registered G1 compares round 1 and round 0 at their own OOF-selected thresholds. If round 1
selects a much lower threshold, its recall rises trivially while false components (phi) explode. Here
round 1 probabilities are additionally thresholded with the *round-0* thresholds ("R1@T0"), and
rho_low (image recall in Q1-Q3 of the round-0 risk), zeta and phi are reported for R0, R1, R1@T0.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from autolabel import evaluate as E  # noqa: E402
from autolabel.data import load_dataset  # noqa: E402
from autolabel.postprocess import hysteresis  # noqa: E402


def rho_low(b, defective, risk_ref, q=3, n_strata=5):
    order = np.argsort(np.asarray(risk_ref), kind="stable")
    strata = np.empty(len(order), np.int64)
    strata[order] = (np.arange(len(order)) * n_strata) // len(order)
    sel = (strata < q) & defective
    return float(1 - b[sel].mean()) if sel.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/*_full_s*")
    ap.add_argument("--data-root", default=os.environ.get("AUTOLABEL_DATA", "data"))
    ap.add_argument("--cache", default=os.environ.get("AUTOLABEL_CACHE", "cache"))
    a = ap.parse_args()
    print("| Lauf | ρ_low R0 | ρ_low R1 | ρ_low R1@T0 | ζ R0 | ζ R1 | ζ R1@T0 | φ R0 | φ R1 | φ R1@T0 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    cache = {}
    for run in sorted(glob.glob(a.runs)):
        f = os.path.join(run, "results.pkl.gz")
        if not os.path.exists(f):
            continue
        d = pickle.load(gzip.open(f))
        res, split = d["res"], d["split"]
        if len(res["rounds"]) < 2 or "risk_U" not in res["rounds"][0]:
            continue
        key = (d["dataset"], tuple(sorted(d["dataset_kw"].items())))
        if key not in cache:
            cache = {key: load_dataset(d["dataset"], a.data_root, a.cache, **d["dataset_kw"])}
        S = cache[key]
        gts = [S[i].mask for i in split["unlabeled"]]
        r0, r1 = res["rounds"][0], res["rounds"][1]
        th = r0["thresholds"]
        m10 = [hysteresis(E.as_prob(p), th["t_seed"], th["t_ext"], int(th["min_area"])) for p in r1["prob_U"]]
        out = []
        for masks in (r0["mask_U"], r1["mask_U"], m10):
            pe = E.pool_errors(masks, gts)
            out.append((rho_low(pe["b"], pe["defective"], r0["risk_U"]), pe["b"].mean(), pe["F"].mean()))
        f = lambda v, p=True: f"{v * 100:.1f}" if p else f"{v:.2f}"
        print(f"| {os.path.basename(run)} | " + " | ".join(f(o[0]) for o in out) + " | " +
              " | ".join(f(o[1]) for o in out) + " | " + " | ".join(f(o[2], False) for o in out) + " |")


if __name__ == "__main__":
    main()
