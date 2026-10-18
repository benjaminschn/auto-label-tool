#!/usr/bin/env python
"""Recompute report.json of a finished benchmark run from results.pkl.gz (no retraining)."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autolabel import evaluate as E  # noqa: E402
from autolabel.data import load_dataset  # noqa: E402
from scripts.run_benchmark import jsonable  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data-root", default=os.environ.get("AUTOLABEL_DATA", "data"))
    ap.add_argument("--cache", default=os.environ.get("AUTOLABEL_CACHE", "cache"))
    a = ap.parse_args()
    for run in a.runs:
        with gzip.open(os.path.join(run, "results.pkl.gz"), "rb") as fh:
            d = pickle.load(fh)
        res, split = d["res"], d["split"]
        samples = load_dataset(d["dataset"], a.data_root, a.cache, **d["dataset_kw"])
        gts_U = [samples[i].mask for i in split["unlabeled"]]
        gts_X = [samples[i].mask for i in split["test"]]
        gts_L = [samples[i].mask for i in split["labeled"]]
        rep_path = os.path.join(run, "report.json")
        report = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
        old_rounds = report.get("rounds", [])
        report["rounds"] = []
        for j, rr in enumerate(res["rounds"]):
            r = E.eval_round(rr, gts_U, gts_X)
            if j < len(old_rounds) and "U_single_member" in old_rounds[j]:
                r["U_single_member"] = old_rounds[j]["U_single_member"]
            report["rounds"].append(r)
        report["conformal"] = E.conformal_check(res["rounds"][-1], gts_U)
        report["self_assessment"] = E.self_assessment(res, gts_U)
        report["audit"] = E.audit_simulation(res, gts_U)
        report["review"] = E.review_curve(res, gts_U)
        if "anomaly_U" in res:
            report["anomaly_baseline"] = E.anomaly_baseline(res, gts_L, gts_U)
        with open(rep_path, "w") as fh:
            json.dump(jsonable(report), fh, indent=1)
        print("re-evaluated", run)


if __name__ == "__main__":
    main()
