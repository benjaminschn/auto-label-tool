#!/usr/bin/env python
"""Aggregate report.json files of benchmark runs into markdown tables (mean ± std over seeds).

Run directories are expected to be named ``<dataset>_<variant>_s<seed>``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def load(pattern):
    runs = defaultdict(list)
    for d in sorted(glob.glob(pattern)):
        f = os.path.join(d, "report.json")
        if not os.path.exists(f):
            continue
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(d.rstrip("/")))
        if not m:
            continue
        runs[m.group(1)].append(json.load(open(f)))
    return runs


def ms(vals, pct=True, digits=1):
    v = np.array([x for x in vals if x is not None and x == x], np.float64)
    if len(v) == 0:
        return "–"
    f = 100.0 if pct else 1.0
    if len(v) == 1:
        return f"{v[0] * f:.{digits}f}"
    return f"{v.mean() * f:.{digits}f} ± {v.std(ddof=1) * f:.{digits}f}"


def get(d, path):
    for p in path.split("."):
        if isinstance(d, list):
            p = int(p)
        elif p not in d:
            return None
        d = d[p]
    return d


QUALITY_COLS = [("tol F1", "tol_f1"), ("px F1", "px_f1"), ("px AP", "px_ap"),
                ("Defekt-Recall", "defect_recall"), ("Defekt-Precision", "defect_precision"),
                ("Bild-Recall", "img_recall"), ("Bild-Spezifität", "img_specificity")]


def quality_table(runs, rows):
    """rows: list of (label, variant key, json path prefix)."""
    out = ["| Variante | n | " + " | ".join(c for c, _ in QUALITY_COLS) + " |",
           "|---|---|" + "---|" * len(QUALITY_COLS)]
    for label, key, prefix in rows:
        reps = runs.get(key, [])
        if not reps:
            continue
        vals = [[get(r, f"{prefix}.{k}") for r in reps] for _, k in QUALITY_COLS]
        out.append(f"| {label} | {len(reps)} | " + " | ".join(ms(v) for v in vals) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/home/user/runs/*")
    a = ap.parse_args()
    runs = load(a.runs)
    for key, reps in sorted(runs.items()):
        print(f"## {key}  ({len(reps)} seeds)")
        rows = [(f"Runde {i}", key, f"rounds.{i}.U") for i in range(len(reps[0]["rounds"]))]
        rows.append(("Runde 0, Einzelmodell", key, "rounds.0.U_single_member"))
        rows.append(("Letzte Runde, recall-sicher", key, f"rounds.{len(reps[0]['rounds']) - 1}.U_safe"))
        rows.append(("Letzte Runde, Testset T", key, f"rounds.{len(reps[0]['rounds']) - 1}.X"))
        rows.append(("Normalitätsmodell allein", key, "anomaly_baseline"))
        print(quality_table(runs, rows))
        print()


if __name__ == "__main__":
    main()
