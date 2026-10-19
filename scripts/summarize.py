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
    """Nested lookup with "." as separator; keys may themselves contain dots (e.g. "alpha=0.1",
    "risk/0.05"), so at each level the longest matching key wins."""
    parts = path.split(".")
    i = 0
    while i < len(parts):
        if d is None:
            return None
        if isinstance(d, list):
            try:
                d = d[int(parts[i])]
            except (ValueError, IndexError):
                return None
            i += 1
            continue
        if not isinstance(d, dict):
            return None
        for j in range(len(parts), i, -1):
            k = ".".join(parts[i:j])
            if k in d:
                d, i = d[k], j
                break
        else:
            return None
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


def conformal_table(runs, keys):
    out = ["| Datensatz | Ziel-Recall | Defekt-Recall auf U (konforme Schwelle) | Defekt-Recall (PAC-Schwelle) | Schwelle konform | gelabelte Defekte |",
           "|---|---|---|---|---|---|"]
    for key in keys:
        reps = runs.get(key, [])
        if not reps:
            continue
        for a in ("0.05", "0.1", "0.2"):
            c = [get(r, f"conformal.alpha={a}") for r in reps]
            c = [x for x in c if x]
            if not c:
                continue
            nl = [get(r, "rounds.0.n_labeled_defects") for r in reps]
            out.append(f"| {key} | {1 - float(a):.0%} | {ms([x['realised_recall_conformal'] for x in c])} | "
                       f"{ms([x['realised_recall_pac'] for x in c])} | "
                       f"{ms([x['t_conformal'] for x in c], pct=False, digits=2)} | {ms(nl, pct=False, digits=0)} |")
    return "\n".join(out)


def self_assessment_table(runs, keys):
    rows = [("Pixel-Precision", "px_precision"), ("Pixel-Recall", "px_recall"),
            ("Defekt-Recall", "defect_recall"), ("Defekt-Precision", "defect_precision"),
            ("mittlerer Dice (tol)", "mean_dice_tol")]
    out = ["| Datensatz | Größe | wahr | Selbsteinschätzung (ohne GT) |", "|---|---|---|---|"]
    for key in keys:
        reps = runs.get(key, [])
        for label, k in rows:
            t = [get(r, f"self_assessment.{k}_true") for r in reps]
            e = [get(r, f"self_assessment.{k}_selfest") for r in reps]
            if any(x is not None for x in t):
                out.append(f"| {key} | {label} | {ms(t)} | {ms(e)} |")
    return "\n".join(out)


def signal_table(runs, keys):
    sig = ["risk", "p_missed_defect", "anomaly_out", "inv_member_dice", "inv_exp_dice"]
    out = ["| Datensatz | Ziel | " + " | ".join(sig) + " |", "|---|---|" + "---|" * len(sig)]
    for key in keys:
        reps = runs.get(key, [])
        for target in ("missed_defect", "bad_label"):
            vals = [ms([get(r, f"self_assessment.auroc_{s}_for_{target}") for r in reps], pct=False, digits=2)
                    for s in sig]
            out.append(f"| {key} | {target} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def audit_table(runs, keys, ns=(10, 20, 40, 80)):
    mets = ["px_recall", "px_precision", "defect_recall", "defect_precision", "mean_dice_tol"]
    out = ["| Datensatz | Metrik | n Audit | Abdeckung PPI | Abdeckung klassisch | Breite PPI | Breite klassisch |",
           "|---|---|---|---|---|---|---|"]
    for key in keys:
        reps = runs.get(key, [])
        for m in mets:
            for n in ns:
                p = [get(r, f"audit.{m}/n={n}/ppi") for r in reps]
                c = [get(r, f"audit.{m}/n={n}/classical") for r in reps]
                p, c = [x for x in p if x], [x for x in c if x]
                if not p:
                    continue
                out.append(f"| {key} | {m} | {n} | {ms([x['coverage'] for x in p])} | "
                           f"{ms([x['coverage'] for x in c])} | {ms([x['width'] for x in p])} | "
                           f"{ms([x['width'] for x in c])} |")
    return "\n".join(out)


def review_table(runs, keys, fracs=("0.00", "0.05", "0.10", "0.20", "0.30")):
    out = ["| Datensatz | Reihenfolge | " + " | ".join(f"{float(f):.0%} geprüft" for f in fracs) + " |",
           "|---|---|" + "---|" * len(fracs)]
    for key in keys:
        reps = runs.get(key, [])
        for order in ("risk", "random", "oracle"):
            vals = [ms([get(r, f"review.{order}/{f}.defect_recall") for r in reps]) for f in fracs]
            out.append(f"| {key} | {order} (Defekt-Recall) | " + " | ".join(vals) + " |")
        for order in ("risk", "random"):
            vals = [ms([get(r, f"review.{order}/{f}.tol_f1") for r in reps]) for f in fracs]
            out.append(f"| {key} | {order} (tol F1) | " + " | ".join(vals) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/*")
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
    keys = sorted(runs)
    print("## Recall-Garantie\n")
    print(conformal_table(runs, keys))
    print("\n## Selbsteinschätzung\n")
    print(self_assessment_table(runs, keys))
    print("\n## Fehler-Signale (AUROC)\n")
    print(signal_table(runs, keys))
    print("\n## Audit\n")
    print(audit_table(runs, keys))
    print("\n## Review\n")
    print(review_table(runs, keys))


if __name__ == "__main__":
    main()
