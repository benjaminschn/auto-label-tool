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


# ----------------------------------------------------------------------------- Phase 1 (00 §6)

def _rounds(reps):
    return len(reps[0].get("phase1", {}).get("rounds", []))


def phase1_pool_table(runs, keys):
    out = ["| Variante | Runde | n | ζ (Fehlbildrate) | K | φ (Fehlkomp./Bild) | φ sauber | Bild-Recall ρ_b | SegF1 | Recall S | Recall M | Recall L |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in keys:
        reps = [r for r in runs.get(key, []) if "phase1" in r]
        if not reps:
            continue
        for j in range(_rounds(reps)):
            P = f"phase1.rounds.{j}"
            sc = lambda c: ms([get(r, f"{P}.pool.size_classes.{c}.recall") for r in reps])
            nn = lambda c: ms([get(r, f"{P}.pool.size_classes.{c}.n") for r in reps], pct=False, digits=0)
            out.append(f"| {key} | {j} | {len(reps)} | {ms([get(r, f'{P}.pool.zeta') for r in reps])} | "
                       f"{ms([get(r, f'{P}.pool.K') for r in reps], pct=False, digits=0)} | "
                       f"{ms([get(r, f'{P}.pool.phi') for r in reps], pct=False, digits=2)} | "
                       f"{ms([get(r, f'{P}.pool.phi_clean') for r in reps], pct=False, digits=2)} | "
                       f"{ms([get(r, f'{P}.pool.img_recall_rho_b') for r in reps])} | "
                       f"{ms([get(r, f'{P}.seg_f1') for r in reps])} | "
                       f"{sc('S')} (n={nn('S')}) | {sc('M')} (n={nn('M')}) | {sc('L')} (n={nn('L')}) |")
    return "\n".join(out)


def kprime_table(runs, keys):
    fr = None
    out = []
    for key in keys:
        reps = [r for r in runs.get(key, []) if "phase1" in r]
        if not reps:
            continue
        fr = fr or get(reps[0], "phase1.rounds.0.kprime.fracs")
        if not out:
            out = ["| Variante | Runde | Reihenfolge | " + " | ".join(f"K′ nach {f:.0%}" for f in fr) +
                   " | AUC K′/K |", "|---|---|---|" + "---|" * (len(fr) + 1)]
        for j in range(_rounds(reps)):
            for order, okey, akey in (("Risiko", "risk", "risk"), ("Zufall", "random_mean", "random"),
                                      ("Orakel", "oracle", "oracle")):
                vals = [[get(r, f"phase1.rounds.{j}.kprime.K_prime.{okey}")[i] for r in reps
                         if get(r, f"phase1.rounds.{j}.kprime.K_prime.{okey}")] for i in range(len(fr))]
                out.append(f"| {key} | {j} | {order} | " + " | ".join(ms(v, pct=False, digits=1) for v in vals) +
                           f" | {ms([get(r, f'phase1.rounds.{j}.kprime.auc_K_prime.{akey}') for r in reps], pct=False, digits=3)} |")
    return "\n".join(out)


def cost_table(runs, keys):
    out = ["| Variante | Runde | Ziel / Kostenverhältnis | n₉₀(K) ohne Review | Kosten min (Risiko) | k* (Risiko) | Kosten min (Zufall) | Kosten min (Orakel) | Ersparnis Risiko vs. k=0 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for key in keys:
        reps = [r for r in runs.get(key, []) if "phase1" in r]
        if not reps:
            continue
        for j in range(_rounds(reps)):
            cks = sorted(get(reps[0], f"phase1.rounds.{j}.kprime.cost") or {})
            for ck in cks:
                C = lambda path: [get(r, f"phase1.rounds.{j}.kprime.cost.{ck}.{path}") for r in reps]
                sav = [1 - a / b for a, b in zip(C("risk.cost_min"), C("risk.cost_k0")) if a is not None and b]
                out.append(f"| {key} | {j} | {ck} | {ms(C('risk.n90_k0'), pct=False, digits=0)} | "
                           f"{ms(C('risk.cost_min'), pct=False, digits=0)} | {ms(C('risk.k_star'), pct=False, digits=0)} | "
                           f"{ms(C('random.cost_min'), pct=False, digits=0)} | {ms(C('oracle.cost_min'), pct=False, digits=0)} | "
                           f"{ms(sav)} |")
    return "\n".join(out)


def g1_table(runs, keys):
    out = ["| Variante | Runde | Bild-Recall je Risiko-Quintil (Runde-0-Risiko; Q1 = niedrigstes Risiko) | Anteil Fehlbilder unter Median des eigenen Risikos |",
           "|---|---|---|---|"]
    for key in keys:
        reps = [r for r in runs.get(key, []) if "phase1" in r and get(r, "phase1.rounds.0.strata_ref")]
        if not reps:
            continue
        for j in range(_rounds(reps)):
            ns = len(get(reps[0], f"phase1.rounds.{j}.strata_ref.strata"))
            cells = []
            for q in range(ns):
                v = [get(r, f"phase1.rounds.{j}.strata_ref.strata.{q}.img_recall") for r in reps]
                cells.append(f"Q{q + 1}: {ms(v)}")
            out.append(f"| {key} | {j} | " + " · ".join(cells) + f" | "
                       f"{ms([get(r, f'phase1.rounds.{j}.share_bad_below_median_risk') for r in reps])} |")
    return "\n".join(out)


def g2_table(runs, keys):
    out = ["| Variante | Runde | t_seed OOF | t_seed U-Oracle | t_ext OOF | t_ext U-Oracle | Δ tol-F1 (Oracle − OOF) | Δ Defekt-Recall | Δ Defekt-Precision |",
           "|---|---|---|---|---|---|---|---|---|"]
    for key in keys:
        reps = [r for r in runs.get(key, []) if "phase1" in r]
        if not reps:
            continue
        for j in range(_rounds(reps)):
            G = lambda k: [get(r, f"phase1.rounds.{j}.g2_threshold_gap.{k}") for r in reps]
            out.append(f"| {key} | {j} | {ms(G('t_seed_oof'), pct=False, digits=2)} | {ms(G('t_seed_oracle'), pct=False, digits=2)} | "
                       f"{ms(G('t_ext_oof'), pct=False, digits=2)} | {ms(G('t_ext_oracle'), pct=False, digits=2)} | "
                       f"{ms(G('tol_f1_gap'))} | {ms(G('defect_recall_gap'))} | {ms(G('defect_precision_gap'))} |")
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
    print("\n## Phase 1: Pool-Qualität (ζ, φ, Größenklassen)\n")
    print(phase1_pool_table(runs, keys))
    print("\n## Phase 1: K′(k) – Restfehlbilder nach Review der k riskantesten Bilder\n")
    print(kprime_table(runs, keys))
    print("\n## Phase 1: Kosten(k) = k·c_review + n₉₀(K′(k))·c_audit (Einheiten c_audit, δ = 0,05, Power 0,9)\n")
    print(cost_table(runs, keys))
    print("\n## Gate G1: Bild-Recall in Risiko-Strata je Runde\n")
    print(g1_table(runs, keys))
    print("\n## Gate G2: OOF-Schwelle vs. U-Oracle-Schwelle\n")
    print(g2_table(runs, keys))


if __name__ == "__main__":
    main()
