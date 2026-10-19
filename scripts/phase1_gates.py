#!/usr/bin/env python
"""Decide the Phase-1 gates G1 and G2 (00 §6) from finished runs with the pre-registered rules in
docs/RESULTS.md §1, and print per-seed numbers (markdown).

G1 (self-training hurts recall where review does not look):
    low-risk strata = quintiles Q1-Q3 of the *round-0* review risk (fixed reference).
    rho_low(r) = 1 - (#bad images in Q1-Q3) / (#defective images in Q1-Q3) for round r.
    Delta = rho_low(1) - rho_low(0) per seed. G1 is POSITIVE for a dataset (self-training lowers
    the U recall in the low-risk strata) if Delta < 0 in >= 2 of 3 seeds AND mean Delta <= -2 pp.
    Also reported: K'(k*) of round 1 vs. round 0 (Go criterion 5: K' must drop >= 10 %).
G2 (threshold transfer OOF -> U):
    gap = tol-F1 on U at the U-oracle thresholds minus tol-F1 at the OOF thresholds (round 0).
    G2 is POSITIVE (transfer needs fixing) for a dataset if mean gap >= 5 pp.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.summarize import get, load  # noqa: E402


def rho_low(rep, rnd, q=(0, 1, 2)):
    st = get(rep, f"phase1.rounds.{rnd}.strata_ref.strata")
    if not st:
        return None
    bad = sum(st[i]["bad"] for i in q)
    nd = sum(st[i]["n_defective"] for i in q)
    return None if nd == 0 else 1 - bad / nd


def fmt(v, pct=True, d=1):
    if v is None or v != v:
        return "–"
    return f"{v * 100:.{d}f}" if pct else f"{v:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/*_full_s*")
    ap.add_argument("--cost-key", default="zeta*=0.02/c_review=0.25")
    a = ap.parse_args()
    runs = load(a.runs)
    print("### G1: Bild-Recall in Q1–Q3 (Runde-0-Risiko), Runde 1 vs. Runde 0\n")
    print("| Variante | Seed | ρ_low R0 | ρ_low R1 | Δ (pp) | K R0 | K R1 | K′(k*) R0 | K′(k*) R1 | Kosten min R0 | Kosten min R1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    g1 = {}
    for key in sorted(runs):
        reps = sorted(runs[key], key=lambda r: r["args"]["seed"])
        if not reps or len(get(reps[0], "phase1.rounds") or []) < 2:
            continue
        deltas = []
        for r in reps:
            r0, r1 = rho_low(r, 0), rho_low(r, 1)
            d = None if r0 is None or r1 is None else r1 - r0
            if d is not None:
                deltas.append(d)
            c = lambda j, p: get(r, f"phase1.rounds.{j}.kprime.cost.{a.cost_key}.risk.{p}")
            print(f"| {key} | {r['args']['seed']} | {fmt(r0)} | {fmt(r1)} | {fmt(d)} | "
                  f"{get(r, 'phase1.rounds.0.pool.K')} | {get(r, 'phase1.rounds.1.pool.K')} | "
                  f"{c(0, 'K_prime_at_k_star')} | {c(1, 'K_prime_at_k_star')} | "
                  f"{fmt(c(0, 'cost_min'), False, 0)} | {fmt(c(1, 'cost_min'), False, 0)} |")
        if deltas:
            pos = sum(d < 0 for d in deltas) >= 2 and np.mean(deltas) <= -0.02
            g1[key] = (pos, float(np.mean(deltas)), len(deltas))
    print("\n**G1-Entscheidung je Datensatz** (positiv = Self-Training senkt den Recall in Q1–Q3):\n")
    for k, (pos, m, n) in g1.items():
        print(f"* {k}: {'POSITIV' if pos else 'negativ'} (mittleres Δ = {m * 100:+.1f} pp, {n} Seeds)")
    print("\n### G2: Schwellenübertrag OOF → U (Runde 0)\n")
    print("| Variante | Seed | t_seed OOF | t_seed Oracle | t_ext OOF | t_ext Oracle | tol-F1 OOF-Schwelle | tol-F1 Oracle-Schwelle | Lücke (pp) | Δ Defekt-Recall (pp) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    g2 = {}
    for key in sorted(runs):
        reps = sorted(runs[key], key=lambda r: r["args"]["seed"])
        gaps = []
        for r in reps:
            G = lambda k: get(r, f"phase1.rounds.0.g2_threshold_gap.{k}")
            if G("tol_f1_gap") is None:
                continue
            gaps.append(G("tol_f1_gap"))
            print(f"| {key} | {r['args']['seed']} | {fmt(G('t_seed_oof'), False, 2)} | {fmt(G('t_seed_oracle'), False, 2)} | "
                  f"{fmt(G('t_ext_oof'), False, 2)} | {fmt(G('t_ext_oracle'), False, 2)} | "
                  f"{fmt(get(r, 'rounds.0.U.tol_f1'))} | {fmt(get(r, 'rounds.0.U_oracle_threshold.tol_f1'))} | "
                  f"{fmt(G('tol_f1_gap'))} | {fmt(G('defect_recall_gap'))} |")
        if gaps:
            g2[key] = (float(np.mean(gaps)) >= 0.05, float(np.mean(gaps)), len(gaps))
    print("\n**G2-Entscheidung je Datensatz** (positiv = Übertrag verbesserungsbedürftig, mittlere Lücke ≥ 5 pp):\n")
    for k, (pos, m, n) in g2.items():
        print(f"* {k}: {'POSITIV' if pos else 'negativ'} (mittlere Lücke = {m * 100:.1f} pp, {n} Seeds)")


if __name__ == "__main__":
    main()
