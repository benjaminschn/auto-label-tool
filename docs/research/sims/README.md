# Simulationen zu Bericht 04

Kleine Monte-Carlo-Studien zur Abdeckung von Audit-Konfidenzschranken (Tabellen S1–S3 in
[`../04_guarantees_measurement.md`](../04_guarantees_measurement.md)). Laufzeit: Minuten auf der CPU.

```bash
python docs/research/sims/sim_recall_cert.py   # S1: Defekt-Recall-LCB (t/Delta, PPI λ=1, Betting-Fieller)
python docs/research/sims/sim2.py              # IPW-CLT-Fieller mit PPS-Design
python docs/research/sims/sim3.py              # S2: Fehlbildrate ζ (exakt hypergeometrisch, WSR-Betting)
```

Prototypen, nicht Teil des Pakets; sie dienen als Ausgangspunkt für `autolabel/ppi.py` v2.

## Red-Team-Prüfungen (Bericht 07)

```bash
python docs/research/sims/redteam_capture_recapture.py   # 06: Transporttest, HT/Huggins/Hybrid unter Annahmeverletzungen (~10 s)
python docs/research/sims/redteam_audit_checks.py        # 04: Prop. 3, Optional Stopping, Kacheln, Satz 2, Mensch ≠ Gold (~1–2 min)
```
