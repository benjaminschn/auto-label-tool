# Phase 1 – Rohtabellen

Erzeugt am 19.10.2026 mit `python scripts/summarize.py --runs "runs/[mbcd]*_s[0-9]"`, `python scripts/phase1_gates.py` und `python scripts/g1_fixed_threshold.py` aus den 24 Läufen `runs/<datensatz>_{full,base}_s{0,1,2}` (nicht im Repo). Werte: Mittelwert ± Standardabweichung über 3 Seeds, Prozent wo nicht anders angegeben. Interpretation in [RESULTS.md](RESULTS.md).

### bsdata_base  (3 seeds)
| Variante | n | tol F1 | px F1 | px AP | Defekt-Recall | Defekt-Precision | Bild-Recall | Bild-Spezifität |
|---|---|---|---|---|---|---|---|---|
| Runde 0 | 3 | 92.1 ± 1.8 | 83.2 ± 2.3 | 89.2 ± 2.7 | 79.7 ± 3.8 | 83.9 ± 3.0 | 85.3 ± 3.6 | 96.1 ± 1.1 |
| Runde 0, Einzelmodell | 3 | 87.6 ± 9.0 | 78.6 ± 9.7 | 83.0 ± 10.2 | 77.5 ± 6.5 | 79.3 ± 9.0 | 84.2 ± 4.7 | 93.8 ± 3.8 |
| Letzte Runde, recall-sicher | 3 | 82.8 ± 10.9 | 71.6 ± 13.0 | – | 87.7 ± 11.0 | 40.2 ± 40.9 | 93.3 ± 10.0 | 39.9 ± 50.7 |
| Letzte Runde, Testset T | 3 | 92.6 ± 3.1 | 83.6 ± 3.8 | 89.0 ± 4.2 | 84.7 ± 1.8 | 85.1 ± 9.9 | 92.1 ± 3.1 | 96.3 ± 4.2 |
| Normalitätsmodell allein | 3 | – | – | – | – | – | – | – |

### bsdata_full  (3 seeds)
| Variante | n | tol F1 | px F1 | px AP | Defekt-Recall | Defekt-Precision | Bild-Recall | Bild-Spezifität |
|---|---|---|---|---|---|---|---|---|
| Runde 0 | 3 | 84.2 ± 8.4 | 75.0 ± 8.5 | 80.9 ± 8.0 | 75.9 ± 2.4 | 60.7 ± 6.4 | 85.0 ± 2.0 | 84.4 ± 4.6 |
| Runde 1 | 3 | 87.6 ± 2.7 | 77.8 ± 3.2 | 85.6 ± 3.1 | 79.6 ± 3.3 | 50.0 ± 3.2 | 89.2 ± 4.0 | 75.1 ± 5.3 |
| Runde 0, Einzelmodell | 3 | 73.8 ± 21.3 | 65.7 ± 20.2 | 65.5 ± 28.3 | 72.1 ± 9.7 | 51.0 ± 4.9 | 82.0 ± 7.9 | 77.6 ± 5.6 |
| Letzte Runde, recall-sicher | 3 | 43.6 ± 37.8 | 36.2 ± 31.8 | – | 90.3 ± 7.6 | 12.3 ± 12.1 | 98.1 ± 3.2 | 16.9 ± 29.3 |
| Letzte Runde, Testset T | 3 | 80.4 ± 7.0 | 71.2 ± 7.8 | 81.9 ± 7.7 | 84.6 ± 8.3 | 39.0 ± 15.3 | 91.9 ± 6.8 | 70.6 ± 11.8 |
| Normalitätsmodell allein | 3 | 8.1 ± 1.8 | 6.2 ± 1.6 | 2.8 ± 0.2 | 81.0 ± 4.3 | 18.8 ± 4.0 | 88.3 ± 5.0 | 68.4 ± 12.9 |

### cfd_base  (3 seeds)
| Variante | n | tol F1 | px F1 | px AP | Defekt-Recall | Defekt-Precision | Bild-Recall | Bild-Spezifität |
|---|---|---|---|---|---|---|---|---|
| Runde 0 | 3 | 88.6 ± 2.0 | 65.9 ± 3.4 | 72.1 ± 1.5 | 94.8 ± 3.2 | 76.0 ± 21.3 | 100.0 ± 0.0 | – |
| Runde 0, Einzelmodell | 3 | 87.8 ± 1.0 | 66.0 ± 0.6 | 68.6 ± 0.8 | 93.9 ± 2.9 | 79.8 ± 19.0 | 100.0 ± 0.0 | – |
| Letzte Runde, recall-sicher | 3 | 35.9 ± 43.6 | 23.6 ± 31.7 | – | 99.1 ± 1.6 | 43.5 ± 51.0 | 100.0 ± 0.0 | – |
| Letzte Runde, Testset T | 3 | 87.7 ± 4.3 | 63.6 ± 2.4 | 69.6 ± 2.3 | 93.4 ± 3.4 | 89.1 ± 7.6 | 100.0 ± 0.0 | – |
| Normalitätsmodell allein | 3 | – | – | – | – | – | – | – |

### cfd_full  (3 seeds)
| Variante | n | tol F1 | px F1 | px AP | Defekt-Recall | Defekt-Precision | Bild-Recall | Bild-Spezifität |
|---|---|---|---|---|---|---|---|---|
| Runde 0 | 3 | 82.0 ± 4.1 | 59.9 ± 1.9 | 58.8 ± 3.3 | 95.3 ± 5.1 | 64.5 ± 24.0 | 99.6 ± 0.7 | – |
| Runde 1 | 3 | 77.5 ± 5.6 | 56.3 ± 3.1 | 60.2 ± 3.9 | 96.0 ± 2.0 | 75.1 ± 23.1 | 99.6 ± 0.7 | – |
| Runde 0, Einzelmodell | 3 | 65.6 ± 8.5 | 49.0 ± 6.9 | 45.6 ± 6.7 | 92.8 ± 2.8 | 63.7 ± 20.4 | 99.6 ± 0.7 | – |
| Letzte Runde, recall-sicher | 3 | 58.4 ± 31.1 | 36.5 ± 22.6 | – | 99.2 ± 0.8 | 18.4 ± 26.1 | 100.0 ± 0.0 | – |
| Letzte Runde, Testset T | 3 | 81.9 ± 2.6 | 59.2 ± 3.3 | 61.4 ± 3.8 | 94.5 ± 2.3 | 71.8 ± 37.0 | 98.6 ± 2.4 | – |
| Normalitätsmodell allein | 3 | 29.5 ± 1.9 | 15.8 ± 1.0 | 9.5 ± 0.5 | 94.5 ± 1.8 | 50.5 ± 9.8 | 100.0 ± 0.0 | – |

### deepcrack_base  (3 seeds)
| Variante | n | tol F1 | px F1 | px AP | Defekt-Recall | Defekt-Precision | Bild-Recall | Bild-Spezifität |
|---|---|---|---|---|---|---|---|---|
| Runde 0 | 3 | 86.5 ± 2.8 | 77.1 ± 3.6 | 88.9 ± 2.0 | 80.7 ± 3.6 | 86.3 ± 7.4 | 99.5 ± 0.9 | – |
| Runde 0, Einzelmodell | 3 | 84.0 ± 5.0 | 75.0 ± 5.5 | 81.9 ± 6.0 | 83.7 ± 4.8 | 73.7 ± 11.3 | 99.9 ± 0.1 | – |
| Letzte Runde, recall-sicher | 3 | 45.1 ± 32.5 | 37.0 ± 28.3 | – | 98.8 ± 1.7 | 36.2 ± 55.1 | 100.0 ± 0.0 | – |
| Letzte Runde, Testset T | 3 | 82.3 ± 5.3 | 72.4 ± 5.6 | 86.7 ± 0.6 | 78.8 ± 9.3 | 90.7 ± 3.0 | 99.2 ± 0.9 | – |
| Normalitätsmodell allein | 3 | – | – | – | – | – | – | – |

### deepcrack_full  (3 seeds)
| Variante | n | tol F1 | px F1 | px AP | Defekt-Recall | Defekt-Precision | Bild-Recall | Bild-Spezifität |
|---|---|---|---|---|---|---|---|---|
| Runde 0 | 3 | 73.5 ± 7.8 | 63.4 ± 7.0 | 71.5 ± 4.3 | 84.0 ± 4.6 | 50.2 ± 17.0 | 98.2 ± 2.1 | – |
| Runde 1 | 3 | 63.9 ± 16.0 | 54.5 ± 14.9 | 77.1 ± 3.3 | 75.8 ± 9.0 | 78.1 ± 10.0 | 91.8 ± 6.8 | – |
| Runde 0, Einzelmodell | 3 | 64.6 ± 10.0 | 55.5 ± 9.0 | 61.6 ± 12.1 | 80.1 ± 3.7 | 54.8 ± 5.7 | 97.4 ± 2.0 | – |
| Letzte Runde, recall-sicher | 3 | 14.5 ± 8.1 | 10.6 ± 6.0 | – | 100.0 ± 0.1 | 55.1 ± 46.5 | 100.0 ± 0.0 | – |
| Letzte Runde, Testset T | 3 | 72.2 ± 6.5 | 61.4 ± 6.6 | 80.4 ± 1.1 | 83.2 ± 8.8 | 76.1 ± 5.6 | 98.8 ± 1.4 | – |
| Normalitätsmodell allein | 3 | 38.0 ± 1.1 | 30.1 ± 0.8 | 21.2 ± 1.9 | 78.2 ± 7.8 | 59.5 ± 10.1 | 99.3 ± 1.0 | – |

### mt_base  (3 seeds)
| Variante | n | tol F1 | px F1 | px AP | Defekt-Recall | Defekt-Precision | Bild-Recall | Bild-Spezifität |
|---|---|---|---|---|---|---|---|---|
| Runde 0 | 3 | 43.0 ± 12.2 | 41.2 ± 11.6 | 48.8 ± 12.1 | 78.1 ± 23.7 | 15.8 ± 8.3 | 86.6 ± 20.6 | 29.9 ± 32.6 |
| Runde 0, Einzelmodell | 3 | 46.5 ± 9.1 | 44.5 ± 9.0 | 37.2 ± 8.1 | 75.1 ± 13.3 | 14.5 ± 6.2 | 89.6 ± 8.0 | 25.8 ± 20.3 |
| Letzte Runde, recall-sicher | 3 | 21.4 ± 11.4 | 20.6 ± 11.0 | – | 98.3 ± 1.2 | 5.0 ± 3.7 | 100.0 ± 0.0 | 1.6 ± 2.5 |
| Letzte Runde, Testset T | 3 | 44.5 ± 9.5 | 42.8 ± 9.0 | 53.2 ± 18.0 | 79.5 ± 23.7 | 17.3 ± 9.7 | 86.9 ± 20.6 | 26.7 ± 31.5 |
| Normalitätsmodell allein | 3 | – | – | – | – | – | – | – |

### mt_full  (3 seeds)
| Variante | n | tol F1 | px F1 | px AP | Defekt-Recall | Defekt-Precision | Bild-Recall | Bild-Spezifität |
|---|---|---|---|---|---|---|---|---|
| Runde 0 | 3 | 43.3 ± 11.8 | 41.6 ± 11.5 | 39.7 ± 9.6 | 77.7 ± 22.3 | 15.0 ± 6.4 | 87.1 ± 19.2 | 28.4 ± 27.7 |
| Runde 1 | 3 | 37.0 ± 3.6 | 35.2 ± 3.0 | 36.6 ± 9.7 | 83.6 ± 16.9 | 13.8 ± 4.9 | 91.9 ± 11.1 | 21.4 ± 20.6 |
| Runde 0, Einzelmodell | 3 | 43.1 ± 5.4 | 41.2 ± 5.3 | 30.4 ± 8.9 | 77.5 ± 12.8 | 14.3 ± 5.0 | 91.5 ± 10.3 | 25.2 ± 21.5 |
| Letzte Runde, recall-sicher | 3 | 27.1 ± 19.9 | 25.8 ± 18.9 | – | 97.0 ± 3.7 | 15.4 ± 10.7 | 100.0 ± 0.0 | 3.1 ± 3.5 |
| Letzte Runde, Testset T | 3 | 40.1 ± 2.1 | 38.2 ± 2.6 | 41.5 ± 12.5 | 86.4 ± 14.1 | 15.8 ± 6.9 | 94.8 ± 7.0 | 21.3 ± 20.0 |
| Normalitätsmodell allein | 3 | 15.4 ± 2.4 | 14.3 ± 2.1 | 8.9 ± 0.4 | 66.1 ± 19.8 | 15.0 ± 3.5 | 78.7 ± 18.1 | 30.9 ± 24.1 |

### Recall-Garantie

| Datensatz | Ziel-Recall | Defekt-Recall auf U (konforme Schwelle) | Defekt-Recall (PAC-Schwelle) | Schwelle konform | gelabelte Defekte |
|---|---|---|---|---|---|
| bsdata_base | 95% | 100.0 ± 0.0 | 100.0 ± 0.0 | 0.00 ± 0.00 | 13 ± 2 |
| bsdata_base | 90% | 88.1 ± 12.5 | 100.0 ± 0.0 | 0.40 ± 0.43 | 13 ± 2 |
| bsdata_base | 80% | 80.8 ± 8.0 | 88.1 ± 12.5 | 0.82 ± 0.16 | 13 ± 2 |
| bsdata_full | 95% | 100.0 ± 0.0 | 100.0 ± 0.0 | 0.00 ± 0.00 | 13 ± 2 |
| bsdata_full | 90% | 91.9 ± 7.0 | 100.0 ± 0.0 | 0.24 ± 0.20 | 13 ± 2 |
| bsdata_full | 80% | 82.9 ± 6.8 | 91.9 ± 7.0 | 0.68 ± 0.24 | 13 ± 2 |
| cfd_base | 95% | 100.0 ± 0.0 | 100.0 ± 0.0 | 0.00 ± 0.00 | 16 ± 2 |
| cfd_base | 90% | 99.4 ± 1.0 | 100.0 ± 0.0 | 0.05 ± 0.03 | 16 ± 2 |
| cfd_base | 80% | 83.6 ± 5.8 | 96.4 ± 6.3 | 0.97 ± 0.03 | 16 ± 2 |
| cfd_full | 95% | 100.0 ± 0.0 | 100.0 ± 0.0 | 0.00 ± 0.00 | 16 ± 2 |
| cfd_full | 90% | 98.9 ± 1.3 | 100.0 ± 0.0 | 0.20 ± 0.25 | 16 ± 2 |
| cfd_full | 80% | 78.1 ± 7.9 | 98.9 ± 1.3 | 0.98 ± 0.01 | 16 ± 2 |
| deepcrack_base | 95% | 99.2 ± 1.3 | 100.0 ± 0.0 | 0.04 ± 0.02 | 27 ± 2 |
| deepcrack_base | 90% | 98.8 ± 1.7 | 99.2 ± 1.3 | 0.06 ± 0.04 | 27 ± 2 |
| deepcrack_base | 80% | 84.1 ± 12.6 | 97.1 ± 2.2 | 0.59 ± 0.30 | 27 ± 2 |
| deepcrack_full | 95% | 100.0 ± 0.1 | 100.0 ± 0.0 | 0.03 ± 0.01 | 27 ± 2 |
| deepcrack_full | 90% | 100.0 ± 0.1 | 100.0 ± 0.1 | 0.03 ± 0.02 | 27 ± 2 |
| deepcrack_full | 80% | 95.2 ± 5.3 | 98.6 ± 1.5 | 0.25 ± 0.33 | 27 ± 2 |
| mt_base | 95% | 100.0 ± 0.0 | 100.0 ± 0.0 | 0.00 ± 0.00 | 13 ± 4 |
| mt_base | 90% | 97.7 ± 2.3 | 100.0 ± 0.0 | 0.11 ± 0.09 | 13 ± 4 |
| mt_base | 80% | 92.2 ± 5.3 | 97.8 ± 2.6 | 0.27 ± 0.16 | 13 ± 4 |
| mt_full | 95% | 100.0 ± 0.0 | 100.0 ± 0.0 | 0.00 ± 0.00 | 13 ± 4 |
| mt_full | 90% | 96.6 ± 2.9 | 100.0 ± 0.0 | 0.31 ± 0.26 | 13 ± 4 |
| mt_full | 80% | 86.3 ± 16.9 | 98.3 ± 2.9 | 0.49 ± 0.42 | 13 ± 4 |

### Selbsteinschätzung

| Datensatz | Größe | wahr | Selbsteinschätzung (ohne GT) |
|---|---|---|---|
| bsdata_base | Pixel-Precision | 84.8 ± 2.5 | 73.6 ± 2.9 |
| bsdata_base | Pixel-Recall | 81.7 ± 3.9 | 84.2 ± 3.0 |
| bsdata_base | Defekt-Recall | 79.7 ± 3.8 | 90.6 ± 8.3 |
| bsdata_base | Defekt-Precision | 83.9 ± 3.0 | 83.7 ± 7.9 |
| bsdata_base | mittlerer Dice (tol) | 89.4 ± 0.5 | 24.7 ± 1.0 |
| bsdata_full | Pixel-Precision | 70.1 ± 2.8 | 59.2 ± 13.6 |
| bsdata_full | Pixel-Recall | 87.4 ± 4.3 | 82.0 ± 3.9 |
| bsdata_full | Defekt-Recall | 79.6 ± 3.3 | 78.2 ± 15.0 |
| bsdata_full | Defekt-Precision | 50.0 ± 3.2 | 52.7 ± 3.1 |
| bsdata_full | mittlerer Dice (tol) | 75.3 ± 3.6 | 26.4 ± 4.9 |
| cfd_base | Pixel-Precision | 61.1 ± 12.5 | 53.7 ± 10.6 |
| cfd_base | Pixel-Recall | 75.5 ± 13.5 | 73.4 ± 8.1 |
| cfd_base | Defekt-Recall | 94.8 ± 3.2 | 79.5 ± 22.1 |
| cfd_base | Defekt-Precision | 76.0 ± 21.3 | 80.3 ± 18.4 |
| cfd_base | mittlerer Dice (tol) | 88.5 ± 2.1 | 59.3 ± 5.7 |
| cfd_full | Pixel-Precision | 54.9 ± 9.9 | 40.7 ± 6.1 |
| cfd_full | Pixel-Recall | 61.8 ± 16.3 | 52.6 ± 16.4 |
| cfd_full | Defekt-Recall | 96.0 ± 2.0 | 64.1 ± 22.6 |
| cfd_full | Defekt-Precision | 75.1 ± 23.1 | 62.3 ± 24.7 |
| cfd_full | mittlerer Dice (tol) | 75.1 ± 5.1 | 40.2 ± 8.7 |
| deepcrack_base | Pixel-Precision | 91.7 ± 2.2 | 76.1 ± 4.5 |
| deepcrack_base | Pixel-Recall | 66.8 ± 6.3 | 68.6 ± 11.8 |
| deepcrack_base | Defekt-Recall | 80.7 ± 3.6 | 44.9 ± 11.2 |
| deepcrack_base | Defekt-Precision | 86.3 ± 7.4 | 69.2 ± 1.7 |
| deepcrack_base | mittlerer Dice (tol) | 86.7 ± 4.6 | 66.2 ± 9.3 |
| deepcrack_full | Pixel-Precision | 79.1 ± 3.9 | 49.2 ± 4.0 |
| deepcrack_full | Pixel-Recall | 43.8 ± 18.1 | 31.7 ± 21.2 |
| deepcrack_full | Defekt-Recall | 75.8 ± 9.0 | 42.9 ± 7.5 |
| deepcrack_full | Defekt-Precision | 78.1 ± 10.0 | 52.7 ± 12.5 |
| deepcrack_full | mittlerer Dice (tol) | 68.8 ± 14.8 | 30.8 ± 15.3 |
| mt_base | Pixel-Precision | 48.7 ± 29.1 | 36.2 ± 22.1 |
| mt_base | Pixel-Recall | 53.9 ± 27.3 | 18.4 ± 10.2 |
| mt_base | Defekt-Recall | 78.1 ± 23.7 | 45.4 ± 44.6 |
| mt_base | Defekt-Precision | 15.8 ± 8.3 | 42.5 ± 25.6 |
| mt_base | mittlerer Dice (tol) | 33.0 ± 22.1 | 11.1 ± 5.7 |
| mt_full | Pixel-Precision | 43.6 ± 20.8 | 37.2 ± 18.3 |
| mt_full | Pixel-Recall | 46.3 ± 32.2 | 15.8 ± 9.1 |
| mt_full | Defekt-Recall | 83.6 ± 16.9 | 49.7 ± 43.7 |
| mt_full | Defekt-Precision | 13.8 ± 4.9 | 32.1 ± 12.8 |
| mt_full | mittlerer Dice (tol) | 27.4 ± 15.6 | 12.5 ± 5.2 |

### Fehler-Signale (AUROC)

| Datensatz | Ziel | risk | p_missed_defect | anomaly_out | inv_member_dice | inv_exp_dice |
|---|---|---|---|---|---|---|
| bsdata_base | missed_defect | 0.73 ± 0.02 | 0.60 ± 0.14 | – | 0.68 ± 0.03 | 0.56 ± 0.04 |
| bsdata_base | bad_label | 0.71 ± 0.05 | 0.56 ± 0.09 | – | 0.72 ± 0.04 | 0.49 ± 0.04 |
| bsdata_full | missed_defect | 0.68 ± 0.02 | 0.63 ± 0.03 | 0.61 ± 0.02 | 0.59 ± 0.05 | 0.56 ± 0.04 |
| bsdata_full | bad_label | 0.65 ± 0.09 | 0.69 ± 0.04 | 0.57 ± 0.04 | 0.64 ± 0.11 | 0.42 ± 0.01 |
| cfd_base | missed_defect | 0.52 ± 0.06 | 0.59 ± 0.08 | – | 0.47 ± 0.05 | 0.52 ± 0.02 |
| cfd_base | bad_label | 0.59 ± 0.12 | 0.62 ± 0.11 | – | 0.55 ± 0.12 | 0.60 ± 0.05 |
| cfd_full | missed_defect | 0.69 ± 0.04 | 0.51 ± 0.07 | 0.45 ± 0.10 | 0.76 ± 0.10 | 0.74 ± 0.10 |
| cfd_full | bad_label | 0.84 ± 0.05 | 0.55 ± 0.09 | 0.48 ± 0.07 | 0.90 ± 0.06 | 0.90 ± 0.06 |
| deepcrack_base | missed_defect | 0.73 ± 0.05 | 0.69 ± 0.03 | – | 0.70 ± 0.07 | 0.71 ± 0.06 |
| deepcrack_base | bad_label | 0.77 ± 0.06 | 0.72 ± 0.04 | – | 0.73 ± 0.08 | 0.74 ± 0.07 |
| deepcrack_full | missed_defect | 0.74 ± 0.02 | 0.61 ± 0.02 | 0.48 ± 0.01 | 0.75 ± 0.02 | 0.77 ± 0.02 |
| deepcrack_full | bad_label | 0.82 ± 0.04 | 0.59 ± 0.06 | 0.51 ± 0.05 | 0.85 ± 0.05 | 0.85 ± 0.04 |
| mt_base | missed_defect | 0.54 ± 0.06 | 0.51 ± 0.02 | – | 0.59 ± 0.04 | 0.52 ± 0.06 |
| mt_base | bad_label | 0.46 ± 0.15 | 0.47 ± 0.08 | – | 0.50 ± 0.14 | 0.39 ± 0.18 |
| mt_full | missed_defect | 0.60 ± 0.13 | 0.52 ± 0.05 | 0.49 ± 0.19 | 0.70 ± 0.06 | 0.54 ± 0.12 |
| mt_full | bad_label | 0.52 ± 0.11 | 0.54 ± 0.05 | 0.60 ± 0.06 | 0.46 ± 0.07 | 0.45 ± 0.24 |

### Audit

| Datensatz | Metrik | n Audit | Abdeckung PPI | Abdeckung klassisch | Breite PPI | Breite klassisch |
|---|---|---|---|---|---|---|
| bsdata_base | px_recall | 10 | 62.1 ± 13.3 | 54.3 ± 9.4 | 20.0 ± 3.3 | 23.6 ± 2.8 |
| bsdata_base | px_recall | 20 | 67.4 ± 10.3 | 65.7 ± 8.5 | 17.5 ± 4.7 | 22.5 ± 6.0 |
| bsdata_base | px_recall | 40 | 74.1 ± 7.3 | 71.6 ± 4.2 | 14.3 ± 4.4 | 18.2 ± 6.0 |
| bsdata_base | px_recall | 80 | 80.8 ± 2.3 | 80.5 ± 2.4 | 11.6 ± 3.9 | 14.6 ± 5.3 |
| bsdata_base | px_precision | 10 | 76.1 ± 8.0 | 59.5 ± 2.9 | 26.8 ± 4.0 | 16.8 ± 2.5 |
| bsdata_base | px_precision | 20 | 82.1 ± 5.3 | 73.7 ± 5.6 | 22.9 ± 3.8 | 16.1 ± 1.7 |
| bsdata_base | px_precision | 40 | 85.8 ± 4.8 | 75.9 ± 10.1 | 18.4 ± 3.3 | 12.2 ± 2.4 |
| bsdata_base | px_precision | 80 | 89.0 ± 1.6 | 78.1 ± 10.0 | 14.0 ± 2.9 | 8.9 ± 2.4 |
| bsdata_base | defect_recall | 10 | 64.4 ± 18.5 | 51.6 ± 2.6 | 36.5 ± 0.7 | 36.4 ± 3.3 |
| bsdata_base | defect_recall | 20 | 81.2 ± 2.9 | 76.9 ± 1.5 | 33.0 ± 3.6 | 36.8 ± 4.5 |
| bsdata_base | defect_recall | 40 | 82.7 ± 3.9 | 82.1 ± 1.3 | 26.7 ± 3.7 | 29.9 ± 4.1 |
| bsdata_base | defect_recall | 80 | 85.1 ± 2.6 | 86.1 ± 1.5 | 19.9 ± 2.8 | 21.9 ± 2.6 |
| bsdata_base | defect_precision | 10 | 56.1 ± 20.4 | 40.5 ± 9.3 | 28.6 ± 2.5 | 26.1 ± 4.8 |
| bsdata_base | defect_precision | 20 | 76.8 ± 11.0 | 64.0 ± 9.6 | 28.8 ± 3.1 | 29.2 ± 4.1 |
| bsdata_base | defect_precision | 40 | 81.5 ± 6.3 | 76.4 ± 2.7 | 23.2 ± 2.0 | 25.3 ± 2.3 |
| bsdata_base | defect_precision | 80 | 85.9 ± 5.0 | 84.3 ± 3.3 | 17.9 ± 2.1 | 20.5 ± 1.7 |
| bsdata_base | mean_dice_tol | 10 | 89.9 ± 0.4 | 64.9 ± 1.1 | 35.7 ± 0.4 | 23.2 ± 0.8 |
| bsdata_base | mean_dice_tol | 20 | 89.9 ± 1.6 | 85.1 ± 2.2 | 26.4 ± 0.9 | 19.4 ± 0.8 |
| bsdata_base | mean_dice_tol | 40 | 90.6 ± 1.4 | 87.0 ± 0.9 | 20.3 ± 0.2 | 14.3 ± 0.2 |
| bsdata_base | mean_dice_tol | 80 | 90.9 ± 0.7 | 88.2 ± 0.9 | 15.1 ± 0.2 | 9.8 ± 0.2 |
| bsdata_full | px_recall | 10 | 74.2 ± 16.2 | 57.6 ± 11.4 | 19.8 ± 4.3 | 19.3 ± 3.5 |
| bsdata_full | px_recall | 20 | 75.4 ± 14.3 | 70.3 ± 10.4 | 16.2 ± 5.1 | 17.0 ± 6.5 |
| bsdata_full | px_recall | 40 | 80.9 ± 12.7 | 75.9 ± 7.4 | 13.0 ± 4.5 | 13.5 ± 6.6 |
| bsdata_full | px_recall | 80 | 84.4 ± 4.5 | 83.1 ± 2.3 | 10.2 ± 3.9 | 10.7 ± 6.1 |
| bsdata_full | px_precision | 10 | 78.3 ± 5.0 | 66.8 ± 4.6 | 38.7 ± 10.7 | 32.6 ± 3.9 |
| bsdata_full | px_precision | 20 | 83.2 ± 1.2 | 75.6 ± 6.1 | 31.4 ± 8.1 | 28.2 ± 2.1 |
| bsdata_full | px_precision | 40 | 83.7 ± 3.2 | 78.4 ± 7.2 | 23.9 ± 6.6 | 21.0 ± 2.2 |
| bsdata_full | px_precision | 80 | 86.4 ± 3.5 | 83.1 ± 5.9 | 17.2 ± 5.2 | 14.8 ± 1.9 |
| bsdata_full | defect_recall | 10 | 69.0 ± 15.1 | 52.4 ± 2.1 | 40.3 ± 3.5 | 36.9 ± 3.2 |
| bsdata_full | defect_recall | 20 | 81.2 ± 3.2 | 76.9 ± 2.7 | 34.3 ± 2.0 | 37.0 ± 3.7 |
| bsdata_full | defect_recall | 40 | 84.5 ± 1.7 | 82.4 ± 2.8 | 27.5 ± 1.7 | 30.2 ± 2.9 |
| bsdata_full | defect_recall | 80 | 86.2 ± 1.7 | 85.7 ± 1.6 | 20.3 ± 0.9 | 22.1 ± 1.6 |
| bsdata_full | defect_precision | 10 | 86.1 ± 5.5 | 80.0 ± 3.0 | 35.1 ± 4.3 | 57.8 ± 1.4 |
| bsdata_full | defect_precision | 20 | 88.1 ± 2.1 | 84.7 ± 2.0 | 28.8 ± 1.2 | 46.7 ± 2.8 |
| bsdata_full | defect_precision | 40 | 87.3 ± 2.3 | 84.9 ± 1.0 | 21.1 ± 0.2 | 33.2 ± 2.7 |
| bsdata_full | defect_precision | 80 | 88.2 ± 0.7 | 87.3 ± 0.3 | 15.0 ± 0.4 | 23.0 ± 2.2 |
| bsdata_full | mean_dice_tol | 10 | 89.7 ± 1.4 | 90.0 ± 3.2 | 50.1 ± 5.0 | 43.0 ± 3.2 |
| bsdata_full | mean_dice_tol | 20 | 91.0 ± 1.7 | 90.5 ± 1.9 | 37.4 ± 4.0 | 30.7 ± 1.8 |
| bsdata_full | mean_dice_tol | 40 | 91.3 ± 0.7 | 88.9 ± 1.2 | 26.7 ± 2.4 | 21.0 ± 0.7 |
| bsdata_full | mean_dice_tol | 80 | 89.3 ± 1.7 | 89.4 ± 1.0 | 18.3 ± 1.6 | 14.3 ± 0.5 |
| cfd_base | px_recall | 10 | 78.6 ± 3.6 | 80.3 ± 4.4 | 13.4 ± 2.2 | 16.2 ± 3.8 |
| cfd_base | px_recall | 20 | 86.3 ± 2.2 | 85.7 ± 2.4 | 9.3 ± 1.6 | 11.0 ± 2.6 |
| cfd_base | px_recall | 40 | 87.9 ± 1.6 | 87.2 ± 2.2 | 5.6 ± 0.9 | 6.6 ± 1.5 |
| cfd_base | px_recall | 80 | 91.5 ± 1.1 | 91.3 ± 1.5 | 1.2 ± 0.2 | 1.4 ± 0.3 |
| cfd_base | px_precision | 10 | 89.6 ± 1.0 | 88.2 ± 0.9 | 13.3 ± 1.3 | 11.7 ± 2.1 |
| cfd_base | px_precision | 20 | 89.7 ± 0.7 | 88.1 ± 1.1 | 8.5 ± 0.8 | 7.7 ± 1.6 |
| cfd_base | px_precision | 40 | 90.1 ± 0.8 | 88.2 ± 1.2 | 4.9 ± 0.5 | 4.5 ± 1.0 |
| cfd_base | px_precision | 80 | 88.7 ± 0.8 | 88.1 ± 0.8 | 1.0 ± 0.1 | 1.0 ± 0.2 |
| cfd_base | defect_recall | 10 | 68.1 ± 29.0 | 50.7 ± 21.0 | 21.6 ± 14.5 | 11.9 ± 5.9 |
| cfd_base | defect_recall | 20 | 76.3 ± 19.2 | 66.5 ± 14.2 | 14.1 ± 8.0 | 9.3 ± 4.3 |
| cfd_base | defect_recall | 40 | 88.0 ± 1.2 | 84.1 ± 4.5 | 9.3 ± 4.5 | 6.8 ± 2.1 |
| cfd_base | defect_recall | 80 | 88.8 ± 0.6 | 90.8 ± 3.7 | 2.5 ± 1.7 | 1.5 ± 0.4 |
| cfd_base | defect_precision | 10 | 82.2 ± 0.5 | 69.9 ± 12.1 | 29.2 ± 9.2 | 27.9 ± 10.5 |
| cfd_base | defect_precision | 20 | 81.5 ± 3.4 | 75.7 ± 7.5 | 20.5 ± 5.5 | 20.4 ± 6.1 |
| cfd_base | defect_precision | 40 | 85.3 ± 3.8 | 84.4 ± 4.9 | 13.2 ± 2.6 | 13.4 ± 3.0 |
| cfd_base | defect_precision | 80 | 87.8 ± 2.0 | 88.0 ± 1.9 | 2.9 ± 0.6 | 3.0 ± 0.7 |
| cfd_base | mean_dice_tol | 10 | 86.0 ± 2.9 | 81.6 ± 4.2 | 9.0 ± 1.2 | 12.3 ± 2.3 |
| cfd_base | mean_dice_tol | 20 | 86.9 ± 3.1 | 86.0 ± 5.3 | 5.7 ± 0.8 | 8.0 ± 1.7 |
| cfd_base | mean_dice_tol | 40 | 86.7 ± 5.0 | 87.9 ± 2.9 | 3.4 ± 0.5 | 4.7 ± 1.2 |
| cfd_base | mean_dice_tol | 80 | 90.8 ± 3.5 | 91.5 ± 0.2 | 0.7 ± 0.1 | 1.0 ± 0.3 |
| cfd_full | px_recall | 10 | 83.9 ± 2.0 | 84.8 ± 3.3 | 18.9 ± 4.1 | 28.4 ± 3.6 |
| cfd_full | px_recall | 20 | 87.9 ± 0.6 | 87.1 ± 1.6 | 12.3 ± 2.5 | 18.3 ± 2.3 |
| cfd_full | px_recall | 40 | 89.8 ± 1.2 | 90.1 ± 0.4 | 7.3 ± 1.6 | 10.7 ± 1.5 |
| cfd_full | px_recall | 80 | 90.7 ± 1.6 | 89.7 ± 1.6 | 1.6 ± 0.4 | 2.3 ± 0.3 |
| cfd_full | px_precision | 10 | 87.9 ± 0.9 | 85.4 ± 1.4 | 20.7 ± 1.2 | 14.7 ± 1.7 |
| cfd_full | px_precision | 20 | 88.0 ± 1.6 | 87.7 ± 1.2 | 13.1 ± 0.6 | 9.7 ± 1.3 |
| cfd_full | px_precision | 40 | 90.8 ± 1.5 | 89.9 ± 1.2 | 7.8 ± 0.6 | 5.7 ± 0.8 |
| cfd_full | px_precision | 80 | 90.1 ± 1.7 | 89.3 ± 1.1 | 1.6 ± 0.1 | 1.2 ± 0.2 |
| cfd_full | defect_recall | 10 | 79.0 ± 12.1 | 41.4 ± 16.3 | 32.3 ± 23.8 | 9.2 ± 4.7 |
| cfd_full | defect_recall | 20 | 84.0 ± 8.8 | 63.9 ± 23.0 | 23.0 ± 16.7 | 7.9 ± 4.0 |
| cfd_full | defect_recall | 40 | 87.7 ± 5.4 | 79.5 ± 4.0 | 13.2 ± 7.0 | 5.8 ± 2.2 |
| cfd_full | defect_recall | 80 | 93.1 ± 2.5 | 95.2 ± 4.1 | 3.4 ± 1.9 | 1.3 ± 0.4 |
| cfd_full | defect_precision | 10 | 79.3 ± 8.9 | 76.8 ± 5.4 | 25.9 ± 9.6 | 26.4 ± 10.1 |
| cfd_full | defect_precision | 20 | 82.5 ± 2.9 | 83.0 ± 2.8 | 18.6 ± 6.6 | 18.9 ± 7.2 |
| cfd_full | defect_precision | 40 | 85.9 ± 3.7 | 84.9 ± 2.8 | 11.2 ± 3.9 | 11.5 ± 4.3 |
| cfd_full | defect_precision | 80 | 89.5 ± 2.7 | 89.7 ± 2.5 | 2.4 ± 0.9 | 2.5 ± 1.0 |
| cfd_full | mean_dice_tol | 10 | 88.8 ± 2.9 | 87.9 ± 1.7 | 14.8 ± 1.4 | 23.3 ± 5.3 |
| cfd_full | mean_dice_tol | 20 | 88.8 ± 1.8 | 87.6 ± 1.6 | 9.2 ± 0.9 | 14.5 ± 3.3 |
| cfd_full | mean_dice_tol | 40 | 90.9 ± 4.0 | 90.1 ± 1.8 | 5.3 ± 0.4 | 8.4 ± 2.0 |
| cfd_full | mean_dice_tol | 80 | 89.7 ± 1.9 | 91.3 ± 1.3 | 1.1 ± 0.1 | 1.8 ± 0.4 |
| deepcrack_base | px_recall | 10 | 79.5 ± 4.2 | 76.9 ± 4.4 | 25.6 ± 2.5 | 28.4 ± 4.1 |
| deepcrack_base | px_recall | 20 | 83.3 ± 4.5 | 80.4 ± 6.3 | 19.1 ± 2.6 | 21.0 ± 2.5 |
| deepcrack_base | px_recall | 40 | 82.7 ± 4.3 | 83.9 ± 3.7 | 14.0 ± 2.2 | 15.5 ± 1.4 |
| deepcrack_base | px_recall | 80 | 87.0 ± 0.4 | 88.0 ± 0.7 | 9.7 ± 1.5 | 10.7 ± 0.7 |
| deepcrack_base | px_precision | 10 | 86.3 ± 2.3 | 81.5 ± 1.6 | 15.7 ± 2.4 | 11.5 ± 2.8 |
| deepcrack_base | px_precision | 20 | 89.0 ± 2.8 | 83.7 ± 4.5 | 12.5 ± 2.1 | 8.4 ± 2.1 |
| deepcrack_base | px_precision | 40 | 89.5 ± 0.8 | 87.5 ± 1.8 | 9.4 ± 1.4 | 6.0 ± 1.9 |
| deepcrack_base | px_precision | 80 | 88.2 ± 0.3 | 87.6 ± 2.3 | 6.5 ± 0.9 | 4.0 ± 1.3 |
| deepcrack_base | defect_recall | 10 | 73.4 ± 11.0 | 74.5 ± 1.3 | 56.4 ± 14.3 | 27.8 ± 3.2 |
| deepcrack_base | defect_recall | 20 | 77.4 ± 7.8 | 77.3 ± 1.2 | 50.3 ± 14.1 | 21.6 ± 2.7 |
| deepcrack_base | defect_recall | 40 | 77.6 ± 6.8 | 84.3 ± 0.8 | 44.2 ± 11.9 | 16.4 ± 1.6 |
| deepcrack_base | defect_recall | 80 | 78.1 ± 10.8 | 86.3 ± 1.7 | 33.0 ± 10.0 | 11.3 ± 0.8 |
| deepcrack_base | defect_precision | 10 | 85.5 ± 2.4 | 72.2 ± 2.4 | 23.6 ± 6.4 | 22.5 ± 10.1 |
| deepcrack_base | defect_precision | 20 | 84.9 ± 3.3 | 78.8 ± 1.8 | 18.2 ± 4.9 | 18.0 ± 7.3 |
| deepcrack_base | defect_precision | 40 | 87.9 ± 1.4 | 81.7 ± 1.9 | 13.5 ± 3.5 | 13.2 ± 5.6 |
| deepcrack_base | defect_precision | 80 | 88.4 ± 1.3 | 87.1 ± 1.5 | 9.6 ± 2.2 | 9.3 ± 4.0 |
| deepcrack_base | mean_dice_tol | 10 | 89.4 ± 0.3 | 83.1 ± 1.2 | 13.5 ± 1.3 | 18.5 ± 6.5 |
| deepcrack_base | mean_dice_tol | 20 | 89.1 ± 0.9 | 87.1 ± 1.9 | 9.2 ± 0.8 | 12.6 ± 4.3 |
| deepcrack_base | mean_dice_tol | 40 | 89.5 ± 1.1 | 87.3 ± 2.1 | 6.2 ± 0.5 | 8.6 ± 3.0 |
| deepcrack_base | mean_dice_tol | 80 | 88.9 ± 1.3 | 89.9 ± 2.6 | 4.1 ± 0.3 | 5.7 ± 1.8 |
| deepcrack_full | px_recall | 10 | 80.5 ± 6.5 | 81.3 ± 4.0 | 32.9 ± 5.2 | 41.2 ± 4.0 |
| deepcrack_full | px_recall | 20 | 85.5 ± 3.9 | 86.3 ± 3.3 | 23.4 ± 4.5 | 29.3 ± 3.6 |
| deepcrack_full | px_recall | 40 | 85.7 ± 2.5 | 87.1 ± 0.9 | 16.6 ± 3.6 | 20.7 ± 2.7 |
| deepcrack_full | px_recall | 80 | 88.3 ± 0.8 | 89.1 ± 0.1 | 11.1 ± 2.6 | 13.8 ± 1.9 |
| deepcrack_full | px_precision | 10 | 87.1 ± 0.4 | 83.2 ± 4.9 | 30.2 ± 3.8 | 17.7 ± 2.4 |
| deepcrack_full | px_precision | 20 | 86.9 ± 1.2 | 85.3 ± 2.7 | 22.5 ± 3.0 | 12.9 ± 1.5 |
| deepcrack_full | px_precision | 40 | 89.8 ± 2.1 | 88.7 ± 1.6 | 16.2 ± 3.0 | 9.2 ± 1.2 |
| deepcrack_full | px_precision | 80 | 88.1 ± 2.0 | 87.1 ± 3.7 | 11.0 ± 2.5 | 6.2 ± 0.7 |
| deepcrack_full | defect_recall | 10 | 87.5 ± 2.3 | 77.3 ± 4.9 | 46.2 ± 5.1 | 34.8 ± 8.9 |
| deepcrack_full | defect_recall | 20 | 89.9 ± 1.4 | 81.3 ± 3.8 | 34.1 ± 5.1 | 25.8 ± 5.8 |
| deepcrack_full | defect_recall | 40 | 89.1 ± 0.7 | 85.9 ± 3.4 | 25.8 ± 3.5 | 18.9 ± 3.9 |
| deepcrack_full | defect_recall | 80 | 89.8 ± 0.9 | 87.4 ± 1.1 | 17.8 ± 1.6 | 12.8 ± 2.5 |
| deepcrack_full | defect_precision | 10 | 85.8 ± 5.2 | 77.2 ± 3.3 | 29.1 ± 6.4 | 30.7 ± 8.0 |
| deepcrack_full | defect_precision | 20 | 86.1 ± 1.7 | 81.3 ± 4.8 | 21.9 ± 5.3 | 23.8 ± 5.7 |
| deepcrack_full | defect_precision | 40 | 85.7 ± 3.0 | 83.6 ± 3.0 | 15.7 ± 3.9 | 17.2 ± 3.8 |
| deepcrack_full | defect_precision | 80 | 86.1 ± 1.4 | 85.9 ± 1.4 | 10.8 ± 2.7 | 11.9 ± 2.3 |
| deepcrack_full | mean_dice_tol | 10 | 90.2 ± 1.0 | 86.9 ± 4.5 | 24.5 ± 6.5 | 35.5 ± 7.5 |
| deepcrack_full | mean_dice_tol | 20 | 90.6 ± 1.0 | 89.1 ± 1.4 | 16.3 ± 4.1 | 23.8 ± 4.6 |
| deepcrack_full | mean_dice_tol | 40 | 90.5 ± 1.0 | 89.9 ± 1.1 | 11.0 ± 2.8 | 16.3 ± 2.9 |
| deepcrack_full | mean_dice_tol | 80 | 88.7 ± 1.3 | 88.5 ± 2.3 | 7.2 ± 1.8 | 10.7 ± 1.9 |
| mt_base | px_recall | 10 | 80.9 ± 6.6 | 23.8 ± 4.9 | 69.1 ± 8.3 | 17.4 ± 7.2 |
| mt_base | px_recall | 20 | 84.7 ± 7.0 | 41.7 ± 2.8 | 59.5 ± 9.2 | 27.5 ± 8.9 |
| mt_base | px_recall | 40 | 86.0 ± 3.9 | 58.5 ± 6.6 | 50.3 ± 8.5 | 34.4 ± 11.9 |
| mt_base | px_recall | 80 | 85.7 ± 4.0 | 71.7 ± 4.9 | 37.4 ± 6.6 | 33.2 ± 10.6 |
| mt_base | px_precision | 10 | 61.1 ± 22.7 | 42.1 ± 11.7 | 44.7 ± 9.6 | 32.6 ± 3.9 |
| mt_base | px_precision | 20 | 69.7 ± 11.5 | 62.9 ± 7.7 | 44.8 ± 18.1 | 39.1 ± 9.8 |
| mt_base | px_precision | 40 | 78.3 ± 8.2 | 74.1 ± 4.0 | 40.1 ± 20.2 | 37.1 ± 12.3 |
| mt_base | px_precision | 80 | 84.3 ± 7.8 | 81.7 ± 4.4 | 34.8 ± 20.1 | 30.0 ± 10.7 |
| mt_base | defect_recall | 10 | 87.4 ± 12.6 | 34.3 ± 27.9 | 65.5 ± 29.6 | 26.8 ± 22.6 |
| mt_base | defect_recall | 20 | 88.8 ± 6.6 | 50.7 ± 34.4 | 59.9 ± 35.2 | 31.3 ± 25.8 |
| mt_base | defect_recall | 40 | 84.3 ± 6.3 | 63.7 ± 33.0 | 51.7 ± 34.6 | 26.2 ± 19.7 |
| mt_base | defect_recall | 80 | 90.3 ± 2.6 | 71.4 ± 26.0 | 43.1 ± 34.9 | 19.7 ± 12.8 |
| mt_base | defect_precision | 10 | 88.8 ± 2.1 | 80.8 ± 9.5 | 32.0 ± 17.2 | 30.3 ± 15.2 |
| mt_base | defect_precision | 20 | 88.8 ± 1.6 | 85.3 ± 2.4 | 24.4 ± 14.5 | 24.5 ± 15.5 |
| mt_base | defect_precision | 40 | 88.5 ± 0.2 | 87.7 ± 2.5 | 18.0 ± 11.5 | 17.6 ± 11.7 |
| mt_base | defect_precision | 80 | 91.0 ± 1.4 | 89.3 ± 1.0 | 12.8 ± 8.4 | 12.3 ± 8.5 |
| mt_base | mean_dice_tol | 10 | 85.3 ± 7.0 | 86.9 ± 9.1 | 43.3 ± 16.6 | 41.8 ± 13.6 |
| mt_base | mean_dice_tol | 20 | 90.1 ± 3.2 | 89.5 ± 2.5 | 30.7 ± 10.1 | 29.3 ± 7.4 |
| mt_base | mean_dice_tol | 40 | 89.5 ± 3.5 | 89.5 ± 2.7 | 21.2 ± 6.6 | 20.2 ± 4.7 |
| mt_base | mean_dice_tol | 80 | 89.3 ± 3.3 | 89.4 ± 3.1 | 14.5 ± 4.5 | 13.9 ± 3.3 |
| mt_full | px_recall | 10 | 76.4 ± 9.6 | 20.2 ± 4.6 | 65.0 ± 13.4 | 16.6 ± 7.8 |
| mt_full | px_recall | 20 | 79.0 ± 9.4 | 38.1 ± 3.6 | 57.8 ± 12.6 | 26.6 ± 8.2 |
| mt_full | px_recall | 40 | 82.5 ± 6.5 | 55.3 ± 2.3 | 51.0 ± 10.3 | 33.9 ± 10.1 |
| mt_full | px_recall | 80 | 85.9 ± 3.9 | 72.1 ± 2.2 | 38.1 ± 7.9 | 33.6 ± 8.6 |
| mt_full | px_precision | 10 | 62.2 ± 23.0 | 38.8 ± 15.2 | 44.1 ± 5.0 | 31.5 ± 1.9 |
| mt_full | px_precision | 20 | 64.1 ± 15.9 | 54.6 ± 15.1 | 44.2 ± 12.3 | 37.1 ± 5.8 |
| mt_full | px_precision | 40 | 71.7 ± 12.3 | 66.1 ± 11.2 | 40.4 ± 17.2 | 36.5 ± 10.0 |
| mt_full | px_precision | 80 | 81.5 ± 7.2 | 78.1 ± 6.8 | 36.2 ± 20.5 | 32.6 ± 11.3 |
| mt_full | defect_recall | 10 | 57.8 ± 48.0 | 30.3 ± 28.6 | 57.2 ± 47.4 | 24.4 ± 23.7 |
| mt_full | defect_recall | 20 | 60.3 ± 47.3 | 46.3 ± 37.1 | 54.2 ± 44.2 | 28.1 ± 25.4 |
| mt_full | defect_recall | 40 | 62.6 ± 41.5 | 59.1 ± 38.8 | 48.2 ± 38.4 | 24.2 ± 20.2 |
| mt_full | defect_recall | 80 | 69.3 ± 35.3 | 66.0 ± 32.5 | 36.9 ± 28.6 | 18.2 ± 13.2 |
| mt_full | defect_precision | 10 | 85.5 ± 5.8 | 80.8 ± 5.6 | 28.3 ± 10.3 | 28.2 ± 10.9 |
| mt_full | defect_precision | 20 | 86.6 ± 3.5 | 84.6 ± 4.3 | 21.7 ± 8.3 | 21.5 ± 8.7 |
| mt_full | defect_precision | 40 | 87.3 ± 3.9 | 86.1 ± 2.9 | 15.8 ± 6.3 | 15.3 ± 6.2 |
| mt_full | defect_precision | 80 | 90.8 ± 2.1 | 90.1 ± 2.4 | 11.1 ± 4.6 | 10.6 ± 4.4 |
| mt_full | mean_dice_tol | 10 | 85.2 ± 5.8 | 85.4 ± 7.5 | 41.3 ± 17.8 | 40.0 ± 14.9 |
| mt_full | mean_dice_tol | 20 | 87.8 ± 1.9 | 87.3 ± 1.7 | 29.2 ± 11.1 | 28.0 ± 8.6 |
| mt_full | mean_dice_tol | 40 | 87.7 ± 1.4 | 87.7 ± 1.3 | 20.2 ± 7.3 | 19.4 ± 5.5 |
| mt_full | mean_dice_tol | 80 | 88.3 ± 1.9 | 88.1 ± 1.0 | 13.9 ± 5.0 | 13.3 ± 3.8 |

### Review

| Datensatz | Reihenfolge | 0% geprüft | 5% geprüft | 10% geprüft | 20% geprüft | 30% geprüft |
|---|---|---|---|---|---|---|
| bsdata_base | risk (Defekt-Recall) | 79.7 ± 3.8 | 83.7 ± 4.2 | 85.6 ± 5.3 | 89.3 ± 3.9 | 91.5 ± 2.9 |
| bsdata_base | random (Defekt-Recall) | 79.7 ± 3.8 | 80.7 ± 3.5 | 81.3 ± 3.7 | 84.4 ± 3.9 | 87.4 ± 2.9 |
| bsdata_base | oracle (Defekt-Recall) | 79.7 ± 3.8 | 88.3 ± 4.6 | 98.5 ± 0.9 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| bsdata_base | risk (tol F1) | 92.1 ± 1.8 | 92.5 ± 1.5 | 93.3 ± 1.2 | 94.4 ± 1.4 | 95.2 ± 2.6 |
| bsdata_base | random (tol F1) | 92.1 ± 1.8 | 92.7 ± 1.9 | 93.1 ± 1.5 | 94.0 ± 1.2 | 94.8 ± 0.7 |
| bsdata_full | risk (Defekt-Recall) | 79.6 ± 3.3 | 81.9 ± 1.9 | 85.0 ± 2.5 | 89.7 ± 1.8 | 92.0 ± 0.8 |
| bsdata_full | random (Defekt-Recall) | 79.6 ± 3.3 | 80.6 ± 3.3 | 81.0 ± 3.4 | 84.2 ± 3.1 | 87.4 ± 1.7 |
| bsdata_full | oracle (Defekt-Recall) | 79.6 ± 3.3 | 80.7 ± 2.3 | 82.0 ± 3.8 | 92.9 ± 2.7 | 100.0 ± 0.0 |
| bsdata_full | risk (tol F1) | 87.6 ± 2.7 | 87.8 ± 2.7 | 88.3 ± 2.5 | 90.6 ± 3.2 | 92.3 ± 3.6 |
| bsdata_full | random (tol F1) | 87.6 ± 2.7 | 88.3 ± 2.8 | 88.8 ± 2.5 | 90.0 ± 2.2 | 91.5 ± 1.7 |
| cfd_base | risk (Defekt-Recall) | 94.8 ± 3.2 | 95.0 ± 3.4 | 95.3 ± 3.7 | 96.2 ± 2.4 | 96.7 ± 1.5 |
| cfd_base | random (Defekt-Recall) | 94.8 ± 3.2 | 95.0 ± 2.8 | 95.0 ± 2.8 | 95.6 ± 2.5 | 96.4 ± 1.1 |
| cfd_base | oracle (Defekt-Recall) | 94.8 ± 3.2 | 97.0 ± 4.4 | 98.7 ± 2.3 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| cfd_base | risk (tol F1) | 88.6 ± 2.0 | 89.5 ± 1.9 | 90.7 ± 2.2 | 92.9 ± 2.4 | 94.2 ± 2.1 |
| cfd_base | random (tol F1) | 88.6 ± 2.0 | 88.9 ± 1.9 | 89.6 ± 1.7 | 90.9 ± 1.2 | 91.8 ± 1.7 |
| cfd_full | risk (Defekt-Recall) | 96.0 ± 2.0 | 96.0 ± 2.0 | 96.3 ± 1.8 | 98.0 ± 0.4 | 98.6 ± 0.5 |
| cfd_full | random (Defekt-Recall) | 96.0 ± 2.0 | 96.3 ± 1.9 | 96.6 ± 1.6 | 97.7 ± 1.4 | 97.7 ± 1.4 |
| cfd_full | oracle (Defekt-Recall) | 96.0 ± 2.0 | 96.9 ± 1.5 | 98.0 ± 1.9 | 99.1 ± 1.6 | 100.0 ± 0.0 |
| cfd_full | risk (tol F1) | 77.5 ± 5.6 | 79.6 ± 4.7 | 81.2 ± 4.3 | 84.5 ± 3.6 | 87.8 ± 2.8 |
| cfd_full | random (tol F1) | 77.5 ± 5.6 | 78.5 ± 5.2 | 79.7 ± 4.6 | 82.7 ± 4.6 | 84.3 ± 4.3 |
| deepcrack_base | risk (Defekt-Recall) | 80.7 ± 3.6 | 83.1 ± 3.3 | 85.4 ± 2.2 | 89.2 ± 0.4 | 91.4 ± 0.6 |
| deepcrack_base | random (Defekt-Recall) | 80.7 ± 3.6 | 81.8 ± 3.7 | 83.1 ± 3.9 | 85.0 ± 3.9 | 87.4 ± 3.7 |
| deepcrack_base | oracle (Defekt-Recall) | 80.7 ± 3.6 | 83.8 ± 2.2 | 87.1 ± 2.9 | 94.9 ± 3.9 | 99.8 ± 0.3 |
| deepcrack_base | risk (tol F1) | 86.5 ± 2.8 | 88.4 ± 1.5 | 90.1 ± 1.3 | 92.3 ± 1.2 | 94.4 ± 1.0 |
| deepcrack_base | random (tol F1) | 86.5 ± 2.8 | 87.6 ± 2.7 | 88.4 ± 2.7 | 90.0 ± 2.7 | 91.6 ± 2.3 |
| deepcrack_full | risk (Defekt-Recall) | 75.8 ± 9.0 | 78.2 ± 8.1 | 81.1 ± 7.9 | 85.4 ± 5.6 | 88.2 ± 4.6 |
| deepcrack_full | random (Defekt-Recall) | 75.8 ± 9.0 | 77.2 ± 8.4 | 78.8 ± 7.2 | 80.9 ± 6.9 | 83.6 ± 5.8 |
| deepcrack_full | oracle (Defekt-Recall) | 75.8 ± 9.0 | 78.9 ± 9.1 | 81.6 ± 8.0 | 87.4 ± 5.7 | 93.5 ± 7.8 |
| deepcrack_full | risk (tol F1) | 63.9 ± 16.0 | 70.7 ± 11.7 | 74.9 ± 10.8 | 82.8 ± 6.8 | 87.9 ± 4.8 |
| deepcrack_full | random (tol F1) | 63.9 ± 16.0 | 67.4 ± 14.1 | 70.0 ± 13.1 | 75.0 ± 10.4 | 79.6 ± 7.8 |
| mt_base | risk (Defekt-Recall) | 78.1 ± 23.7 | 79.0 ± 22.9 | 80.4 ± 21.7 | 82.9 ± 18.8 | 85.5 ± 15.4 |
| mt_base | random (Defekt-Recall) | 78.1 ± 23.7 | 79.0 ± 22.7 | 80.1 ± 21.4 | 81.9 ± 20.1 | 84.2 ± 17.9 |
| mt_base | oracle (Defekt-Recall) | 78.1 ± 23.7 | 78.2 ± 23.6 | 78.2 ± 23.6 | 78.4 ± 23.4 | 89.2 ± 9.0 |
| mt_base | risk (tol F1) | 43.0 ± 12.2 | 44.2 ± 13.1 | 47.5 ± 15.9 | 49.9 ± 17.4 | 52.0 ± 18.1 |
| mt_base | random (tol F1) | 43.0 ± 12.2 | 44.2 ± 12.4 | 46.8 ± 13.7 | 51.9 ± 15.9 | 55.9 ± 16.2 |
| mt_full | risk (Defekt-Recall) | 83.6 ± 16.9 | 85.4 ± 15.1 | 86.7 ± 13.6 | 89.2 ± 10.5 | 91.4 ± 8.6 |
| mt_full | random (Defekt-Recall) | 83.6 ± 16.9 | 83.9 ± 16.8 | 84.7 ± 15.8 | 86.4 ± 14.3 | 88.5 ± 12.1 |
| mt_full | oracle (Defekt-Recall) | 83.6 ± 16.9 | 83.7 ± 16.7 | 83.7 ± 16.7 | 83.9 ± 16.6 | 85.8 ± 17.5 |
| mt_full | risk (tol F1) | 37.0 ± 3.6 | 43.7 ± 9.0 | 49.3 ± 12.9 | 55.2 ± 16.2 | 60.4 ± 19.1 |
| mt_full | random (tol F1) | 37.0 ± 3.6 | 38.4 ± 3.9 | 41.2 ± 5.8 | 47.9 ± 8.7 | 52.7 ± 9.4 |

### Phase 1: Pool-Qualität (ζ, φ, Größenklassen)

| Variante | Runde | n | ζ (Fehlbildrate) | K | φ (Fehlkomp./Bild) | φ sauber | Bild-Recall ρ_b | SegF1 | Recall S | Recall M | Recall L |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bsdata_base | 0 | 3 | 7.6 ± 0.7 | 65 ± 8 | 0.07 ± 0.02 | 0.04 ± 0.01 | 79.6 ± 1.9 | 83.2 ± 2.3 | 39.9 ± 10.2 (n=112 ± 4) | 97.6 ± 1.6 (n=222 ± 9) | 93.3 ± 4.7 (n=35 ± 4) |
| bsdata_full | 0 | 3 | 9.2 ± 1.2 | 77 ± 7 | 0.22 ± 0.07 | 0.19 ± 0.04 | 75.5 ± 2.9 | 75.0 ± 8.5 | 35.9 ± 15.2 (n=112 ± 4) | 95.4 ± 4.1 (n=222 ± 9) | 78.3 ± 31.3 (n=35 ± 4) |
| bsdata_full | 1 | 3 | 7.7 ± 0.9 | 65 ± 11 | 0.36 ± 0.08 | 0.34 ± 0.05 | 79.5 ± 2.8 | 77.8 ± 3.2 | 35.0 ± 11.1 (n=112 ± 4) | 98.7 ± 0.8 (n=222 ± 9) | 100.0 ± 0.0 (n=35 ± 4) |
| cfd_base | 0 | 3 | 6.7 ± 4.5 | 6 ± 4 | 0.82 ± 0.60 | – | 93.3 ± 4.5 | 65.9 ± 3.4 | 56.2 ± 23.4 (n=10 ± 2) | 93.8 ± 4.6 (n=31 ± 6) | 100.0 ± 0.0 (n=78 ± 1) |
| cfd_full | 0 | 3 | 6.0 ± 6.3 | 5 ± 5 | 3.12 ± 4.21 | – | 94.0 ± 6.3 | 59.9 ± 1.9 | 72.0 ± 20.6 (n=10 ± 2) | 93.8 ± 7.3 (n=31 ± 6) | 98.7 ± 2.2 (n=78 ± 1) |
| cfd_full | 1 | 3 | 5.2 ± 2.5 | 4 ± 2 | 2.08 ± 2.45 | – | 94.8 ± 2.5 | 56.3 ± 3.1 | 68.1 ± 11.9 (n=10 ± 2) | 94.9 ± 6.5 (n=31 ± 6) | 99.6 ± 0.7 (n=78 ± 1) |
| deepcrack_base | 0 | 3 | 24.3 ± 2.7 | 99 ± 6 | 0.86 ± 0.47 | – | 75.7 ± 2.7 | 77.1 ± 3.6 | 34.8 ± 6.8 (n=192 ± 30) | 80.9 ± 6.6 (n=424 ± 17) | 99.1 ± 1.6 (n=469 ± 27) |
| deepcrack_full | 0 | 3 | 19.7 ± 3.4 | 81 ± 16 | 3.38 ± 1.14 | – | 80.3 ± 3.4 | 63.4 ± 7.0 | 59.1 ± 5.9 (n=192 ± 30) | 83.9 ± 5.6 (n=424 ± 17) | 94.1 ± 4.1 (n=469 ± 27) |
| deepcrack_full | 1 | 3 | 29.7 ± 9.1 | 122 ± 40 | 1.21 ± 0.49 | – | 70.3 ± 9.1 | 54.5 ± 14.9 | 46.8 ± 9.6 (n=192 ± 30) | 75.9 ± 10.6 (n=424 ± 17) | 87.5 ± 8.5 (n=469 ± 27) |
| mt_base | 0 | 3 | 6.7 ± 7.4 | 71 ± 78 | 1.93 ± 1.30 | 2.09 ± 1.41 | 76.1 ± 26.7 | 41.2 ± 11.6 | 56.8 ± 37.1 (n=54 ± 7) | 77.9 ± 32.4 (n=152 ± 6) | 86.0 ± 11.7 (n=136 ± 5) |
| mt_full | 0 | 3 | 6.6 ± 6.6 | 70 ± 70 | 2.48 ± 1.91 | 2.60 ± 1.97 | 76.3 ± 23.8 | 41.6 ± 11.5 | 66.7 ± 15.3 (n=54 ± 7) | 84.2 ± 21.7 (n=152 ± 6) | 75.2 ± 26.7 (n=136 ± 5) |
| mt_full | 1 | 3 | 4.8 ± 5.0 | 51 ± 53 | 2.42 ± 1.39 | 2.59 ± 1.47 | 82.6 ± 18.1 | 35.2 ± 3.0 | 76.7 ± 11.7 (n=54 ± 7) | 88.5 ± 16.2 (n=152 ± 6) | 80.9 ± 20.2 (n=136 ± 5) |

### Phase 1: K′(k) – Restfehlbilder nach Review der k riskantesten Bilder

| Variante | Runde | Reihenfolge | K′ nach 0% | K′ nach 5% | K′ nach 10% | K′ nach 20% | K′ nach 30% | K′ nach 50% | AUC K′/K |
|---|---|---|---|---|---|---|---|---|---|
| bsdata_base | 0 | Risiko | 64.7 ± 7.6 | 53.7 ± 10.0 | 43.7 ± 15.9 | 27.7 ± 3.5 | 22.7 ± 6.1 | 3.3 ± 2.3 | 0.234 ± 0.036 |
| bsdata_base | 0 | Zufall | 64.7 ± 7.6 | 61.7 ± 7.3 | 58.4 ± 7.1 | 51.9 ± 6.5 | 45.1 ± 5.4 | 32.1 ± 4.0 | 0.497 ± 0.005 |
| bsdata_base | 0 | Orakel | 64.7 ± 7.6 | 22.7 ± 6.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.038 ± 0.003 |
| bsdata_full | 0 | Risiko | 77.3 ± 7.0 | 69.0 ± 5.3 | 60.3 ± 2.1 | 45.7 ± 6.4 | 33.3 ± 4.0 | 19.3 ± 2.9 | 0.321 ± 0.038 |
| bsdata_full | 0 | Zufall | 77.3 ± 7.0 | 73.6 ± 6.2 | 69.8 ± 5.4 | 62.0 ± 5.2 | 53.8 ± 4.3 | 38.0 ± 3.3 | 0.496 ± 0.005 |
| bsdata_full | 0 | Orakel | 77.3 ± 7.0 | 35.3 ± 8.5 | 1.3 ± 2.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.045 ± 0.006 |
| bsdata_full | 1 | Risiko | 65.0 ± 10.6 | 59.7 ± 7.6 | 50.3 ± 9.0 | 37.0 ± 6.2 | 28.7 ± 2.1 | 18.7 ± 2.1 | 0.335 ± 0.019 |
| bsdata_full | 1 | Zufall | 65.0 ± 10.6 | 62.0 ± 10.3 | 58.8 ± 10.1 | 52.2 ± 8.7 | 45.5 ± 7.0 | 31.9 ± 5.0 | 0.496 ± 0.001 |
| bsdata_full | 1 | Orakel | 65.0 ± 10.6 | 23.0 ± 8.9 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.038 ± 0.005 |
| cfd_base | 0 | Risiko | 5.7 ± 3.8 | 5.3 ± 4.0 | 5.3 ± 4.0 | 4.3 ± 3.2 | 3.7 ± 2.1 | 2.3 ± 0.6 | 0.473 ± 0.057 |
| cfd_base | 0 | Zufall | 5.7 ± 3.8 | 5.4 ± 3.5 | 5.1 ± 3.3 | 4.5 ± 3.0 | 4.2 ± 2.7 | 2.9 ± 2.1 | 0.513 ± 0.008 |
| cfd_base | 0 | Orakel | 5.7 ± 3.8 | 2.0 ± 3.5 | 0.7 ± 1.2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.028 ± 0.023 |
| cfd_full | 0 | Risiko | 5.0 ± 5.3 | 4.3 ± 4.2 | 3.7 ± 3.1 | 2.3 ± 2.3 | 2.0 ± 1.7 | 1.3 ± 1.5 | 0.345 ± 0.043 |
| cfd_full | 0 | Zufall | 5.0 ± 5.3 | 4.7 ± 4.9 | 4.5 ± 4.6 | 3.9 ± 4.0 | 3.6 ± 3.7 | 2.6 ± 2.8 | 0.513 ± 0.021 |
| cfd_full | 0 | Orakel | 5.0 ± 5.3 | 2.3 ± 4.0 | 1.0 ± 1.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.024 ± 0.031 |
| cfd_full | 1 | Risiko | 4.3 ± 2.1 | 4.3 ± 2.1 | 4.0 ± 1.7 | 2.3 ± 0.6 | 1.7 ± 0.6 | 1.0 ± 1.0 | 0.314 ± 0.046 |
| cfd_full | 1 | Zufall | 4.3 ± 2.1 | 4.1 ± 1.9 | 3.9 ± 1.8 | 3.3 ± 1.5 | 3.1 ± 1.3 | 2.1 ± 0.9 | 0.502 ± 0.036 |
| cfd_full | 1 | Orakel | 4.3 ± 2.1 | 1.0 ± 1.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.020 ± 0.012 |
| deepcrack_base | 0 | Risiko | 99.0 ± 6.2 | 86.0 ± 5.3 | 76.0 ± 2.0 | 58.7 ± 1.2 | 47.7 ± 3.2 | 25.0 ± 3.6 | 0.325 ± 0.032 |
| deepcrack_base | 0 | Zufall | 99.0 ± 6.2 | 94.2 ± 6.3 | 89.4 ± 5.9 | 79.7 ± 5.5 | 69.3 ± 4.4 | 49.6 ± 3.0 | 0.501 ± 0.007 |
| deepcrack_base | 0 | Orakel | 99.0 ± 6.2 | 78.7 ± 7.4 | 58.0 ± 7.9 | 17.7 ± 10.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.120 ± 0.013 |
| deepcrack_full | 0 | Risiko | 80.7 ± 15.8 | 72.7 ± 15.6 | 65.0 ± 14.1 | 52.3 ± 11.0 | 40.7 ± 10.4 | 23.7 ± 10.1 | 0.350 ± 0.059 |
| deepcrack_full | 0 | Zufall | 80.7 ± 15.8 | 76.8 ± 15.0 | 72.5 ± 13.7 | 64.3 ± 12.3 | 55.9 ± 11.6 | 40.0 ± 8.3 | 0.496 ± 0.005 |
| deepcrack_full | 0 | Orakel | 80.7 ± 15.8 | 60.3 ± 15.0 | 39.7 ± 14.6 | 5.0 ± 8.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.097 ± 0.017 |
| deepcrack_full | 1 | Risiko | 121.7 ± 39.9 | 107.0 ± 36.3 | 94.7 ± 34.6 | 75.3 ± 28.0 | 60.0 ± 22.9 | 33.0 ± 13.1 | 0.331 ± 0.018 |
| deepcrack_full | 1 | Zufall | 121.7 ± 39.9 | 115.7 ± 38.1 | 109.2 ± 35.7 | 97.5 ± 32.1 | 84.7 ± 29.0 | 60.6 ± 21.0 | 0.496 ± 0.006 |
| deepcrack_full | 1 | Orakel | 121.7 ± 39.9 | 101.3 ± 39.1 | 80.7 ± 38.8 | 40.3 ± 38.0 | 14.0 ± 24.2 | 0.0 ± 0.0 | 0.147 ± 0.045 |
| mt_base | 0 | Risiko | 70.7 ± 78.2 | 67.7 ± 75.3 | 62.7 ± 71.5 | 55.3 ± 63.5 | 46.7 ± 52.0 | 29.3 ± 31.6 | 0.465 ± 0.060 |
| mt_base | 0 | Zufall | 70.7 ± 78.2 | 66.9 ± 74.2 | 63.2 ± 69.9 | 56.3 ± 62.4 | 49.8 ± 55.0 | 35.6 ± 39.2 | 0.503 ± 0.001 |
| mt_base | 0 | Orakel | 70.7 ± 78.2 | 35.0 ± 60.6 | 17.3 ± 30.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.033 ± 0.037 |
| mt_full | 0 | Risiko | 70.0 ± 69.7 | 62.3 ± 66.4 | 57.3 ± 61.1 | 50.3 ± 53.1 | 39.7 ± 43.1 | 21.3 ± 23.1 | 0.357 ± 0.034 |
| mt_full | 0 | Zufall | 70.0 ± 69.7 | 66.2 ± 66.1 | 62.5 ± 62.4 | 55.2 ± 55.2 | 48.5 ± 48.9 | 34.8 ± 35.1 | 0.496 ± 0.008 |
| mt_full | 0 | Orakel | 70.0 ± 69.7 | 32.0 ± 55.4 | 14.3 ± 24.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.033 ± 0.033 |
| mt_full | 1 | Risiko | 51.3 ± 53.3 | 45.7 ± 47.9 | 41.3 ± 43.4 | 34.0 ± 34.7 | 27.3 ± 27.4 | 12.0 ± 10.5 | 0.405 ± 0.117 |
| mt_full | 1 | Zufall | 51.3 ± 53.3 | 48.4 ± 50.6 | 45.7 ± 47.7 | 40.4 ± 42.1 | 35.5 ± 37.2 | 25.4 ± 26.7 | 0.479 ± 0.023 |
| mt_full | 1 | Orakel | 51.3 ± 53.3 | 18.7 ± 32.3 | 1.0 ± 1.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.024 ± 0.025 |

### Phase 1: Kosten(k) = k·c_review + n₉₀(K′(k))·c_audit (Einheiten c_audit, δ = 0,05, Power 0,9)

| Variante | Runde | Ziel / Kostenverhältnis | n₉₀(K) ohne Review | Kosten min (Risiko) | k* (Risiko) | Kosten min (Zufall) | Kosten min (Orakel) | Ersparnis Risiko vs. k=0 |
|---|---|---|---|---|---|---|---|---|
| bsdata_base | 0 | zeta*=0.02/c_review=0.25 | 711 ± 38 | 285 ± 19 | 449 ± 195 | 339 ± 12 | 147 ± 3 | 59.8 ± 3.8 |
| bsdata_base | 0 | zeta*=0.02/c_review=1.0 | 711 ± 38 | 573 ± 91 | 364 ± 78 | 711 ± 38 | 196 ± 9 | 19.1 ± 15.4 |
| bsdata_base | 0 | zeta*=0.05/c_review=0.25 | 498 ± 37 | 177 ± 19 | 362 ± 79 | 261 ± 10 | 72 ± 2 | 64.4 ± 4.6 |
| bsdata_base | 0 | zeta*=0.05/c_review=1.0 | 498 ± 37 | 388 ± 52 | 184 ± 80 | 497 ± 37 | 120 ± 8 | 22.1 ± 5.9 |
| bsdata_full | 0 | zeta*=0.02/c_review=0.25 | 730 ± 23 | 323 ± 6 | 769 ± 15 | 340 ± 11 | 150 ± 1 | 55.7 ± 0.7 |
| bsdata_full | 0 | zeta*=0.02/c_review=1.0 | 730 ± 23 | 730 ± 23 | 0 ± 0 | 730 ± 23 | 208 ± 6 | 0.0 ± 0.0 |
| bsdata_full | 0 | zeta*=0.05/c_review=0.25 | 540 ± 15 | 235 ± 12 | 677 ± 112 | 262 ± 9 | 75 ± 2 | 56.4 ± 3.3 |
| bsdata_full | 0 | zeta*=0.05/c_review=1.0 | 540 ± 15 | 529 ± 16 | 75 ± 115 | 540 ± 15 | 132 ± 7 | 2.0 ± 1.7 |
| bsdata_full | 1 | zeta*=0.02/c_review=0.25 | 711 ± 46 | 325 ± 12 | 776 ± 43 | 340 ± 11 | 147 ± 4 | 54.3 ± 1.4 |
| bsdata_full | 1 | zeta*=0.02/c_review=1.0 | 711 ± 46 | 711 ± 46 | 0 ± 1 | 711 ± 46 | 196 ± 12 | 0.0 ± 0.1 |
| bsdata_full | 1 | zeta*=0.05/c_review=0.25 | 497 ± 53 | 241 ± 13 | 659 ± 99 | 261 ± 11 | 72 ± 3 | 51.3 ± 2.7 |
| bsdata_full | 1 | zeta*=0.05/c_review=1.0 | 497 ± 53 | 488 ± 48 | 63 ± 105 | 496 ± 54 | 120 ± 11 | 1.7 ± 2.3 |
| cfd_base | 0 | zeta*=0.02/c_review=0.25 | 80 ± 2 | 80 ± 2 | 0 ± 0 | 79 ± 3 | 66 ± 1 | 0.0 ± 0.0 |
| cfd_base | 0 | zeta*=0.02/c_review=1.0 | 80 ± 2 | 80 ± 2 | 0 ± 0 | 80 ± 2 | 71 ± 4 | 0.0 ± 0.0 |
| cfd_base | 0 | zeta*=0.05/c_review=0.25 | 64 ± 7 | 54 ± 1 | 67 ± 5 | 54 ± 2 | 38 ± 1 | 15.9 ± 7.8 |
| cfd_base | 0 | zeta*=0.05/c_review=1.0 | 64 ± 7 | 64 ± 7 | 0 ± 0 | 64 ± 7 | 43 ± 4 | 0.0 ± 0.0 |
| cfd_full | 0 | zeta*=0.02/c_review=0.25 | 78 ± 5 | 77 ± 5 | 4 ± 7 | 77 ± 5 | 66 ± 1 | 0.9 ± 1.5 |
| cfd_full | 0 | zeta*=0.02/c_review=1.0 | 78 ± 5 | 78 ± 5 | 0 ± 0 | 78 ± 5 | 70 ± 5 | 0.0 ± 0.0 |
| cfd_full | 0 | zeta*=0.05/c_review=0.25 | 60 ± 12 | 51 ± 5 | 41 ± 33 | 51 ± 6 | 38 ± 1 | 13.5 ± 8.3 |
| cfd_full | 0 | zeta*=0.05/c_review=1.0 | 60 ± 12 | 60 ± 12 | 0 ± 0 | 60 ± 12 | 42 ± 5 | 0.0 ± 0.0 |
| cfd_full | 1 | zeta*=0.02/c_review=0.25 | 79 ± 2 | 77 ± 5 | 9 ± 16 | 78 ± 2 | 66 ± 1 | 1.9 ± 3.2 |
| cfd_full | 1 | zeta*=0.02/c_review=1.0 | 79 ± 2 | 79 ± 2 | 0 ± 0 | 79 ± 2 | 69 ± 2 | 0.0 ± 0.0 |
| cfd_full | 1 | zeta*=0.05/c_review=0.25 | 62 ± 6 | 51 ± 6 | 55 ± 25 | 53 ± 2 | 38 ± 1 | 18.5 ± 2.5 |
| cfd_full | 1 | zeta*=0.05/c_review=1.0 | 62 ± 6 | 62 ± 6 | 0 ± 0 | 62 ± 7 | 41 ± 2 | 0.0 ± 0.0 |
| deepcrack_base | 0 | zeta*=0.02/c_review=0.25 | 389 ± 15 | 213 ± 4 | 379 ± 18 | 220 ± 4 | 143 ± 3 | 45.2 ± 1.4 |
| deepcrack_base | 0 | zeta*=0.02/c_review=1.0 | 389 ± 15 | 389 ± 15 | 0 ± 0 | 389 ± 15 | 218 ± 7 | 0.0 ± 0.0 |
| deepcrack_base | 0 | zeta*=0.05/c_review=0.25 | 352 ± 11 | 148 ± 5 | 379 ± 18 | 155 ± 5 | 78 ± 2 | 57.9 ± 0.4 |
| deepcrack_base | 0 | zeta*=0.05/c_review=1.0 | 352 ± 11 | 352 ± 11 | 0 ± 0 | 352 ± 11 | 152 ± 6 | 0.0 ± 0.0 |
| deepcrack_full | 0 | zeta*=0.02/c_review=0.25 | 386 ± 19 | 212 ± 12 | 374 ± 50 | 220 ± 4 | 139 ± 2 | 45.0 ± 1.3 |
| deepcrack_full | 0 | zeta*=0.02/c_review=1.0 | 386 ± 19 | 386 ± 19 | 0 ± 0 | 386 ± 19 | 199 ± 14 | 0.0 ± 0.0 |
| deepcrack_full | 0 | zeta*=0.05/c_review=0.25 | 340 ± 22 | 147 ± 13 | 374 ± 50 | 154 ± 5 | 74 ± 5 | 56.9 ± 2.1 |
| deepcrack_full | 0 | zeta*=0.05/c_review=1.0 | 340 ± 22 | 340 ± 22 | 0 ± 0 | 340 ± 22 | 134 ± 16 | 0.0 ± 0.0 |
| deepcrack_full | 1 | zeta*=0.02/c_review=0.25 | 392 ± 19 | 211 ± 8 | 370 ± 38 | 220 ± 4 | 149 ± 8 | 46.1 ± 1.5 |
| deepcrack_full | 1 | zeta*=0.02/c_review=1.0 | 392 ± 19 | 392 ± 19 | 0 ± 0 | 392 ± 19 | 240 ± 38 | 0.0 ± 0.0 |
| deepcrack_full | 1 | zeta*=0.05/c_review=0.25 | 359 ± 23 | 146 ± 10 | 370 ± 38 | 155 ± 5 | 84 ± 11 | 59.4 ± 0.8 |
| deepcrack_full | 1 | zeta*=0.05/c_review=1.0 | 359 ± 23 | 359 ± 23 | 0 ± 0 | 359 ± 23 | 175 ± 40 | 0.0 ± 0.0 |
| mt_base | 0 | zeta*=0.02/c_review=0.25 | 693 ± 322 | 368 ± 48 | 718 ± 568 | 364 ± 56 | 152 ± 20 | 38.4 ± 27.6 |
| mt_base | 0 | zeta*=0.02/c_review=1.0 | 693 ± 322 | 693 ± 322 | 0 ± 0 | 691 ± 325 | 205 ± 78 | 0.0 ± 0.0 |
| mt_base | 0 | zeta*=0.05/c_review=0.25 | 439 ± 343 | 235 ± 116 | 513 ± 497 | 243 ± 120 | 73 ± 20 | 33.1 ± 25.6 |
| mt_base | 0 | zeta*=0.05/c_review=1.0 | 439 ± 343 | 437 ± 343 | 2 ± 3 | 437 ± 343 | 126 ± 78 | 0.4 ± 0.7 |
| mt_full | 0 | zeta*=0.02/c_review=0.25 | 746 ± 220 | 369 ± 6 | 940 ± 25 | 391 ± 7 | 152 ± 17 | 47.1 ± 17.6 |
| mt_full | 0 | zeta*=0.02/c_review=1.0 | 746 ± 220 | 725 ± 244 | 8 ± 8 | 744 ± 223 | 204 ± 70 | 3.8 ± 4.9 |
| mt_full | 0 | zeta*=0.05/c_review=0.25 | 442 ± 315 | 220 ± 70 | 470 ± 432 | 257 ± 88 | 73 ± 17 | 37.6 ± 27.0 |
| mt_full | 0 | zeta*=0.05/c_review=1.0 | 442 ± 315 | 428 ± 321 | 20 ± 22 | 441 ± 315 | 125 ± 70 | 4.3 ± 5.0 |
| mt_full | 1 | zeta*=0.02/c_review=0.25 | 641 ± 347 | 336 ± 73 | 650 ± 566 | 340 ± 97 | 147 ± 13 | 36.0 ± 31.9 |
| mt_full | 1 | zeta*=0.02/c_review=1.0 | 641 ± 347 | 641 ± 347 | 0 ± 0 | 636 ± 353 | 185 ± 53 | 0.0 ± 0.0 |
| mt_full | 1 | zeta*=0.05/c_review=0.25 | 384 ± 306 | 196 ± 96 | 436 ± 385 | 232 ± 126 | 69 ± 13 | 32.7 ± 31.0 |
| mt_full | 1 | zeta*=0.05/c_review=1.0 | 384 ± 306 | 379 ± 307 | 15 ± 27 | 382 ± 306 | 106 ± 53 | 1.4 ± 2.4 |

### Gate G1: Bild-Recall in Risiko-Strata je Runde

| Variante | Runde | Bild-Recall je Risiko-Quintil (Runde-0-Risiko; Q1 = niedrigstes Risiko) | Anteil Fehlbilder unter Median des eigenen Risikos |
|---|---|---|---|
| bsdata_base | 0 | Q1: 99.4 ± 0.6 · Q2: 60.5 ± 30.2 · Q3: 74.3 ± 4.1 · Q4: 30.0 ± 52.0 · Q5: 19.4 ± 17.0 | 11.4 ± 7.3 |
| bsdata_full | 0 | Q1: 92.4 ± 9.7 · Q2: 87.5 ± 2.6 · Q3: 71.4 ± 14.3 · Q4: 56.4 ± 19.3 · Q5: 30.4 ± 11.9 | 25.3 ± 5.6 |
| bsdata_full | 1 | Q1: 91.9 ± 9.3 · Q2: 87.5 ± 2.6 · Q3: 71.4 ± 14.3 · Q4: 62.4 ± 13.2 · Q5: 52.3 ± 19.3 | 29.0 ± 3.0 |
| cfd_base | 0 | Q1: 94.1 ± 5.9 · Q2: 92.2 ± 3.4 · Q3: 96.1 ± 3.4 · Q4: 92.2 ± 13.6 · Q5: 91.7 ± 3.6 | 48.9 ± 18.4 |
| cfd_full | 0 | Q1: 96.1 ± 6.8 · Q2: 96.1 ± 3.4 · Q3: 98.0 ± 3.4 · Q4: 96.1 ± 6.8 · Q5: 83.3 ± 19.1 | 20.2 ± 17.8 |
| cfd_full | 1 | Q1: 96.1 ± 3.4 · Q2: 98.0 ± 3.4 · Q3: 94.1 ± 5.9 · Q4: 96.1 ± 3.4 · Q5: 89.6 ± 9.5 | 17.8 ± 16.8 |
| deepcrack_base | 0 | Q1: 94.7 ± 0.6 · Q2: 84.5 ± 2.4 · Q3: 78.1 ± 3.5 · Q4: 70.7 ± 8.1 · Q5: 50.1 ± 11.2 | 25.1 ± 4.7 |
| deepcrack_full | 0 | Q1: 92.5 ± 6.4 · Q2: 88.6 ± 2.1 · Q3: 81.4 ± 9.1 · Q4: 73.6 ± 4.9 · Q5: 65.2 ± 6.9 | 28.5 ± 7.9 |
| deepcrack_full | 1 | Q1: 85.6 ± 6.3 · Q2: 79.5 ± 5.3 · Q3: 73.2 ± 13.4 · Q4: 63.5 ± 8.7 · Q5: 49.5 ± 14.5 | 26.7 ± 3.4 |
| mt_base | 0 | Q1: 93.3 ± 6.9 · Q2: 78.6 ± 22.4 · Q3: 67.8 ± 40.0 · Q4: 62.5 ± 43.3 · Q5: 51.1 ± 47.3 | 50.3 ± 18.4 |
| mt_full | 0 | Q1: 96.7 ± 2.0 · Q2: 81.1 ± 21.9 · Q3: 72.2 ± 30.1 · Q4: 62.1 ± 38.1 · Q5: 58.8 ± 32.9 | 31.3 ± 10.4 |
| mt_full | 1 | Q1: 96.9 ± 2.8 · Q2: 84.1 ± 16.0 · Q3: 81.9 ± 22.5 · Q4: 74.4 ± 24.1 · Q5: 64.9 ± 31.7 | 32.6 ± 15.3 |

### Gate G2: OOF-Schwelle vs. U-Oracle-Schwelle

| Variante | Runde | t_seed OOF | t_seed U-Oracle | t_ext OOF | t_ext U-Oracle | Δ tol-F1 (Oracle − OOF) | Δ Defekt-Recall | Δ Defekt-Precision |
|---|---|---|---|---|---|---|---|---|
| bsdata_base | 0 | 0.77 ± 0.06 | 0.73 ± 0.06 | 0.77 ± 0.06 | 0.58 ± 0.11 | 0.8 ± 0.7 | 2.7 ± 1.6 | -5.4 ± 4.6 |
| bsdata_full | 0 | 0.80 ± 0.00 | 0.73 ± 0.06 | 0.67 ± 0.23 | 0.66 ± 0.16 | 2.6 ± 3.8 | 4.2 ± 5.1 | -4.7 ± 4.6 |
| bsdata_full | 1 | 0.80 ± 0.00 | 0.77 ± 0.06 | 0.80 ± 0.00 | 0.77 ± 0.06 | 0.0 ± 0.1 | 0.5 ± 1.0 | -3.1 ± 5.3 |
| cfd_base | 0 | 0.53 ± 0.31 | 0.33 ± 0.06 | 0.50 ± 0.36 | 0.30 ± 0.10 | 2.2 ± 1.9 | 1.8 ± 4.0 | -5.1 ± 16.5 |
| cfd_full | 0 | 0.40 ± 0.26 | 0.20 ± 0.00 | 0.18 ± 0.07 | 0.20 ± 0.00 | 1.6 ± 2.4 | 3.3 ± 4.9 | -19.2 ± 25.3 |
| cfd_full | 1 | 0.67 ± 0.23 | 0.47 ± 0.06 | 0.67 ± 0.23 | 0.38 ± 0.13 | 8.9 ± 8.1 | 2.0 ± 1.9 | -22.4 ± 13.8 |
| deepcrack_base | 0 | 0.80 ± 0.00 | 0.77 ± 0.06 | 0.80 ± 0.00 | 0.34 ± 0.06 | 4.8 ± 1.8 | 1.6 ± 1.7 | -5.0 ± 3.5 |
| deepcrack_full | 0 | 0.33 ± 0.06 | 0.27 ± 0.12 | 0.20 ± 0.17 | 0.12 ± 0.04 | 4.0 ± 5.3 | 3.5 ± 6.9 | -12.5 ± 23.4 |
| deepcrack_full | 1 | 0.80 ± 0.00 | 0.43 ± 0.15 | 0.80 ± 0.00 | 0.40 ± 0.20 | 20.1 ± 14.8 | 17.5 ± 7.3 | -46.2 ± 7.5 |
| mt_base | 0 | 0.47 ± 0.25 | 0.67 ± 0.06 | 0.30 ± 0.21 | 0.31 ± 0.15 | 14.2 ± 17.1 | -13.4 ± 21.6 | 5.5 ± 8.4 |
| mt_full | 0 | 0.50 ± 0.30 | 0.47 ± 0.12 | 0.35 ± 0.19 | 0.27 ± 0.11 | 7.0 ± 6.4 | 9.4 ± 27.5 | 0.3 ± 8.8 |
| mt_full | 1 | 0.57 ± 0.32 | 0.67 ± 0.12 | 0.39 ± 0.32 | 0.34 ± 0.12 | 13.5 ± 10.5 | 3.4 ± 16.9 | 0.8 ± 5.5 |

## Gates (phase1_gates.py)

#### G1: Bild-Recall in Q1–Q3 (Runde-0-Risiko), Runde 1 vs. Runde 0

| Variante | Seed | ρ_low R0 | ρ_low R1 | Δ (pp) | K R0 | K R1 | K′(k*) R0 | K′(k*) R1 | Kosten min R0 | Kosten min R1 |
|---|---|---|---|---|---|---|---|---|---|---|
| bsdata_full | 0 | 86.9 | 86.9 | 0.0 | 70 | 69 | 0 | 0 | 326 | 329 |
| bsdata_full | 1 | 88.3 | 88.3 | 0.0 | 78 | 73 | 0 | 0 | 326 | 335 |
| bsdata_full | 2 | 89.9 | 88.9 | -0.9 | 84 | 53 | 0 | 0 | 317 | 311 |
| cfd_full | 0 | 94.1 | 98.0 | 3.9 | 11 | 5 | 11 | 5 | 82 | 80 |
| cfd_full | 1 | 98.0 | 94.1 | -3.9 | 3 | 6 | 1 | 6 | 76 | 80 |
| cfd_full | 2 | 98.0 | 96.1 | -2.0 | 1 | 2 | 1 | 0 | 73 | 72 |
| deepcrack_full | 0 | 83.6 | 70.4 | -13.2 | 98 | 167 | 0 | 0 | 213 | 219 |
| deepcrack_full | 1 | 92.7 | 84.5 | -8.2 | 67 | 92 | 0 | 0 | 200 | 204 |
| deepcrack_full | 2 | 86.2 | 83.5 | -2.8 | 77 | 106 | 0 | 0 | 224 | 210 |
| mt_full | 0 | 91.5 | 99.0 | 7.5 | 44 | 4 | 0 | 4 | 371 | 253 |
| mt_full | 1 | 64.9 | 73.8 | 8.9 | 149 | 109 | 0 | 0 | 362 | 364 |
| mt_full | 2 | 96.8 | 92.6 | -4.1 | 17 | 41 | 0 | 0 | 374 | 392 |

**G1-Entscheidung je Datensatz** (positiv = Self-Training senkt den Recall in Q1–Q3):

* bsdata_full: negativ (mittleres Δ = -0.3 pp, 3 Seeds)
* cfd_full: negativ (mittleres Δ = -0.7 pp, 3 Seeds)
* deepcrack_full: POSITIV (mittleres Δ = -8.0 pp, 3 Seeds)
* mt_full: negativ (mittleres Δ = +4.1 pp, 3 Seeds)

#### G2: Schwellenübertrag OOF → U (Runde 0)

| Variante | Seed | t_seed OOF | t_seed Oracle | t_ext OOF | t_ext Oracle | tol-F1 OOF-Schwelle | tol-F1 Oracle-Schwelle | Lücke (pp) | Δ Defekt-Recall (pp) |
|---|---|---|---|---|---|---|---|---|---|
| bsdata_full | 0 | 0.80 | 0.70 | 0.80 | 0.70 | 87.1 | 87.9 | 0.7 | 2.6 |
| bsdata_full | 1 | 0.80 | 0.80 | 0.80 | 0.80 | 90.7 | 90.7 | -0.0 | 0.0 |
| bsdata_full | 2 | 0.80 | 0.70 | 0.40 | 0.49 | 74.7 | 81.6 | 7.0 | 9.9 |
| cfd_full | 0 | 0.70 | 0.20 | 0.24 | 0.20 | 78.3 | 82.7 | 4.3 | 8.9 |
| cfd_full | 1 | 0.30 | 0.20 | 0.10 | 0.20 | 86.4 | 86.9 | 0.5 | 0.9 |
| cfd_full | 2 | 0.20 | 0.20 | 0.20 | 0.20 | 81.1 | 81.1 | -0.0 | 0.0 |
| deepcrack_full | 0 | 0.40 | 0.20 | 0.40 | 0.14 | 65.6 | 75.7 | 10.1 | 10.3 |
| deepcrack_full | 1 | 0.30 | 0.20 | 0.10 | 0.07 | 81.1 | 82.0 | 0.8 | 3.6 |
| deepcrack_full | 2 | 0.30 | 0.40 | 0.10 | 0.14 | 73.7 | 74.7 | 1.0 | -3.5 |
| mt_full | 0 | 0.50 | 0.40 | 0.50 | 0.40 | 56.6 | 57.1 | 0.6 | 2.9 |
| mt_full | 1 | 0.80 | 0.40 | 0.40 | 0.20 | 33.9 | 47.2 | 13.3 | 39.6 |
| mt_full | 2 | 0.20 | 0.60 | 0.14 | 0.21 | 39.5 | 46.5 | 7.0 | -14.3 |

**G2-Entscheidung je Datensatz** (positiv = Übertrag verbesserungsbedürftig, mittlere Lücke ≥ 5 pp):

* bsdata_full: negativ (mittlere Lücke = 2.6 pp, 3 Seeds)
* cfd_full: negativ (mittlere Lücke = 1.6 pp, 3 Seeds)
* deepcrack_full: negativ (mittlere Lücke = 4.0 pp, 3 Seeds)
* mt_full: POSITIV (mittlere Lücke = 7.0 pp, 3 Seeds)

## G1-Kontrolle post hoc (g1_fixed_threshold.py)

| Lauf | ρ_low R0 | ρ_low R1 | ρ_low R1@T0 | ζ R0 | ζ R1 | ζ R1@T0 | φ R0 | φ R1 | φ R1@T0 |
|---|---|---|---|---|---|---|---|---|---|
| bsdata_full_s0 | 86.9 | 86.9 | 86.9 | 8.1 | 7.9 | 7.9 | 0.19 | 0.31 | 0.31 |
| bsdata_full_s1 | 88.3 | 88.3 | 88.3 | 9.0 | 8.4 | 8.4 | 0.17 | 0.31 | 0.32 |
| bsdata_full_s2 | 89.9 | 88.9 | 91.2 | 10.5 | 6.6 | 5.6 | 0.30 | 0.45 | 0.61 |
| cfd_full_s0 | 94.1 | 98.0 | 98.0 | 13.1 | 6.0 | 3.6 | 0.29 | 0.65 | 0.46 |
| cfd_full_s1 | 98.0 | 94.1 | 98.0 | 3.6 | 7.1 | 1.2 | 1.12 | 0.67 | 1.30 |
| cfd_full_s2 | 98.0 | 96.1 | 98.0 | 1.2 | 2.4 | 1.2 | 7.95 | 4.90 | 10.94 |
| deepcrack_full_s0 | 83.6 | 70.4 | 87.6 | 23.6 | 40.1 | 16.8 | 3.00 | 0.70 | 4.38 |
| deepcrack_full_s1 | 92.7 | 84.5 | 98.7 | 17.3 | 23.8 | 2.3 | 2.48 | 1.24 | 7.22 |
| deepcrack_full_s2 | 86.2 | 83.5 | 96.1 | 18.2 | 25.1 | 2.8 | 4.67 | 1.68 | 7.64 |
| mt_full_s0 | 91.5 | 99.0 | 93.5 | 4.2 | 0.4 | 2.5 | 2.33 | 3.76 | 2.62 |
| mt_full_s1 | 64.9 | 73.8 | 73.8 | 14.1 | 10.3 | 10.3 | 0.65 | 0.99 | 1.00 |
| mt_full_s2 | 96.8 | 92.6 | 99.5 | 1.6 | 3.9 | 0.2 | 4.46 | 2.50 | 8.66 |
