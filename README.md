# auto-label-tool

Automatisches, pixelgenaues Labeling von Kratzern und feinen Oberflächendefekten aus **sehr wenigen
Masken** – mit **messbarer, statistisch abgesicherter Labelqualität**.

> Status: Forschungs-Prototyp. Ergebnisse auf öffentlichen Industriedatensätzen siehe
> [`docs/RESULTS.md`](docs/RESULTS.md), Methode und Begründungen in [`docs/METHOD.md`](docs/METHOD.md).

## Die Idee in fünf Sätzen

1. **Kreuzvalidiertes Ensemble**: Die wenigen gelabelten Bilder werden in K Folds geteilt; jedes
   Ensemble-Mitglied lässt einen Fold aus. So bekommt *jedes* gelabelte Bild eine ehrliche
   Vorhersage – die Grundlage für Kalibrierung, Schwellen, Garantien und Qualitätsschätzung, ohne ein
   einziges Bild für Validierung zu „opfern“.
2. **Ungelabelte Bilder arbeiten mit**: als Normalitätsmodell (was ist normale Oberfläche inkl.
   Reflexionen/Texturen?), als Hintergründe für *Residual-Copy-Paste* gelabelter Defekte (das Modell
   sieht jeden Defekt auf hunderten echten Oberflächen und in abgeschwächter Form) und für
   *fold-konsistentes Self-Training*.
3. **Übersehene Defekte werden kontrolliert**: die Detektionsschwelle kommt aus *Conformal Prediction*
   – unter Austauschbarkeit gilt P(Defekt gefunden) ≥ 1−α, verteilungsfrei. Ist die Labelmenge zu klein
   für eine Garantie, sagt das Tool das.
4. **Jedes Label bewertet sich selbst** (erwarteter Dice, Wahrscheinlichkeit eines übersehenen
   Defekts, Ensemble-Uneinigkeit, unerklärte Anomalie) → priorisierte Review-Liste.
5. **Qualität wird gemessen, nicht behauptet**: ein kleines *zufälliges* Audit + die
   Selbsteinschätzung aller Bilder ergeben mit *Prediction-Powered Inference* gültige
   Konfidenzintervalle für Recall/Precision/Dice des gesamten Label-Pools.

## Schnellstart

```bash
pip install -e .            # torch, timm, opencv, scikit-learn, scipy
# images/: alle Bilder, masks/: die vorhandenen Masken (gleicher Dateiname, Defekt > 0)
autolabel run --images images/ --masks masks/ --out run1/
```

Ausgabe in `run1/`:

| Datei/Ordner | Inhalt |
|---|---|
| `labels/` | Pseudo-Label-Masken (0/255) |
| `labels_recall_safe/` | Masken mit konformer Detektionsschwelle (mehr Recall, mehr Fehlalarme) |
| `ignore/` | unsichere Pixel → beim Training des Produktivmodells als *ignore* behandeln |
| `prob/`, `overlays/` | Wahrscheinlichkeitskarten und Overlays für die Sichtprüfung |
| `review_queue.csv` | alle Bilder nach Risiko sortiert (oben zuerst prüfen) |
| `audit_sample.txt` | zufällige Audit-Stichprobe (für die Qualitätsmessung) |
| `quality_report.json` | Schwellen, Kreuzvalidierungs-Metriken, Recall-Garantie, Selbsteinschätzung |

Qualität messen: die Bilder aus `audit_sample.txt` von Hand labeln (z. B. 30 Stück) und

```bash
autolabel estimate --run run1/ --audit-masks audit_masks/
```

→ Punktschätzer und 90-%-Konfidenzintervalle für Pixel-Recall/-Precision, Defekt-Recall und
mittleren Dice aller Auto-Labels (PPI), zum Vergleich auch das reine Audit-Intervall.

## Benchmark reproduzieren

```bash
bash scripts/download_datasets.sh data/      # Magnetic-Tile, BSData, DeepCrack, CrackForest (GitHub)
export AUTOLABEL_DATA=data AUTOLABEL_CACHE=cache
python scripts/run_benchmark.py --dataset mt --n-defect 10 --n-free 5 --seed 0 --out runs/mt_full_s0
python scripts/summarize.py --runs "runs/*"
```

`run_benchmark.py` simuliert das Szenario: ein Testset T wird gruppenweise abgetrennt, aus dem Rest
werden wenige Bilder als gelabelt gewählt (L), alle übrigen sind der ungelabelte Pool U, dessen
Ground Truth nur zur *Bewertung* der Auto-Labels dient.

## Struktur

```
autolabel/
  data.py          Datensätze, Few-Label-Split-Protokoll
  model.py         U-Net (timm-Encoder) mit Full-Resolution-Pfad
  augment.py       defektzentriertes Cropping, Residual-Copy-Paste, prozedurale Defekte
  train.py         Training eines Ensemble-Mitglieds (Labels, Synthese, Pseudo-Labels mit Ignore)
  anomaly.py       kontaminationsrobustes Normalitätsmodell aus ungelabelten Bildern
  pipeline.py      Gesamtablauf (Cross-Fitting, Kalibrierung, Schwellen, Self-Training, Selbsteinschätzung)
  postprocess.py   Hysterese-Schwellen, Schwellenwahl
  calibration.py   isotone Kalibrierung
  conformal.py     Recall-Garantien (Split-Conformal, PAC)
  quality.py       Selbsteinschätzung je Bild, Review-Priorität
  ppi.py           Audit-Schätzer mit Konfidenzintervallen (PPI)
  classify.py      Defekttyp-Zuordnung (few-shot) mit konformen Vorhersagemengen
  evaluate.py      Benchmark-Auswertung gegen verborgene Ground Truth
  cli.py           Kommandozeile
scripts/           Benchmark, Auswertung, Datensatz-Download
docs/              Methode und Ergebnisse
tests/             Unit-Tests (pytest)
```
