# Hand-off: Weiterarbeit lokal (Mac mini M4)

Stand: 18.10.2026. Dieses Dokument reicht, um ohne den bisherigen Chat weiterzumachen.
Begründungen der Methode stehen in [`METHOD.md`](METHOD.md), Nutzung im [`README`](../README.md).

> **Nachtrag Forschung (18.10.2026, abends):** Fünf Literaturberichte, ein Theoriebeitrag und eine
> Synthese liegen in [`research/`](research/). **Zuerst [`research/00_SYNTHESE.md`](research/00_SYNTHESE.md)
> lesen.** Dort standen in v1 das Nachfolger-Konzept „CertAL“ (durch den Nachtrag vom 19.10. überholt: Kern ist jetzt
> CertAL-lite), eine Liste der Schwächen des aktuellen Codes
> (§3, u. a. gilt die Recall-Garantie für U nicht) und eine überarbeitete Roadmap (§6). Diese ersetzt
> die Prioritäten in Abschnitt 4/7 dieses Dokuments, soweit sie sich widersprechen. Die
> Baseline-Läufe aus Abschnitt 4 bleiben als Referenz sinnvoll (Roadmap-Phase 1).

> **Nachtrag 19.10.2026 (Red-Team):** Der Red-Team-Bericht [`research/07_red_team.md`](research/07_red_team.md) hat
> CertAL geprüft. Ergebnis: Das volle CertAL ist überdimensioniert, und mehrere Gültigkeitsaussagen galten nur unter
> unausgesprochenen Annahmen (Mensch = Gold, Review-Stratum außerhalb des Audits, nachgelernte Rangfolge, D_max,
> überwachte Fixe-n-Schranke). Empfohlener Kern ist **CertAL-lite**: klassische Akzeptanzprüfung mit Rectifying
> (Review in Risikoreihenfolge, danach uniformes Audit ohne Zurücklegen aus dem ganzen Pool unter Protokoll P,
> exakte Schranke bei festem n), siehe [`research/00_SYNTHESE.md`](research/00_SYNTHESE.md) v2 §4 und 07 §6.
> Reihenfolge (wie 00 §6): Zuerst laufen die Baseline-Läufe aus Phase 1 (Abschnitt 4 dieses Dokuments); keine
> weitere CertAL-Komponente vor Abschluss von Phase 1. Ausgenommen ist nur `autolabel/certify.py`, der bereits
> vorhandene Statistik-Kern von CertAL-lite (Phase 0). Er ist teilweise implementiert und unit-getestet
> (`tests/test_certify.py`), aber noch nicht in die Labeling-Pipeline eingebunden und nicht auf echten Daten
> erprobt; φ-Zertifikat, R als eigenes Stratum und mehrere adversariale Sims fehlen noch (00 §4.7). Diese
> offenen Teile werden neben Phase 1 ergänzt, nicht vor ihr. [`research/04_guarantees_measurement.md`](research/04_guarantees_measurement.md)
> enthält am Anfang eine Korrekturliste.

---

## 1. Ziel (Kurzfassung der Aufgabe)

Pixelgenaue Auto-Labels für anspruchsvolle industrielle Defektsegmentierung (verschiedene Aufgaben,
Defekte klein bis groß, eine oder mehrere Klassen) aus **sehr wenigen Masken**
und vielen ungelabelten Bildern. Hohe Labelqualität, übersehene Defekte sind besonders kritisch, und
die Qualität muss **messbar** sein, obwohl kaum Ground Truth existiert. Die Lösung soll auf echten
öffentlichen Datensätzen validiert werden.

## 2. Was fertig ist – und was davon getestet ist

| Baustein | Datei | Status |
|---|---|---|
| Datensätze + Few-Label-Split (T gruppenweise getrennt, L ⊂ Pool, U = Rest) | `autolabel/data.py` | genutzt, Splits geprüft |
| Metriken (Pixel, Pixel mit 2-px-Toleranz, Defekt-/Komponentenebene, Bildebene, AP/AUROC) | `autolabel/metrics.py` | Unit-Tests ✔ |
| U-Net mit Full-Res-Pfad, timm-Encoder, Gewichte von GitHub-Releases | `autolabel/model.py` | genutzt ✔ |
| Defektzentriertes Cropping, Residual-Copy-Paste, prozedurale Kratzer | `autolabel/augment.py` | Unit-Tests ✔, visuell geprüft ✔ |
| Training eines Members (Labels + Synthese + Pseudo-Labels mit Ignore) | `autolabel/train.py` | genutzt ✔ |
| Normalitätsmodell (PatchCore-artig, eigenes Bild ausgeschlossen, k=3) | `autolabel/anomaly.py` | genutzt ✔, kNN-Test ✔ |
| Pipeline: Cross-Fitting K=4, Kalibrierung, Hysterese-Schwellen, konforme Schwelle, fold-konsistentes Self-Training, Selbsteinschätzung, Review-Risiko | `autolabel/pipeline.py` | Runde 0 auf MT voll gelaufen ✔, Runde 1 nur im Mini-Smoke-Test |
| Recall-Garantie (Split-Conformal + PAC/Clopper-Pearson) | `autolabel/conformal.py` | Unit-Tests ✔ |
| PPI-Audit-Schätzer (endliche Population, Ratio + Mittelwert) | `autolabel/ppi.py` | Unit-Tests + Simulation ✔ |
| Benchmark-Auswertung (Qualität U/T, Konformitäts-Check, Selbsteinschätzung, Audit-Simulation, Review-Kurven, Anomalie-Baseline) | `autolabel/evaluate.py` | lief im Mini-Smoke-Test (ältere Version) |
| Benchmark-Runner | `scripts/run_benchmark.py` | Mini-Smoke-Test ✔ (vor den letzten Änderungen) |
| Re-Evaluation ohne Training, Tabellen | `scripts/reevaluate.py`, `scripts/summarize.py` | **ungetestet** |
| Oracle (volle GT) | `scripts/run_oracle.py` | **ungetestet** |
| Defekttyp-Zuordnung (few-shot, konforme Mengen) | `autolabel/classify.py`, `scripts/eval_types.py` | **ungetestet** |
| CLI `autolabel run` / `autolabel estimate` | `autolabel/cli.py` | **ungetestet** |
| Geräteauswahl CUDA > MPS > CPU | `model.get_device()` | nur CPU-Pfad getestet, **MPS ungetestet** |

`python -m pytest -q` → 13 Tests grün (CPU).

### Erste (einzige) echte Zahlen
Magnetic Tile, Seed 0, L = 15 Bilder (10 mit Defekt, 11 Defekte), U = 1060 (301 defekt), T = 269.
Runde 0, **kreuzvalidiert auf L** (Out-of-fold, nicht U!):
tol-F1 = 0,31, Defekt-Recall = 0,73, gewählte Schwellen t_seed = 0,6 / t_ext = 0,42.
Mit nur 11 gelabelten Defekten ist die konforme Schwelle für 90 % Recall = 0,086 (also sehr niedrig),
eine PAC-Garantie ist nicht möglich (t = 0). Das ist das erwartete, ehrliche Verhalten.
**Metriken auf U/T gibt es noch keine** – der Remote-Lauf war dafür zu langsam (≈ 75 min/Lauf auf
4 CPU-Kernen) und wurde für den Umzug abgebrochen.

## 3. Setup auf dem Mac mini M4

```bash
git clone https://github.com/benjaminschn/auto-label-tool && cd auto-label-tool
python3 -m venv .venv && source .venv/bin/activate        # Python 3.11/3.12 empfohlen
pip install torch torchvision timm opencv-python-headless scikit-learn scikit-image scipy pillow pytest
pip install -e .
python -m pytest -q                                        # 13 passed erwartet

bash scripts/download_datasets.sh data/                    # ~0,5 GB, alles von GitHub
export AUTOLABEL_DATA=$PWD/data AUTOLABEL_CACHE=$PWD/cache
export AUTOLABEL_DEVICE=mps                                # automatisch, hier nur explizit
export PYTORCH_ENABLE_MPS_FALLBACK=1                       # falls ein Op auf MPS fehlt -> CPU
```

Vortrainierte Gewichte (resnet18d, wide_resnet50_2, …) werden beim ersten Lauf automatisch nach
`~/.cache/autolabel/weights` geladen (`AUTOLABEL_WEIGHTS` überschreibt den Pfad).

### Erste Validierung (in dieser Reihenfolge, je wenige Minuten)
1. **MPS-Smoke-Test** – prüft, dass alles auf MPS läuft (Log-Zeilen zeigen `[mps]`):
   ```bash
   python scripts/run_benchmark.py --dataset mt --iters 20 --k-folds 2 --rounds 2 \
       --max-unlabeled 40 --test-frac 0.05 --out runs/smoke_s0
   python scripts/reevaluate.py runs/smoke_s0
   python scripts/summarize.py --runs "runs/smoke_*"
   python scripts/eval_types.py runs/smoke_s0
   ```
   Falls MPS Ärger macht: `AUTOLABEL_DEVICE=cpu` (M4-CPU ist trotzdem deutlich schneller als die
   Remote-Maschine).
2. **CLI-Test** auf einem Mini-Ordner (z. B. 30 MT-Bilder, 5 Masken) – `autolabel run` und
   `autolabel estimate` wurden noch nie end-to-end ausgeführt.
3. **Geschwindigkeit messen**: Log zeigt Sekunden pro 100 Iterationen. Remote (CPU): ≈ 46 s/100 it,
   Inferenz ≈ 0,08 s/Bild/Member. Auf MPS deutlich schneller erwartet; Datengenerierung (numpy/cv2,
   1 Thread) kann dann zum Engpass werden.

## 4. Experimentplan (Priorität absteigend)

Namenskonvention der Ordner: `runs/<dataset>_<variante>_s<seed>` (wird von `summarize.py` gruppiert).
Ein voller Lauf schreibt `report.json` (alle Metriken), `results.pkl.gz` (Masken, Wahrscheinlichkeiten
uint8, Features – für `reevaluate.py`) und `log.txt`.

```bash
R="python scripts/run_benchmark.py"
# A) Hauptmethode, je Datensatz 3 Seeds
for s in 0 1 2; do
  $R --dataset mt        --seed $s --out runs/mt_full_s$s
  $R --dataset bsdata    --seed $s --out runs/bsdata_full_s$s
  $R --dataset cfd       --seed $s --out runs/cfd_full_s$s
  $R --dataset deepcrack --seed $s --out runs/deepcrack_full_s$s
done
# B) Ablation: ohne ungelabelte Daten (keine Synthese, kein Normalitätsmodell, kein Self-Training)
for d in mt bsdata cfd; do for s in 0 1; do
  $R --dataset $d --seed $s --rounds 1 --p-synth 0 --no-anomaly --out runs/${d}_base_s$s
done; done
# C) Label-Budget
for n in 5 20 40; do $R --dataset mt --n-defect $n --seed 0 --out runs/mt_n${n}_s0; done
# D) weitere Ablationen (Seed 0, MT + BSData)
$R --dataset mt --teacher ensemble --out runs/mt_teacherens_s0      # fold- vs. Ensemble-Teacher
$R --dataset mt --p-procedural 0.3 --out runs/mt_proc_s0            # prozedurale Kratzer
$R --dataset mt --bg-ignore-quantile 1.0 --out runs/mt_noign_s0      # ohne Anomalie-Ignore
$R --dataset bsdata --no-rot90 --out runs/bsdata_norot_s0
# E) Obergrenze: volle Ground Truth
python scripts/run_oracle.py --dataset mt --out runs/oracle_mt
python scripts/run_oracle.py --dataset bsdata --out runs/oracle_bsdata
# Auswertung
python scripts/summarize.py --runs "runs/*" > docs/RESULTS_raw.md
```

Wichtigste Vergleiche für die Ergebnisdoku (`docs/RESULTS.md` ist noch zu schreiben):
* `*_base` vs. `*_full` Runde 0 vs. `*_full` Runde 1 → Nutzen von Synthese / Self-Training.
* `rounds.*.U` vs. `rounds.*.U_oracle_threshold` → wie gut übertragen OOF-Schwellen aufs Ensemble?
* `rounds.*.U_single_member` vs. `rounds.*.U` → Ensemble-Gewinn.
* `rounds.*.X` (Testset) vs. `oracle_*` → Abstand zur vollen Supervision.
* `conformal` → hält die Recall-Garantie auf U (realised ≥ target)?
* `self_assessment` → Selbsteinschätzung vs. Wahrheit; AUROC der Fehler-Signale.
* `audit` → Abdeckung (Soll 90 %) und Breite PPI vs. klassisch bei n = 10/20/40/80.
* `review` → Defekt-Recall nach Prüfung der riskantesten 5/10/20 % vs. zufällige Reihenfolge.

## 5. Bekannte Risiken / offene Fragen

1. **Schwellen-Transfer**: Schwellen werden auf OOF-Vorhersagen *eines* Members gewählt, auf U aber
   auf den Ensemble-Mittelwert angewandt (Mittelung glättet Wahrscheinlichkeiten). Prüfen über
   `U_oracle_threshold`; falls großer Abstand: Schwellen auf Ensemble-Pseudo-OOF kalibrieren oder
   Rang-/Quantil-Abgleich der Verteilungen.
2. **Wenige gelabelte Defekte ⇒ schwache Garantien**: n Defekte, α = 0,1 braucht n ≥ 9 für eine
   nicht-triviale konforme Schwelle, PAC deutlich mehr. Optionen: Garantien auf Komponentenebene
   über mehr Bilder, α = 0,2, oder iterativ (Review-Bilder → L).
3. **Crack-Datensätze (DeepCrack, CFD)**: jedes Bild enthält Risse ⇒ Synthese-Hintergründe tragen
   ungelabelte Risse. Abgefedert durch `bg_ignore_quantile=0.98` (Runde 0) und Teacher-Ignore
   (Runde 1); Effekt ist ungemessen (Ablation `--bg-ignore-quantile 1.0`).
4. **MT-Klassen „uneven“/„fray“** sind großflächig und dominieren Pixelmetriken – daher immer auch
   Defektebene berichten. Großflächige Defekte werden nicht synthetisiert (`paste_large=False`).
5. **BSData**: L und U dürfen aus derselben Verschleißsequenz stammen (realistisch: L ist Teil des
   Pools); T ist sequenzdisjunkt. `make_split(strict_groups=True)` für die konservative Variante.
6. **PPI**: Default λ = 1 (beste Abdeckung bei kleinen Audits); λ-Tuning (`lam=None`) ist schmaler,
   unterschreitet aber bei n < 50 die Soll-Abdeckung (Simulation in `tests/` bzw. Chat-Notizen).
7. **MPS**: ungetestet. Bekannte Fallstricke: kein float64 (wird nicht genutzt), manche Ops fehlen
   (→ `PYTORCH_ENABLE_MPS_FALLBACK=1`), Nichtdeterminismus.

## 6. Datensätze

Lokal verfügbar über `scripts/download_datasets.sh` (alle GitHub):

| Name | Inhalt | Warum relevant |
|---|---|---|
| Magnetic Tile (`mt`) | 1344 Graubilder, 5 Defektklassen + 952 defektfrei | Risse dünn/kontrastarm, Defekte selten |
| BSData (`bsdata`) | 1104 Bilder Kugelgewindetrieb, Pitting, 710 defektfrei | **reflektierende Metalloberfläche**, kleine Defekte, Schmutz |
| DeepCrack (`deepcrack`) | 537 Bilder, Risse | dünne Strukturen (Stresstest) |
| CrackForest (`cfd`) | 118 Bilder, Risse | dünne Strukturen, klein |

Von der Remote-Umgebung **nicht erreichbar**, lokal aber sehr empfehlenswert (Loader ergänzen oder
`load_folder` nutzen):
* **KolektorSDD2** (vicos.si) – 3335 Bilder, 356 mit kleinen Defekten: *am nächsten am Zielszenario*.
* **MVTec AD** (mvtec.com) – Kategorien mit Kratzern (metal_nut, screw, tile, wood, leather …).
* **MSD** Mobile-Phone-Screen (OneDrive-Link im GitHub-Repo jianzhang96/MSD) – Kratzer auf
  spiegelnden Displays.
* **VisA**, **Real-IAD** (Hugging Face).

Ein neuer Loader = eine Funktion in `autolabel/data.py`, die `Sample`-Objekte liefert, plus Eintrag
in `LOADERS` / `DEFAULT_ROOTS` und in `DATASET_KW` (`scripts/run_benchmark.py`).

## 7. Lohnende nächste Forschungsschritte

1. **Bessere Features fürs Normalitätsmodell**: lokal ist Hugging Face erreichbar → DINOv2/v3-Features
   statt WideResNet50 (`anomaly.py`, `create_encoder`). Vermutlich der größte Hebel für das
   „übersehene Defekte“-Signal.
2. **Anomaliekarte als 4. Eingangskanal** des U-Nets (`model.UNet(in_chans=4)` ist vorbereitet;
   Synthese-Crops brauchen dann eine neu berechnete Karte).
3. **Aktiver Lernkreislauf**: Top-Risiko-Bilder aus `review_queue.csv` labeln → zu L hinzufügen →
   neu laufen lassen; Gewinn pro gelabeltem Bild gegen Zufallsauswahl messen.
4. **Mehrklassen-Labels**: `classify.py` (Typ je Defektkomponente) auf MT evaluieren
   (`scripts/eval_types.py`), ggf. Multi-Label-Kopf im U-Net.
5. **Größere Encoder / mehr Iterationen**, sobald MPS-Speed bekannt ist (`--encoder resnet34`,
   `--iters 1500`, `--crop 256`).

## 8. Code-Landkarte (wo was zu ändern ist)

* Hyperparameter Training: `TrainConfig` in `autolabel/train.py`
  (iters, crop, p_synth, p_procedural, rot90, boundary_ignore, …).
* Pipeline-Parameter: `AutoLabelConfig` in `autolabel/pipeline.py`
  (k_folds, rounds, alpha, delta, teacher, bg_clean_frac, bg_ignore_quantile, ignore_ratio, …).
* CLI-Flags von `run_benchmark.py` decken die wichtigsten Felder ab; weitere einfach ergänzen.
* Ablauf einer Runde: `AutoLabeler.run` → je Fold `train_member` → `predict_image` auf Fold/U/T →
  `IsotonicCalibrator`, `select_thresholds`, `conformal_threshold` → Masken → nächste Runde mit
  `_pseudo_labels` (Teacher k für Student k).
* Auswertung: `autolabel/evaluate.py`; neue Auswertungen lassen sich per `scripts/reevaluate.py` auf
  alte Läufe anwenden, ohne neu zu trainieren.

## 9. Mit Claude Code lokal weitermachen

`CLAUDE.md` im Repo-Root verweist auf dieses Dokument. Ein guter Einstiegsprompt:
„Lies docs/HANDOFF.md. Führe Abschnitt 3 (Validierung) aus und behebe Fehler, dann starte den
Experimentplan aus Abschnitt 4 und schreibe docs/RESULTS.md.“
