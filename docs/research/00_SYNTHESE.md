# 00 – Synthese: CertAL, zertifiziertes Auto-Labeling für industrielle Defektsegmentierung

Stand 18.10.2026. Zusammenführung der Recherchen [01](01_foundation_models_anomaly.md)–
[05](05_human_in_the_loop_task_adaptation.md) und des Theoriebeitrags [06](06_completeness_theory.md).
Der Rahmen ist **aufgabenunabhängig**: verschiedene Produkte, Materialien und Kameras, Defekte klein
bis groß, eine oder mehrere Klassen, 5–30 Masken pro Aufgabe, Tausende ungelabelte Bilder.

> Alle Gewinnerwartungen sind Hypothesen aus der Literatur oder aus Theorie. Auf unseren Daten ist
> davon noch nichts gemessen. Zahlen aus den Berichten tragen dort ihre Quellenmarker
> ([R] README/Ergebnisdatei, [A] Abstract/Snippet, unverified).

---

## 1. Die Kernaussage in einem Absatz

Kein Modell wird aus 5–30 Masken auf einer fremden Industrieoberfläche *beweisbar* gute Labels
erzeugen. Die Literatur zeigt das deutlich: Selbst die besten unüberwachten DINOv3-Verfahren erreichen
auf MVTec AD 2 nur ≈ 57 % SegF1 [A], Few-Shot-Foundation-Modelle liegen auf Defekten bei 28–48 mIoU [A].
Der Ausweg ist eine **Arbeitsteilung**: *Modelle* erzeugen Labels **und Vorhersagen über ihre eigenen
Fehler**. Ein **randomisiertes, sequentielles Mensch-Audit** mit bekannten Auswahlwahrscheinlichkeiten
liefert das **Zertifikat**, und es korrigiert dabei, was es prüft. Dann gilt:
**Genauigkeit kauft Effizienz, das Stichprobendesign kauft Gültigkeit.** Bessere Modelle senken nur
die menschlichen Minuten bis zum Zertifikat. Schlechte Modelle machen das Zertifikat teurer, aber nie
falsch. Daraus folgt die Zielgröße, an der sich jede Komponente messen lassen muss:

> **Nordstern-Metrik: Menschminuten bis zum Zertifikat.** Beispielziel: „Anteil Bilder mit
> übersehenem Defekt ≤ 2 % *und* Defekt-Recall ≥ 90 %, jeweils mit 95 % Konfidenz, für den
> ausgelieferten (korrigierten) Pool.“

---

## 2. Gesicherte Befunde (Konsens über die Berichte)

| # | Befund | Quelle |
|---|---|---|
| B1 | **Der Encoder ist der stärkste Hebel**, nicht der Lernalgorithmus. UniMatch V2: 75,2 → 86,3 mIoU nur durch DINOv2-B statt ResNet-101 [R]. DINOv3 verbessert vor allem die Pixelebene (Dinomaly DINOv3-L: P-AP 74,2 auf MVTec AD [R]). DINOv3 ist aber nicht überall besser: Auf MPDD ist es schlechter als RADIO und DINOv2 [R], auf Röntgen schlechter als ImageNet [A] | 01, 03 |
| B2 | **Wenige echte Masken bringen viel**: KSDD2 Pixel-AP 45,8 (unüberwacht) → 59,1 (16 Masken) → 81,3 (alle) [R]. Unüberwachte Normalität allein reicht auf harten Daten nicht | 01 |
| B3 | **VLMs, SegGPT, PerSAM und SAM gehören nicht in den Labelpfad.** CLIP-Zero-Shot P-AP ≈ 41 [R], MLLMs ≤ 74,9 % Accuracy auf MMAD [A], SAM3 auf Rissen 56,5 IoU [R]. SAM ist nützlich als Klick-Werkzeug und als gegateter Refiner für kompakte Defekte mit klaren Kanten | 01, 03, 05 |
| B4 | **P-AUROC ist für Labelqualität wertlos** (CandyCane 97,3 bei P-AP 1,8 [R]). Maßgeblich sind SegF1 bei fester Schwelle, Recall auf Defektebene je Größenklasse, AU-PRO@0,05 und Fehlbildrate | 01 |
| B5 | **Mutual-Scoring über den kontaminierten Pool** (MuSc, unser kNN mit Ausschluss des eigenen Bildes) scheitert an *konsistenten Anomalien*: Wiederkehrende Defekttypen erklären sich gegenseitig. CoDeGraph +9,2 P-AP [R]. Ein greedy Coreset bevorzugt Defektpatches | 01 |
| B6 | **Generative Defektsynthese ist für unser Setting unbewiesen.** Diffusionsgeneratoren wurden nie gegen eine starke Copy-Paste-Baseline mit echten Defekten verglichen. Der Mask-Bild-Abgleich ist die Schwachstelle (AnomalyDiffusion ohne Alignment: P-AP 81,4 → 64,9 [P]). Nicht-generative Synthese ist der beste Aufwand-Nutzen-Punkt | 02 |
| B7 | **Aktives Lernen schlägt Zufall bei Kleinstbudget oft nicht.** Typikalität bzw. Abdeckung ist anfangs besser als Unsicherheit. Prüfen kostet ≈ 3 s/Bild, Korrigieren ein Vielfaches, Zeichnen ≈ 79 s/Objekt [A] | 05 |
| B8 | **Hyperparameter-Tuning auf 5–30 Labels ist überwiegend Rauschen** (Perez et al. 2021). Besser: nnU-Net-artige Regeln aus einem Daten-Fingerprint, dazu wenige Kandidaten | 05 |
| B9 | **Modellbasierte Garantien aus L sind mit ≈ 10 Bildern praktisch nicht zu haben.** Die marginale Split-Conformal-Abdeckung ist bedingt Beta(11,1)-verteilt: P(Recall < 0,9) = 0,31. Bildweise CRC ist unerfüllbar. **Designbasierte Garantien** (randomisiertes Audit, Betting-Konfidenzsequenzen) gelten ohne Austauschbarkeit und ohne Annahmen über die Abhängigkeit zwischen Komponenten | 04 |
| B10 | **Übersehene Defekte schätzt keine etablierte Methode direkt.** Das ist die zentrale Forschungslücke, von drei Berichten unabhängig benannt | 01, 02, 03 |

---

## 3. Was am bestehenden Code falsch oder schwach ist

Sortiert nach Schwere. Details stehen in 04 §4.8 und 02 §9.

1. **Die Recall-Garantie gilt für U nicht** (04):
   * Schwellen aus Einzel-Member-OOF werden auf den Ensemble-Mittelwert angewandt. Wegen
     max-of-mean ≤ mean-of-max ist das antikonservativ: Im Toy-Modell liegt die Missrate bei 0,12–0,32
     statt 0,1.
   * U-Vorhersagen sind *in-sample*. Runde 0 nutzt 70 % von U als Negativ-Hintergründe, Runde 1
     trainiert auf allen U-Pseudo-Labels. Genau die übersehenen Defekte werden so als „normal“ gelernt.
   * Komponenten werden gepoolt, obwohl sie innerhalb eines Bildes abhängig sind.
   * `certified_recall` wird bei einem per F1-Grid-Search gewählten `t_seed` ausgegeben. Das ist nicht
     zertifiziert.
2. **Die Audit-Intervalle (t-Quantil plus Delta-Methode) unterschreiten die Sollabdeckung bei seltenen
   Defekten** (04 S1/S2): audit-only liegt teils bei 0,23–0,69 statt 0,95, PPI mit λ = 1 bei 0,83–0,88.
   Getuntes λ unterdeckt, weil es auf derselben Stichprobe getunt wird.
3. **Unsichtbare Positive in der Synthese** (02): Wird ein additives Residuum geclippt oder mit kleinem
   Gain eingefügt, ist der Defekt unsichtbar, bleibt aber als positiv gelabelt. Das trainiert
   Fehlalarme. Außerdem fehlen großflächige Defekte, multiplikative Defekte (Verfärbungen) und eine
   kontextpassende Platzierung.
4. **Review und Audit sind getrennt**, und die Qualität des *korrigierten* Pools wird nicht berichtet (04).
5. **Normalitätsmodell:** WRN50 mit 8-px-Raster ist der schwächste Baustein. Es gibt keinen Schutz gegen
   konsistente Anomalien und keinen Gruppenausschluss (01).

---

## 4. Der Vorschlag: CertAL

CertAL besteht aus fünf Schichten. Neu gegenüber der Literatur ist vor allem das Zusammenspiel
(Abschnitt 5). Die einzelnen Bausteine sind teils etabliert.

```
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │ S5  Aufgabenanpassung: Fingerprint-Regeln, ≤8 Kandidaten, 1-SE; ein Nutzerknopf FN:FP │
 └──────────────────────────────────────────────────────────────────────────────────────┘
 ┌─────────────────────────────┐   ┌──────────────────────────────┐
 │ S1 Labeling-Engine          │   │ S2 Kontrolldetektoren        │   unabhängig by design:
 │ doppelt cross-fitted (L∪U), │   │ DINO-kNN (Kacheln, Gruppen-  │   keine gemeinsamen
 │ heterogene Member,          │   │ ausschluss, Burnout), opt.   │   Eingaben außer dem Bild
 │ Synthese v2, Self-Training  │   │ Zero-Shot-VFM                │
 └──────────────┬──────────────┘   └───────────────┬──────────────┘
                └───────────────┬──────────────────┘
          ┌─────────────────────▼──────────────────────────────────────────┐
          │ S3 Evidenz & Vollständigkeit: Regions-Stacker q_r,                 │
          │ Injection-Recovery, Capture-Recapture → Prädiktoren f_i, σ_i       │
          └─────────────────────┬──────────────────────────────────────────┘
          ┌─────────────────────▼──────────────────────────────────────────┐
          │ S4 Measure-and-Repair: Kacheln als Einheit, π_i geschlossen        │
          │ (Review-Stratum + Stichprobe), sequentielles Betting-Audit mit     │
          │ prädiktablem PPI + Fieller, designbasierte Schwellenwahl,          │
          │ jederzeit stoppbar → Zertifikat für den korrigierten Pool          │
          └────────────────────────────────────────────────────────────────┘
```

### S1 – Labeling-Engine (Labels plus ehrliche Scores)

* **Doppeltes Cross-Fitting über L ∪ U** (04-N2). Die Folds umfassen L *und* U. Die Member von Fold k
  berühren L_k ∪ U_k in keiner Rolle: nicht als Label, nicht als Hintergrund, nicht als Pseudo-Label.
  Sie sagen L_k und U_k vorher. Damit stammen Vorhersagen auf L und auf U vom *selben* Mechanismus,
  Self-Training lernt nie die eigenen Fehler auf den zu labelnden Bildern, und alle Schwellen und
  Kalibrierungen übertragen sich ehrlich. Das schließt die zwei größten Garantie-Lecks.
* **Heterogene Member pro Fold** (01-R1, 03-R1): das bestehende ImageNet-resnet18d-U-Net mit
  Full-Res-Stem, dazu ein DINOv3- bzw. DINOv2-Member (ConvNeXt-T/S oder ViT-S/B, fine-tuned bzw. LoRA,
  mit Full-Res-Pfad). Die Gewichtung erfolgt per OOF, damit ein Member abgewählt werden kann, wo
  DINOv3 schwach ist (Graubild, Röntgen).
* **Synthese v2** (02-E2–E5), foldspezifisch:
  * Labels aus dem *tatsächlich erzeugten* Signal (Differenzmaske, Ignore-Band, kein unsichtbares
    Positiv);
  * Null-Synthese-Negative gegen Artefakt-Shortcuts;
  * Gain-Verteilung abgestimmt auf die Detektierbarkeit realer Defekte;
  * kontextpassende Platzierung;
  * Mehrband- bzw. multiplikativer Transfer für große Defekte und Verfärbungen;
  * klassenbalancierte Bänke.
* **Masken:** Hauptlabel aus dem Fold-Ensemble-Mittel, kalibriert auf denselben Mechanismus (OOF des
  Fold-Ensembles). Die recall-sichere Variante entsteht per Union-Regel (pixelweises Maximum, CV+-Argument, 04-Prop. 1).
* **Trainingsrezept** (03-R2/R3): Soft-Targets plus Ignore plus Konfidenzgewichte, Dice Semimetric
  Loss, SVLS am Rand, komponenten-normierter Term (blob) und Skeleton-Recall für langgestreckte
  Klassen. Ob sie zugeschaltet werden, entscheidet die Fingerprint-Regel, nicht ein Mensch.

### S2 – Kontrolldetektoren (unabhängig vom Labeler)

* **DINO-kNN-Normalität** (01-R1/R2): DINOv3 (Fallback DINOv2), mehrschichtig, native Auflösung mit
  Kacheln. Kein greedy Coreset; stattdessen Zufalls- oder SoftPatch-Subsampling. Ausschluss auf
  **Gruppenebene** (Sequenz, Bauteil). Als zweite Statistik ein k-Profil- bzw. Burnout-Score gegen
  konsistente Anomalien. Optional eine label-geführte Bereinigung der Bank (01-11.1).
* **Optional** ein Zero-Shot-VFM (AnomalyVFM) als drittes Signal, das ohne Training über `torch.hub`
  läuft. Die Lizenz ist zu prüfen.
* **Designregel:** Kontrolldetektoren werden **nie** in den Labeler eingespeist, also kein
  Anomaliekanal als U-Net-Input. Ihr Wert liegt in *unkorrelierten Fehlern* (06 §6–7). Ein
  schwächerer, aber unabhängiger Detektor kann für die Messung wertvoller sein als ein stärkerer,
  korrelierter.

### S3 – Evidenz und Vollständigkeit

* **Regions-Stacker** (01-R3, 03-R4): Für jede Kandidatenregion aus Labeler und Kontrolldetektoren
  wird `q_r = P(echter Defekt)` geschätzt. Merkmale: Konfidenz, Ensemble-Uneinigkeit, Normalitäts-
  p-Wert, Fläche, Kontrast, Burnout. Gefittet auf OOF-Kandidaten, im Audit *prädiktabel*
  nachkalibriert.
* **Injection-Recovery** (02-N1): Synthetische Defekte mit bekannter Lage werden eingefügt. Daraus
  entstehen Sensitivitätskarten pro Bild bzw. Kachel und die vollständigen Fanggeschichten aller
  Detektoren, einschließlich der Zelle „von allen übersehen“. Die Injektionsbank ist cross-fitted
  bzw. reserviert, damit kein Synthetik-Leck entsteht.
* **Vollständigkeitsschätzer** (06): Horvitz–Thompson mit Injektion (Annahme T), Huggins–Alho auf
  echten Fanggeschichten (Annahme CI) und ein Hybrid. Dazu ein **GT-freier Test der
  Transportannahme** auf den beobachtbaren Zellen; die Übereinstimmung der drei Schätzer dient als
  Triangulation.
* **Ausgabe pro Einheit:** f_i^M (erwartete übersehene Defekte), f_i^D (erwartete Defekte), f_i^F
  (erwartete Fehlalarme), f_i^b = P(übersehener Defekt), σ_i. Das ist der Input für S4. **Fehler in
  S3 kosten nur Effizienz, nie Gültigkeit.**

### S4 – Measure-and-Repair (die zertifizierende Säule)

* **Einheit = Kachel** (04-N6), etwa 256–512 px mit fester Zuordnungsregel. D_max pro Einheit sinkt
  auf ≈ 1–2, die Betting-Schranken für Recall werden um Größenordnungen schärfer, und ein Mensch prüft
  eine Kachel schneller als ein ganzes Bild.
* **Inklusionswahrscheinlichkeiten in geschlossener Form** (04-Prop. 3):
  π_i* = min{1, σ_i·√(κ/(ν − m_i))}. Einheiten mit m_i + κσ_i² ≥ ν bilden das Review-Stratum
  (π = 1). Die übrigen werden nach Neyman bzw. Active Inference gezogen, mit Uniform-Floor
  (≈ 30 % des Ziehungsbudgets) gegen unsichtbare Misses. Jede geprüfte Einheit wird **korrigiert
  und gemessen**. Die Prüfung ist **verify-first**: OK / Defekt fehlt / Fehlalarm / Grenze falsch,
  bei Bedarf mit Korrektur (05).
* **Sequentielles Audit mit Betting-Konfidenzsequenzen** (04-Satz 1, Prop. 4):
  * PPI-Kontrollvariate mit **prädiktablem** λ_t und prädiktabel nachkalibrierten Prädiktoren;
  * IPW mit adaptivem q_t;
  * Fieller-Inversion für Verhältnis-Metriken (Recall, Precision, Dice);
  * exakte hypergeometrische Schranke für die Fehlbildrate ζ.
* **Designbasierte Schwellenwahl** (04-Satz 2, N3): Pixel- und Review-Schwelle werden per
  Fixed-Sequence-Testing auf dem Audit gewählt. Das ist PAC-Labels (Candès, Ilyas, Zrnic, ICML 2026)
  erweitert auf Segmentierung.
* **Stoppregeln, jederzeit gültig:** Erfolg, Futility, Budget. Dazu ein **Vorab-Machbarkeitscheck**
  (Heuhaufen-Schranke, 04-N5): Ist Defekt-Recall mit dem Budget überhaupt zertifizierbar, oder nur ζ?
* **Rückkopplungsregel:** Audit-Ground-Truth fließt nicht in die zu zertifizierenden Labels zurück. Ins
  Produktivmodell darf sie, abzüglich eines reservierten Testanteils.

### S5 – Aufgabenanpassung ohne Experten

* **Fingerprint-Regeln** aus den Masken (05 §5.2):
  * Defektbreite → Full-Res-Pfad und Toleranz;
  * Ausdehnung → Crop-Größe bzw. Ganzbild-Zweig;
  * Elongation → Skeleton-Term;
  * Kontrast → Gain-Bereich;
  * Zahl der Defekte → K und α;
  * dazu **ein** Nutzerknopf: Kostenverhältnis FN:FP → F_β.
* **Höchstens 4–8 Kandidaten**, ausgewählt per wiederholtem Cross-Fitting mit der 1-SE-Regel; bei
  Gleichstand wird ensembled statt gewählt. Die Regeln werden **Leave-One-Dataset-Out** validiert
  (05-Idee 3).
* **Mehrklassen:** hierarchische Pseudo-Labels (Defekt sicher, Klasse unsicher → ignore im Klassenkopf),
  Typzuordnung mit konformen Mengen (vorhanden, `classify.py`) und Entdeckung neuer Typen durch
  Clustering „unerklärter“ Kandidaten (01-R10, 05-Idee 5).
* **Billiges Feedback verwerten:** „Defekt fehlt“-Flags als MIL-Constraint, Fehlalarme als
  Hard-Negatives (05-Idee 2).

---

## 5. Was daran neu ist (und was nicht)

**Übernommen bzw. etabliert:** DINO-Features, kNN-Normalität, Copy-Paste-Synthese, Self-Training,
Cross-Fitting, PPI, Betting-Konfidenzsequenzen, PAC-Labels, nnU-Net-Regeln, Capture–Recapture an sich.

**Neu nach unserer Recherche** (die Recherche war nicht erschöpfend; Volltexte vieler Paper waren
gesperrt):

| # | Beitrag | Nächstliegende Vorarbeit | Unterschied |
|---|---|---|---|
| C1 | **Vollständigkeitsschätzung für Auto-Labels**: Injection beobachtet die Zelle „von allen übersehen“, echte Fanggeschichten testen die Transportannahme ohne GT, Panel nach Unabhängigkeit gebaut (06) | Capture–Recapture (Ökologie, Software-Inspektion), Injection-Recovery (Astronomie), MetaSeg (nur FP) | Erste Übertragung auf Pixel-Labeling. Testbare Transportannahme. Unabhängigkeit wird zum Konstruktionsziel |
| C2 | **Measure-and-Repair-Design** mit geschlossener Form für π: eine menschliche Handlung korrigiert und misst zugleich, zertifiziert wird der *korrigierte* Pool (04-N1, 05-Idee 1) | Active Inference (nur Messung), PAC-Labels (sequentiell, Klassifikation) | Gemeinsame Optimierung, Lemma 0 (Monotonie) für Segmentierung |
| C3 | **Segmentierungs-PAC-Labels**: Bild bzw. Kachel als Cluster, Verhältnis-Metriken über Fieller-Betting, designbasierte Wahl von Pixel- und Review-Schwelle (04-Satz 2, N3, N4) | PAC-Labels, LTT, CRC | Verhältnisse ohne Delta-Methode, prädiktables λ löst das Unterdeckungsproblem |
| C4 | **Doppeltes Cross-Fitting über L ∪ U** mit fold-konsistentem Self-Training (04-N2) | CV+, Jackknife+, Cross-Prediction-PPI | Schließt die Self-Training-Leckage, ist mit Synthese und Pseudo-Labels kombiniert |
| C5 | **Kacheln als Stichprobeneinheit** für scharfe Recall-Schranken (04-N6) | Cluster-Sampling (Survey-Statistik) | Gezielt gegen das D_max-Problem der Betting-Schranken |
| C6 | **Synthese-Labelvertrag**: Differenzmasken, Null-Negative, Detektierbarkeits-Matching (02-N2/N3/N5) | AnomalyDiffusion-Alignment, DRAEM | Label aus dem erzeugten Signal statt aus der Bedingung |
| C7 | **Nordstern „Menschminuten bis zum Zertifikat“** als Benchmark-Metrik für Auto-Labeling | Annotationskosten-Studien, Active Testing | Verbindet Labelqualität, Messbarkeit und Kosten in einer Zahl |

Die eigentliche These, die die Benchmarks prüfen sollen:

> **H\*:** Bei gleichem Qualitätsziel braucht CertAL deutlich weniger Menschminuten bis zum Zertifikat
> als (a) „bestes Modell + Top-k-Review + separates Zufallsaudit“ und (b) „bestes Modell + Vollkontrolle“.
> Die Abdeckung des Zertifikats hält in *allen* Szenarien, auch wenn Modellannahmen bewusst verletzt
> werden.

---

## 6. Roadmap für die lokale Weiterarbeit

Reihenfolge nach Nutzen pro Aufwand. Phase 0 braucht **keine GPU** und kann sofort starten.

| Phase | Inhalt | Rechenbedarf | Ergebnis bzw. Entscheidung |
|---|---|---|---|
| **0 – Statistikkern v2** | `ppi.py` v2: Betting-CS (WSR, Gitter-Mischung), Fieller-Inversion, exakte hypergeometrische Schranke, IPW mit prädiktablem λ, Measure-and-Repair-π, designbasierte Schwellenwahl, Kachel-Einheiten. Ehrliche Berichte: bildweise CRC, Beta-Abdeckung, „keine Garantie möglich“. Simulationssuite nach 04 §4.10 (inkl. adversarialer Szenarien) | CPU, Minuten | Abdeckung ≥ 1−δ in allen Szenarien → Statistikkern freigegeben |
| **1 – Baseline-Benchmarks** | HANDOFF-Plan (MT, BSData, DeepCrack, CFD), dazu Loader für KSDD2 und MVTec AD 2 (`test_public`); Metriken nach Größenklassen, SegF1, ζ | M4, ≈ 1 Tag | Referenzzahlen; Prüfung des Schwellentransfers (U vs. U-Oracle) |
| **2 – Kontrolldetektoren und Vollständigkeit** | DINO-kNN (Kacheln, Gruppen, Burnout), Regions-Stacker, Injection-Recovery, Schätzer aus 06 | M4, 1–2 Tage | H1–H5 aus 06; reduziert die Vollständigkeitsschätzung den FN-Schätzfehler um ≥ 25 %? |
| **3 – Labeling-Engine v2** | doppeltes Cross-Fitting, DINOv3-Member, Synthese v2, Trainingsrezept | M4/GPU, mehrere Tage | Recall pro Größenklasse; Synthese-Ablationen nach 02 §9.1 |
| **4 – Mensch-in-der-Schleife-Simulation** | GT als simulierter Mensch, Kostenmodell aus 05 §3.1; CertAL vs. Baselines (a)/(b); Kacheln vs. Bilder; verify-first vs. Vollkorrektur | CPU (nutzt gespeicherte Läufe) | **Hauptergebnis: H\***, Kurven „Minuten → Zertifikat“ |
| **5 – Aufgabenanpassung** | Fingerprint-Regeln mit LODO-Validierung; optional Cross-Task-Defekt-Prior (Vortraining auf den übrigen Datensätzen) | GPU sinnvoll | ≥ 90 % der Oracle-Konfiguration ohne Tuning? |

Phase 0 und Phase 4 sind CPU-leicht. Sie lassen sich hier in der Cloud-Session umsetzen und gegen die
Simulationen absichern, während lokal Phase 1–3 laufen.

---

## 7. Risiken und offene Punkte

* **Mensch = Goldstandard** ist eine Annahme. Ein kleiner Doppel-Annotationsanteil schätzt die
  Reviewer-Fehlerrate (04 §6). Hinzu kommt Automation Bias beim Korrigieren (05).
* **Heuhaufen-Grenze:** Hohen Defekt-Recall bei sehr seltenen Defekten zu zertifizieren, kann
  hunderte Audits kosten (2000 saubere Bilder, ≤ 7 Fehlbilder ⇒ ≈ 640). Hebel sind Kacheln, ein starkes
  Kontrollsignal als Proposal-Verteilung und die Wahl von ζ als Primärmetrik.
* **Lizenzen:** DINOv3, SAM 3, AnomalyVFM/RADIO und die Datensätze (MVTec CC BY-NC-SA) müssen vor
  produktiver Nutzung geprüft werden. DINOv2 (Apache 2.0) ist der Fallback.
* **Rechenkosten** des doppelten Cross-Fittings und der Injektionen sind auf dem M4 noch nicht gemessen.
* **Quellenlage:** Viele Zahlen stammen aus Abstracts und READMEs, weil die Volltexte in dieser
  Umgebung gesperrt waren. Vor Veröffentlichung sind sie gegen die PDFs zu prüfen.
* **Zertifikat ≠ Produktivqualität:** Eine zertifizierte Labelfehlerrate begrenzt nur die Verzerrung
  der Evaluation auf Pseudo-Labels. Das Produktivmodell wird auf einem reservierten Audit-Anteil
  designbasiert gemessen (04 §2.5).
