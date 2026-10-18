# 05 – Human-in-the-Loop-Effizienz und schnelle Anpassung an neue Aufgaben

Stand: 18.10.2026. Literaturreview für `auto-label-tool` (pixelgenaues Auto-Labeling industrieller Defekte
aus 5–30 Masken pro Aufgabe). Bereits implementierte Bausteine (Cross-Fitting-Ensemble, Normalitätsmodell,
Residual-Copy-Paste, fold-konsistentes Self-Training, Kalibrierung, Hysterese, konforme Recall-Schwellen,
Selbsteinschätzung + Review-Queue, Zufallsaudit + PPI, Few-Shot-Typklassifikator, CLI) werden hier
vorausgesetzt und nicht neu vorgeschlagen.

> **Zur Recherche.** Recherchiert per Websuche (Okt. 2026). Volltexte von arXiv/CVF/PMLR konnten aus der
> Umgebung nicht direkt abgerufen werden; Zahlen stammen aus Abstracts bzw. Auszügen, die die Suche
> lieferte, und aus GitHub-READMEs. Gegen Ende war das gemeinsame Suchbudget der parallelen Reviewer
> erschöpft; einige Punkte (v. a. Tesla-Data-Engine, Snorkel, Weighted Conformal, clDice, Bearman et al.)
> stammen aus Vorwissen und sind mit **(unverified)** markiert. Keine Zahl in diesem Dokument wurde
> erfunden; eigene Annahmen sind als *Annahme* gekennzeichnet.

---

## 0. Executive Summary

1. **Aktives Lernen (AL) schlägt Random bei kleinem Budget oft nicht zuverlässig.** Bei sehr kleinen Budgets
   sind *typische/abdeckende* Stichproben (TypiClust, ProbCover, MaxHerding) besser als unsicherheitsbasierte;
   Unsicherheit wird erst später nützlich. Benchmarks (Mittal et al. GCPR 2023/IJCV 2025; Lüth et al.
   NeurIPS 2023) zeigen: welche AL-Methode gewinnt, hängt von Redundanz, Semi-Supervision und Budget ab.
   Konsequenz: AL immer gegen Random benchmarken, Auswahl **gemischt** (Risiko + Abdeckung + Zufall).
2. **Für seltene Defekte ist der größte AL-Hebel nicht die Akquisitionsfunktion, sondern das Finden der
   Defekte überhaupt** – also Vorauswahl über das Normalitätsmodell plus Abdeckung in Foundation-Feature-Räumen
   (DINOv2/v3). Imbalance-Studien (Combes et al. 2026; Cai et al., Halbleiter-XRM) bestätigen, dass
   „rareness-aware“-Auswahl und Entropie/Core-Set unter starker Imbalance funktionieren.
3. **Korrigieren ist billiger als Zeichnen, Verifizieren billiger als Korrigieren.** Belegte Größenordnungen:
   Ja/Nein-Feedback ≈ 2,7 s/Bild (One-bit Supervision), SAM-gestützte Maske 34 → 14 s, Polygonmaske COCO
   ≈ 79 s/Instanz; Verify-only-Training spart 6–9× Zeit bei Boxen (Papadopoulos et al. 2016). Die Ersparnis
   ist aber stark aufgabenabhängig (21,5 % bis ~85 % in klinischen Korrekturstudien).
4. **Die wichtigste Protokollentscheidung ist statistisch, nicht algorithmisch:** Jede menschliche
   Auswahl muss mit **bekannter Inklusionswahrscheinlichkeit** protokolliert werden. Dann zählen Review-Bilder
   als *vollständig beobachtetes Stratum* für die Qualitätsschätzung (statt verloren zu sein), und
   PPI/Conformal bleiben gültig (Active Statistical Inference, Zrnic & Candès 2024; StratPPI, Fisch et al. 2024).
5. **Billiges Bild-Level-Feedback („OK / Defekt fehlt / Fehlalarm“) ist ungenutztes Potenzial.** Es lässt sich
   als Mixed Supervision (Božič et al. 2021, KolektorSDD2) bzw. als Ein-Bit-Supervision in den
   Retraining-Loss einbauen und liefert zugleich Kalibrierdaten auf Bildebene.
6. **SAM/SAM2/SAM3 sind gute *Werkzeuge* (Klick-Refinement), aber schlechte *Auto-Labeler* für feine Defekte.**
   Mehrere Arbeiten berichten, dass SAM ohne Anpassung auf Industriedefekten/Rissen schwach ist; im
   Few-Shot-Defektbenchmark von Liu et al. (2025) erreicht SAM2 im Video-Modus 47,9 mIoU. Realistische Klicks
   verschlechtern interaktive Methoden zusätzlich (RClicks, NeurIPS 2024).
7. **Hyperparameter-Tuning auf 5–30 Labels ist großteils Rauschen** (Perez et al. NeurIPS 2021: Auswahl per CV
   kaum besser als zufällig). Empfehlung: nnU-Net-Philosophie – *feste Defaults + Regeln aus einem
   Daten-Fingerprint (Defektgröße/-breite/-elongation) + sehr wenige empirische Entscheidungen*, ausgewählt per
   wiederholtem OOF mit 1-SE-Regel.
8. **Cross-Task-Vortraining ist vielversprechend, aber unbelegt für supervised Defektsegmentierung.** Es gibt
   aufgabenübergreifende Anomaliemodelle (AnomalyCLIP, Dinomaly/Dinomaly2, AnomalyDINO) und verfeinerte
   Mehrklassen-Masken (Defect Spectrum), aber kein etabliertes „Defekt-Foundation-Segmentierungsmodell“.
   MVTec AD 2 zeigt große Lücken (max. 58,7 % AU-PRO). → eigenes Forschungsexperiment.
9. **Integration:** Export als COCO-RLE, CVAT-XML (Masken-RLE relativ zur BBox), Label-Studio-Predictions
   (eigenes bit-gepacktes RGBA-RLE, *nicht* COCO-RLE!) und ein FiftyOne-Weg; Rückimport inkl.
   Zeitstempeln, Status und Inklusionswahrscheinlichkeit.
10. **Vorgeschlagenes Protokoll:** ≈ 90–120 Menschminuten für eine neue Aufgabe bis zu einem Label-Pool mit
    gemessenen Konfidenzintervallen; danach Runden à 30 min mit klarer Stopp-Regel (CI-Obergrenze der
    Miss-Rate ≤ Ziel *oder* Grenznutzen pro 10 min unter Schwelle).

---

## 1. Was an unserem Setting besonders ist

* **Extrem kleines L, seltener Vordergrund.** Eine zufällige Bildstichprobe aus U enthält bei 5–30 %
  Defektrate wenige Defekte; AL-Methoden, die für balancierte Klassifikation entwickelt wurden, übertragen
  sich schlecht (Pitfall „limited data distributions“, Lüth et al. 2023).
* **Mehrere Verwendungen derselben Menschminute:** (a) Pool-Qualität direkt verbessern (riskante Labels
  korrigieren), (b) das Modell verbessern (informative Bilder labeln), (c) Qualität messen (unverzerrtes Audit).
* **Unsere Garantien hängen an Austauschbarkeit.** Konforme Schwellen und PPI setzen voraus, dass die
  gelabelten/auditierten Bilder eine (gewichtete) Zufallsstichprobe sind. Jede aktive Auswahl, die nicht
  mit Wahrscheinlichkeiten protokolliert wird, zerstört diese Basis. Das ist der rote Faden dieses Berichts.

---

## 2. Aktives Lernen bei winzigem Budget

### 2.1 Befundlage: AL vs. Random

| Arbeit | Idee | Evidenz | Relevanz für uns |
|---|---|---|---|
| Mittal et al., *Parting with Illusions about Deep AL* (arXiv 1912.05361, 2019) | Realistische Evaluation | AL-Methoden profitieren stark von Semi-Supervision, sind dann aber kaum besser als Random; Gewinne mit moderner Augmentation klein | Wir haben starke Semi-Supervision (Self-Training, Synthese) → AL-Gewinne werden kleiner sein als in Papers |
| Mittal, Niemeijer, Schäfer, Brox, *Best Practices in AL for Semantic Segmentation* (GCPR 2023, Best Paper; arXiv 2302.04075; erweitert IJCV 2025) | Benchmark Segmentierungs-AL | Redundanz der Daten, SSL-Integration und Budget entscheiden. PASCAL-VOC bei 10 % Budget: **Random + SSL am besten**; Cityscapes/A2D2: Einzelbild-Akquisition am besten | Industriebilder sind oft hochredundant (gleiches Produkt) → AL eher sinnvoll; aber gegen Random+SSL messen |
| Lüth, Bungert, Klein, Jaeger, *Navigating the Pitfalls of AL Evaluation* (NeurIPS 2023; arXiv 2301.10625) | 5 Pitfalls: zu wenige Datenverteilungen, Startbudgets, Query-Größen; Klassifikator-Konfiguration und alternative Trainingsparadigmen ignoriert | Großer Benchmark mit Praxisempfehlungen (Details nicht verifiziert) | Unser eigener Benchmark muss mehrere Startbudgets (5/10/30) und Query-Größen testen |
| Combes, Derville, Coeurjolly, *When Imbalance Comes Twice* (arXiv 2601.06209, 2026) | AL für binäre Segmentierung in Qualitätskontrolle unter Imbalance und Label-Shift | Entropie und Core-Set bleiben auch bei hoher Imbalance effizient; starker Label-Shift senkt die Effizienz messbar | Direkt unser Szenario; bestätigt: AL lohnt, aber Verteilungsverschiebung zwischen L und U beachten |

**Bewertung.** Die Literatur ist konsistent: AL-Gewinne sind real, aber kleiner und instabiler als
oft berichtet, und am kleinsten genau in unserem Regime (sehr kleines Budget + starke Semi-Supervision).
AL darf daher **nie** ohne Random-Kontrolle evaluiert werden, und die Selektion sollte einen
Zufallsanteil enthalten (auch aus statistischen Gründen, Abschnitt 4).

### 2.2 Typikalität und Abdeckung (Low-Budget-Regime)

* **TypiClust** (Hacohen, Dekel, Weinshall, ICML 2022, arXiv 2202.02794): Theorie eines
  „Phasenübergangs“ – bei niedrigem Budget typische, bei hohem Budget untypische Beispiele abfragen.
  Clustering in selbstüberwachtem Feature-Raum, je Cluster das dichteste Beispiel. Mit SSL auf CIFAR-10
  mit 10 Labels **93,2 % Accuracy, +39,4 Punkte gegenüber Random**.
* **ProbCover** (Yehuda et al., NeurIPS 2022, arXiv 2205.11320): Max-Coverage mit Kugeln fester Radius δ;
  besser als SOTA im Low-Budget-Regime, besonders zusammen mit SSL.
* **MaxHerding / Generalized Coverage** (Bae, Noh, Sutherland, ECCV 2024, arXiv 2407.12212): ProbCover ist
  **extrem empfindlich gegenüber δ**, die Heuristik versagt oft; glatte Kernel-Coverage (Kernel Herding)
  ist robuster und günstiger.
* **DCoM** (gleiche Gruppe, Code im TypiClust-Repo): pro Beispiel adaptive Radien, kombiniert Typikalität
  und Unsicherheit – Übergang zwischen Low- und High-Budget.

**Anwendbarkeit.** Diese Methoden sind für *Klassifikation* entwickelt; für uns ist die Einheit nicht das
Bild, sondern der *Defekt*. Sinnvolle Übertragung: Coverage-Auswahl **in zwei Räumen** – (i) Bild-Embedding
(DINOv2/v3 global) für Hintergrund-/Oberflächenvielfalt, (ii) Embeddings der **Kandidatenregionen**
(Patches um Komponenten mit hoher Anomalie oder Vorhersage) für Defektvielfalt. MaxHerding statt ProbCover,
weil wir keinen Radius pro Aufgabe tunen wollen. Kosten: vernachlässigbar (Feature-Extraktion ohnehin für
Normalitätsmodell nötig).

### 2.3 Cold Start

* **HaCon** (Chen et al., arXiv 2210.02442, NeurIPS-Workshop 2022; MIDL 2023): der erste AL-Batch ist oft
  *schlechter als Random*, weil er Ausreißer enthält; kontrastive Features + Pseudo-Labels für
  Label-Diversität schlagen Random deutlich.
* **COLosSAL** (MICCAI 2023): erster Cold-Start-Benchmark für 3D-Segmentierung (5 MSD-Tasks), Vergleich
  von Unsicherheits- vs. Diversitätsstrategien gegen Random.
* **Levy et al., *From Cold Start to Active Learning*** (arXiv 2601.18532, 2026): Foundation-Embeddings +
  Clustering für den Start, dann Unsicherheit + Diversität. Dice CheXmask 0,918 → 0,929, Montgomery 0,928 →
  0,950; SynthStrip leicht schlechter. Vor allem **geringere Streuung** (Std. 0,05 bei Random vs. 0,0075).
  Autoren betonen: Gewinne oft klein.
* **SAM-basiertes Cold Start** (MICCAI 2025, Paper 0877; Autoren nicht verifiziert): SAM-Zero-Shot-Masken
  und Unsicherheitskarten zur Auswahl der ersten Bilder.

**Für uns:** Der Hauptnutzen eines guten Starts ist **Varianzreduktion** – wichtig, weil ein schlechtes
L (z. B. nur die auffälligsten Defekte) sowohl das Modell als auch die konforme Schwelle verzerrt.

### 2.4 Region-/Superpixel-AL und alternative Query-Typen

* **Cai et al., *Revisiting Superpixels for AL in Semantic Segmentation with Realistic Annotation Costs***
  (CVPR 2021): mit klickbasiertem Kostenmaß schlagen Superpixel Rechtecke+Polygone auf Cityscapes/VOC;
  ein falsches Kostenmaß (Anteil gelabelter Pixel) lässt Superpixel schlecht aussehen. Klassenbalancierte
  Akquisition.
* **Adaptive Superpixel** (Kim et al., ICCV 2023), **ESA** (arXiv 2408.13491: Entity- + Superpixel-Auswahl,
  berichtet 98 % Klickkosten-Reduktion vs. pixelbasiert), **Multi-class Label Query** (Hwang et al.,
  NeurIPS 2023: „welche Klassen kommen in dieser Region vor?“ statt Maske).
* **Active Label Correction (ALC)** (Kim, Hwang, Kwak, Ok, ICML 2024, arXiv 2403.10820): Annotator
  **korrigiert Foundation-Model-Pseudo-Labels** statt Pixel neu zu klassifizieren; Akquisition mit
  Superpixel-Look-ahead; Nutzerstudie; korrigiert 2,6 Mio. fehlerhafte Pixel in PASCAL.
  **A²LC** (AAAI 2026, arXiv 2506.11599): zusätzlich automatische Korrektur; übertrifft frühere Methoden mit
  **20 % ihres Budgets**, +27,23 % bei gleichem Budget auf Cityscapes; 94–95 % der Vollsupervision.
* **Diffusion-Driven Two-Stage AL for Low-Budget Semantic Segmentation** (NeurIPS 2025; Autoren nicht
  verifiziert): Diffusions-Features, extreme Pixelbudgets, vier Benchmarks.

**Anwendbarkeit.** Superpixel-Abfragen sind für Szenen mit vielen Klassen gedacht. Bei 1–3 px breiten
Kratzern zerschneiden Superpixel die Struktur; die Einheit sollte bei uns die **Defektkomponente** sein
(„ist diese Komponente ein Defekt? fehlt hier einer?“). Die ALC-Idee „korrigieren statt labeln“ ist
dagegen direkt übertragbar – sie ist genau unser Review-Modus und liefert theoretische und empirische
Rückendeckung.

### 2.5 AL speziell für Defekt-/Anomaliedaten

* **Halbleiter-XRM** (Cai, Pahwa et al., ICIP 2022; arXiv 2507.17359): kontrastives Vortraining auf
  ungelabelten Scans + „rareness-aware“-Akquisition (Pseudo-Labels der Vorrunde schätzen Klassenverteilung,
  bevorzugen Bilder mit seltenen Klassen) → **98 % der Vollsupervision mit 4,9 % der Labels**.
* **Hybride Strategie für Porositätsdetektion** auf reflektierenden Alu-Teilen (Int. J. Adv. Manuf.
  Technol. 2025; Autoren nicht verifiziert): kalibrierte Unsicherheit + strukturierte Diversität + begrenzter
  Zufallsanteil.
* **Imbalanced Civil Infrastructure** (arXiv 2210.10586): klassische AL-Methoden verbessern Minoritätsklassen
  oft nicht; eigene Methode +32 % Minority-Recall vs. BALD.

**Konsequenz:** In allen erfolgreichen Defekt-AL-Arbeiten ist ein Mechanismus für **Seltenheit** zentral
(Seltenheits-Prior, Hybrid mit Zufall). Unser Normalitätsmodell ist der natürliche Seltenheits-Prior.

### 2.6 AL mit Foundation-Model-Features

* **Gupte et al., *Revisiting AL in the Era of Vision Foundation Models*** (TMLR 2024, arXiv 2401.14555):
  untersucht mit DINOv2/OpenCLIP Initialpool, Diversität und Repräsentativität vs. Unsicherheit im
  Low-Budget-Regime; eigene Strategie: Dropout-Unsicherheit + Diversität; auch biomedizinische OOD-Daten.
* **DINOv3** (Meta, Backbones seit 14.08.2025 auf HF; timm ≥ 1.0.20; ViT-S 21M bis 7B; eigene Lizenz):
  starke dichte Features – relevant sowohl für Coverage-Auswahl als auch für das Normalitätsmodell.

**Fazit 2.** Für uns sinnvoll: **gemischte Akquisition** aus (i) Risiko/Unsicherheit (unsere
Review-Signale), (ii) Coverage (MaxHerding auf DINOv3-Region-Embeddings), (iii) Zufall mit bekannter
Wahrscheinlichkeit. Gewichte nicht tunen, sondern ein festes Schema (z. B. 40/40/20) und – erst ab ≥ 3
Runden – ggf. ein Bandit über die Strategien (ALBL, Hsu & Lin, AAAI 2015).

---

## 3. Annotationseffiziente Interaktion

### 3.1 Kostenmodell (belegte Anker + Annahmen)

| Aktion | Belegte Zeit | Quelle | Übertragung auf Defekte (*Annahme*, zu messen) |
|---|---|---|---|
| Bild-Level Ja/Nein („Label OK?“) | **2,72 s/Bild**, 92,3 % Präzision (ImageNet) | Hu et al., One-bit Supervision, NeurIPS 2020 | 3–6 s bei hochauflösendem Overlay mit Zoom |
| Box zeichnen (Crowd/ILSVRC) | 42 s bzw. 35 s/Box | Papadopoulos et al. 2016 / 2017 | – |
| Extreme Clicking (4 Klicks) | **7 s/Box** (5× schneller) | Papadopoulos et al., ICCV 2017 | – |
| SAM-gestützte Maske (Klicks) | **34 → 14 s/Maske** über 6 Retrainings | Kirillov et al., Segment Anything, 2023 | 10–30 s für kompakte Defekte; für dünne Risse deutlich mehr |
| Interaktive Maske vs. Polygon | **3× schneller** als Polygon | Benenson et al., CVPR 2019 | – |
| Polygonmaske (COCO) | **≈ 79,2 s/Instanz** (22 h/1000) | COCO-Annotationsstatistik (Lin et al. 2014), via Suche | 60–300 s für einen Riss „from scratch“ |
| Videoframe mit SAM / SAM 2 im Loop | 37,8 → 7,4 s/Frame (5,1×); 8,4× vs. bisherige modellgestützte Verfahren | Ravi et al., SAM 2, 2024 | Analogie: Bildserie als „Video“ (Abschn. 5.3) |
| Dichte Szenenannotation | > 1,5 h/Bild | Cordts et al., Cityscapes, CVPR 2016 | Obergrenze, nicht vergleichbar |
| Korrektur vs. manuell (klinisch) | 17–24 min → 3–7 min (Prostata); anderswo nur −21,5 % | Palazzo et al. 2023; ESTRO-2020-Poster | Ersparnis stark abhängig von Vorlagenqualität |
| Unverändert akzeptierte Masken (SAM 2, CT/MRT) | 30 % (MRT-Femur) bzw. 53 % (CT-Lebertumor) | BMC Med. Imaging 2026 (TU Dresden) | Anteil „accept“ ist der wichtigste Kostentreiber |

**Folgerungen für das Interface.** (1) *Verify first*: ein Accept/Flag-Durchgang kostet 10–30× weniger als
eine Korrektur und filtert, wo Korrektur nötig ist. (2) Korrekturen auf **Topologie** beschränken
(fehlende/falsche Komponenten); Grenzen nur korrigieren, wenn sie außerhalb der Toleranz liegen (unsere
Metriken und Ignore-Ringe tolerieren 2 px ohnehin). (3) Zeiten **pro Aktion loggen** – die Annahmen in der
rechten Spalte müssen pro Aufgabe gemessen werden; sie sind Input für die Budgetallokation (Abschn. 4) und
Idee 5 (Abschn. 10).

### 3.2 Korrigieren statt Zeichnen

Evidenz: ALC/A²LC (Abschn. 2.4) zeigen, dass Korrektur-Queries pro Kosteneinheit mehr bringen als
Labeling-Queries; der SAM-Data-Engine-Verlauf (34 → 14 s) zeigt, dass Korrekturkosten mit jeder
Retrain-Runde fallen, weil die Vorlagen besser werden. Radiotherapie-Studien zeigen die Spannweite: von
−21,5 % bis −70…85 % Zeit. **Risiko:** *Automation Bias* – Annotatoren übernehmen fehlerhafte Vorlagen.
Gegenmittel: (i) Audit-Bilder werden *nicht* vorbefüllt gezeigt oder zumindest mit einem
„übersehene Defekte?“-Pflichtschritt geprüft; (ii) ein kleiner Anteil absichtlich „blind“ gelabelter
Bilder misst den Bias (vergleiche Korrektur- vs. Blindlabel auf denselben Bildern).

### 3.3 Klick-/Scribble-basierte interaktive Segmentierung

* **SimpleClick** (Liu et al., ICCV 2023, arXiv 2210.11006): plain ViT (MAE), **4,15 NoC@90 auf SBD**
  (−21,8 % vs. vorher bestes), out-of-domain auf medizinischen Bildern evaluiert.
* **FocalClick** (Chen et al., CVPR 2022, arXiv 2204.02574): definiert explizit **Interactive Mask
  Correction** – bestehende Masken korrigieren, ohne richtige Teile zu zerstören („Progressive Merge“);
  Benchmark DAVIS-585 mit Initialmasken. Genau unser Anwendungsfall (Korrektur von Pseudo-Labels).
  Code: ClickSEG. FocalClick-XL (arXiv 2506.14686) als Nachfolger.
* **FocSAM** (Huang et al., CVPR 2024, arXiv 2405.18706): SAM mit Dwin-MSA und P-DyReLU; erreicht
  SimpleClick-ViT-H-NoC bei **≈ 5,6 % der CPU-Inferenzzeit** – relevant für einen Mac mini.
* **SAM / SAM 2 / SAM 3**: SAM 3 (Carion et al., ICLR 2026, arXiv 2511.16719; 848M Parameter) segmentiert
  alle Instanzen eines *Konzepts* per Text oder **Bild-Exemplar**; Fine-Tuning-Code vorhanden. Der
  SAM-3-Data-Engine nutzt **KI-Verifizierer** für „Mask Verification“ und „Exhaustivity Verification“, die
  laut Paper menschliches Niveau erreichen oder übertreffen.
* **RClicks** (NeurIPS 2024, arXiv 2410.11722): 475 K echte Nutzerklicks; mit realistischen Klicks
  schneiden Methoden **schlechter als im Standardbenchmark** ab, die meisten sind nicht robust.
* **Domänengrenzen:** Für Risse/Industriedefekte berichten mehrere Arbeiten, dass SAM ohne Anpassung nicht
  direkt nutzbar ist (LoRA-Crack-Studie, Struct. Health Monit. 2025; CrackSAM, arXiv 2312.04233; DefectSAM,
  IEEE 2025). Im Few-Shot-Defektbenchmark (Liu et al., arXiv 2502.01216) erreicht SAM2 im
  Video-Tracking-Modus **47,9 mIoU**, Feature-Matching + FastSAM 39,3 mIoU (aber 90,9 FPS);
  Meta-Learning-Methoden schneiden schwach ab.

**Anwendbarkeit.** Interaktive Modelle sind für *großflächige, kompakte* Defekte (MT „uneven/fray“, Pitting
auf BSData) ein Zeitgewinn; für 1–3 px Kratzer ist unser eigenes U-Net mit Full-Res-Pfad wahrscheinlich
der bessere „Interaktor“: ein Klick auf einen übersehenen Kratzer → lokale Absenkung der Hysterese-Schwelle
um den Klick (Region-Growing im Wahrscheinlichkeitsfeld). Das ist billig zu implementieren und braucht kein
neues Modell. *Annahme, zu testen.*

### 3.4 Verify-only und Bild-Level-Feedback als schwache Supervision

| Feedback (Kosten) | Literatur | Wie im Retraining nutzen |
|---|---|---|
| **Accept** (Label korrekt) | Papadopoulos et al. 2016 (Verify-only, 6–9× Zeitersparnis bei Boxen); Konyushkova et al. 2018 (gelernter Dialog Verify↔Draw schlägt beides allein) | Pseudo-Label wird zu „hartem“ Label: Gewicht 1, Ignore-Regionen entfernen; zählt als Kalibrierpunkt |
| **„Defekt fehlt“** (Bild-Flag) | Božič et al. 2021 Mixed Supervision (Bildlabels + wenige Pixelmasken ≈ Vollsupervision auf KSDD2); One-bit Supervision (Hu et al. 2020: falsche Vermutungen als negative Information nutzen) | MIL-Loss außerhalb der aktuellen Maske (Top-k-Pooling der Logits muss hoch sein); Bild in die Korrektur-Queue; lokale Schwellenabsenkung; Bild-Level-Label für Typklassifikator |
| **„Fehlalarm“** (Komponente anklicken) | Hard-Negative-Mining (Standard) | Komponente wird explizit negativ (Gewicht > 1), Hintergrund-Patch ins Normalitätsmodell |
| **Rating** (gut/mittel/schlecht) | Lidar-Baum-Segmentierung (arXiv 2508.15646): Ratings trainieren ein Rating-Modell, das den Segmentierer verbessert, +34 % korrekt erkannte Instanzen | Ratings kalibrieren die Selbsteinschätzung (erwarteter Dice) direkt – zusätzliche Trainingsdaten für `quality.py` |
| **Ein Klick** auf übersehenen Defekt (≈ +3–5 s, *Annahme*) | Punkt-Supervision (Bearman et al., ECCV 2016, **unverified**) | Punktlabel + lokales Region-Growing → Pseudo-Maske mit breitem Ignore-Ring |

**Bewertung.** Im Industriekontext ist „Defekt fehlt“ das wertvollste Signal (FN sind am teuersten), und
es ist nahezu gratis zu erheben, *wenn* der Reviewer ohnehin das Overlay ansieht. Das aktuelle Tool
erzeugt diese Information nicht und verwertet sie nicht – hier liegt der größte Hebel pro
Implementierungsaufwand.

---

## 4. Review-Queue + AL + Audit: Allokation eines festen Menschbudgets

### 4.1 Literatur

* **Active Testing** (Kossen et al., ICML 2021, arXiv 2103.05331): Testpunkte aktiv wählen, Bias durch
  Importance-Weighting entfernen und gleichzeitig Varianz senken. **Active Surrogate Estimators**
  (Kossen et al., NeurIPS 2022) als Nachfolger.
* **Active Statistical Inference** (Zrnic & Candès, ICML 2024 Oral, arXiv 2403.03208): Labels gezielt dort
  einholen, wo das Modell unsicher ist; Vorhersagen ersetzen Labels, wo es sicher ist; **gültige CIs** für
  beliebige Black-Box-Modelle; deutlich weniger Labels als bei nicht-adaptiver Stichprobe. Die optimale
  Stichprobenregel gewichtet nach (Wurzel der) erwarteten Fehlervarianz (Form nach Paper, Details prüfen).
* **Cost-Optimal Active AI Model Evaluation** (Angelopoulos, Eisenstein, Berant, Agarwal, Fisch; arXiv
  2506.07949, 2025): kostenoptimale Aufteilung zwischen billigem „weak rater“ und teurem „strong rater“;
  gleiche Präzision bei deutlich kleinerem Budget, besonders bei heteroskedastischen Fehlern.
  **Direkt übertragbar:** weak rater = Verify-Klick (3–6 s), strong rater = vollständige Korrektur (60 s+).
* **Stratified PPI** (Fisch et al., NeurIPS 2024, arXiv 2406.04291): Stratifizierung verengt
  PPI-Intervalle deutlich, wenn die Güte des Autoraters zwischen Strata variiert.
* **Actively Testing while Learning** (Yu, Shi, Yu; NeurIPS 2023, nur Folien gefunden – Details
  **unverified**): gemeinsames Lern-/Testziel mit Rückkopplung.
* **ALBL** (Hsu & Lin, AAAI 2015): Auswahl der AL-Strategie als Multi-Armed-Bandit; besser als jede
  Einzelstrategie in deren Studien.
* **Data Engines:** SAM (assisted-manual → semi-automatic → fully automatic; 6 Retrainings; 34 → 14 s/Maske),
  SAM 2 (5,1× bzw. 8,4× schneller), SAM 3 (KI-Verifizierer übernehmen Verifikation). Tesla-„Data Engine“
  (Karpathy, Vorträge; **unverified**) und Snorkel (Ratner et al., VLDB 2017; **unverified**) als
  konzeptionelle Vorbilder. Gemeinsames Muster: **Modell im Loop wird oft retrainiert, Menschen wechseln
  von Erstellen zu Verifizieren, Verifikation wird selbst gelernt**.

### 4.2 Kernaussage für unser Tool: Review-Bilder sind ein vollständig beobachtetes Stratum

METHOD.md sagt korrekt: Review-Bilder dürfen nicht naiv als Audit dienen. Das heißt aber nicht, dass sie für
die Messung wertlos sind. Mit Stratifizierung gilt für eine Pool-Größe Q = Σᵢ yᵢ (z. B. Anzahl gefundener
Defekte):

    Q = Σ_{i ∈ Top-k (reviewt, π=1)} yᵢ          (exakt beobachtet)
      + Σ_{i ∈ Rest} fᵢ + (N_Rest/n) Σ_{j ∈ Audit ⊂ Rest} (yⱼ − fⱼ)    (PPI auf dem Rest)

Allgemeiner, mit Poisson-Sampling und bekannten Inklusionswahrscheinlichkeiten πᵢ (Risiko-proportional
mit Untergrenze ε·n/N):

    Q̂ = Σᵢ fᵢ + Σ_{i reviewt} (yᵢ − fᵢ)/πᵢ

Das ist unverzerrt (Horvitz-Thompson + PPI, vgl. Zrnic & Candès 2024), und **jede Korrektur zählt
gleichzeitig für Qualität und Messung**. Die Qualität des *ausgelieferten* Pools (reviewte Bilder
korrigiert) ergibt sich als Q̂ − Σ_R yᵢ + Σ_R yᵢ^korr mit gleichem Fehler. Die Untergrenze ε verhindert
explodierende Gewichte für „sichere“ Bilder – genau dort verstecken sich übersehene Defekte.

**Zwei Fallstricke, die das Protokoll abfangen muss:**

1. *Retraining auf Audit-Bildern.* Wird ein Audit-Bild in L aufgenommen, ist fᵢ für dieses Bild nicht mehr
   „out-of-sample“; der Korrekturterm wird verzerrt. Saubere Lösung: Audit-Bilder erst **nach** der
   Schätzung in L übernehmen und für die nächste Runde ein frisches Audit ziehen; alternativ
   Cross-Prediction-PPI mit fold-getrennten Vorhersagen (Zrnic & Candès, PNAS 2024, **unverified**) – unsere
   Cross-Fitting-Architektur ist dafür prädestiniert.
2. *Konforme Schwellen nach aktiver Auswahl.* Aktiv gewählte Bilder sind nicht austauschbar mit U. Lösung:
   L in **L_cal** (zufällig bzw. mit bekanntem π gezogen) und **L_train** (alle) trennen; konforme Schwellen
   nur auf L_cal bzw. mit Gewichten 1/π (Weighted Conformal Prediction, Tibshirani et al., NeurIPS 2019,
   **unverified**). Bei Folds: OOF-Scores der L_cal-Defekte verwenden.

### 4.3 Allokationsheuristik (bis zur eigenen Messung)

* Die Verwendungen (a) Korrektur riskanter Labels und (b) Training überlappen stark: korrigierte riskante
  Bilder sind auch Trainingsdaten. Aber AL-Literatur warnt: bei kleinem Budget sind *rein* riskante Bilder
  oft Ausreißer/ambig (TypiClust-Phasenübergang, HaCon). Daher für (b) einen Coverage-Anteil beimischen.
* (c) Messung kommt fast gratis aus (a), wenn π protokolliert wird; ein separater uniformer Anteil
  (ε ≈ 10–20 % des Budgets) schützt gegen blinde Flecken der Selbsteinschätzung.
* Weak/Strong-Rater-Aufteilung (Angelopoulos et al. 2025): erst Verify-Klicks über viele Bilder, dann
  Korrekturen nur auf geflaggten – optimale Mischung hängt von gemessenen Kosten und Fehlerraten ab und
  kann pro Aufgabe aus den Logs der ersten Runde geschätzt werden.

---

## 5. Aufgabenadaption ohne Experten-Tuning

### 5.1 Modellauswahl mit 5–30 Labels ist Rauschen

* **Perez, Kiela, Cho, *True Few-Shot Learning with Language Models*** (NeurIPS 2021, arXiv 2105.11447):
  Cross-Validation und MDL ohne Held-out-Daten sind im Mittel **nur marginal besser als zufällige Auswahl**
  und wählen oft deutlich schlechtere Modelle; HP-Auswahl kostete ADAPET 2–10 % auf 8 Tasks. Obwohl für
  Sprachmodelle gezeigt, ist das statistische Argument (Varianz der Auswahlmetrik ≫ Unterschied der
  Konfigurationen) allgemein.
* **nnU-Net** (Isensee et al., Nature Methods 2021): Selbstkonfiguration aus *festen Parametern,
  abhängigen Regeln und wenigen empirischen Entscheidungen* auf Basis eines Daten-Fingerprints
  (Bildgröße, Spacing, Intensitätsverteilung). Empirisch wird u. a. per Cross-Validation zwischen
  Konfigurationen/Ensemble gewählt (aus dem Paper; Detailregeln hier nicht erneut geprüft).
  **nnU-Net Revisited** (Isensee et al., MICCAI 2024, arXiv 2404.09556): viele behauptete
  Architekturgewinne halten rigoroser Validierung nicht stand; CNN-U-Net + gutes Framework + Skalierung
  gewinnt.
* **Bayesian Optimization** mit verrauschtem Small-Sample-Ziel: Bei OOF auf ~10 Defekten ist die Std. der
  Defekt-Recall-Schätzung grob √(0,7·0,3/10) ≈ 0,15 (eigene Rechnung, Binomial) – größer als typische
  Konfigurationsunterschiede. BO lohnt hier nicht; Entscheidungen müssen *wenige, robuste* Achsen betreffen.

### 5.2 Regeln aus dem Defekt-Fingerprint (Vorschlag)

Aus den wenigen Masken lassen sich robuste Statistiken ziehen, die Konfigurationsentscheidungen
deterministisch treiben (nnU-Net-Prinzip). Vorschlag (Regeln sind *Annahmen*, zu validieren mit Idee 3):

| Fingerprint (aus L-Masken und U) | Entscheidung | Regel (Startwert) |
|---|---|---|
| Max. Inkreisradius entlang Skelett → Defektbreite w (P10, P50) | Full-Res-Stem, Toleranz | Full-Res an, wenn w_P10 ≤ 4 px; Toleranz = clip(round(0,5·w_P50), 1, 3) |
| BBox-Längsseite ℓ (P90), Defektfläche (P50 relativ zum Bild) | Crop-Größe, Zweig für Großdefekte | crop = clip(nächste 32er-Stufe ≥ max(192, 2,5·ℓ_P90), 128, 512); bei Fläche_P50 > 5 % zusätzlicher Downscale-Ganzbildpfad |
| Elongation (Skelettlänge²/Fläche) | Loss | Bei hoher Elongation Skelett-/Topologie-Term (clDice, Shit et al. CVPR 2021, **unverified**) zu Dice+BCE |
| Defekte pro Bild, Anteil defekter Bilder in L | p_center, Sampling | p_center so, dass ≈ 50 % der Crops Defektpixel enthalten |
| Residuen-Kontrast der Defektbank | Gain-Bereich der Synthese | Gain-Untergrenze = P10 des beobachteten Kontrastverhältnisses |
| Anzahl gelabelter Defekte n_d | K-Folds, α der Garantie | K = min(5, n_d), α so wählen, dass ⌊α(n_d+1)⌋ ≥ 1 (sonst ehrlich „keine Garantie“) |
| Bildgröße, Speicher (MPS/GPU) | Batch/Iterationen | Iterationen ∝ √(n_d)·Basis, gedeckelt durch Zeitbudget |
| Kostenverhältnis FN:FP (einziger Nutzer-Knopf) | Schwellenziel | F_β mit β = √(Kosten FN/Kosten FP) statt tol-F1 |

Danach höchstens **4–8 empirische Kandidaten** (z. B. Encoder {resnet18d, größerer Encoder}, Crop {Regel,
2×Regel}, Synthese {an, aus}), bewertet per **wiederholtem Cross-Fitting** (2–3 Fold-Permutationen) auf
Defektebene mit gepaartem Bootstrap über Defekte. Abweichen vom Regel-Default nur bei Gewinn > 1 SE
(One-Standard-Error-Regel); bei Gleichstand **ensemblen statt wählen** (passt zum bestehenden Ensemble).

### 5.3 Cross-Task-Vortraining, universelle Defekt-/Anomaliemodelle

* **Defect Spectrum** (Yang et al., ECCV 2024, arXiv 2310.17316): verfeinerte, mehrklassige Pixelmasken für
  MVTec, VISION, DAGM, Cotton (DAGM/Cotton erstmals pixelgenau); Defect-Gen-Synthese bis +9,85 mIoU.
  → **Bester öffentlicher Baustein für ein klassenagnostisches Defekt-Vortraining.**
* **AnomalyCLIP** (Zhou et al., ICLR 2024, arXiv 2310.18961): objektagnostische Prompts, trainiert auf
  Hilfsdaten, Zero-Shot auf 17 Datensätzen (Industrie + Medizin) – Beleg, dass ein „Normal-vs-Defekt“-Begriff
  über Domänen transferiert.
* **Dinomaly** (Guo et al., CVPR 2025): Multi-Class-UAD, I-AUROC 99,6/98,7/89,3 % (MVTec AD/VisA/Real-IAD);
  **Dinomaly2** (arXiv 2510.17611) inkl. Few-Shot-Setting und DINOv3-Variante (bessere Pixelmetriken).
* **AnomalyDINO** (Damm et al., WACV 2025, arXiv 2405.14529): trainingsfreies Patch-kNN auf DINOv2;
  1-Shot-MVTec I-AUROC 93,1 → 96,6 %. → Direkter Kandidat zur Aufwertung unseres Normalitätsmodells.
* **MVTec AD 2** (Heckler-Kram et al., IJCV 2026, arXiv 2503.21622): > 8000 hochaufgelöste Bilder,
  8 Szenarien; Methoden mit > 90 % AU-PRO auf MVTec AD erreichen **max. 58,7 %**; SegF1 als
  Hauptmetrik (RoBiS: 51,00 bzw. 46,52 % SegF1, stark objektabhängig). Private Testsplits nur über Server.
  Lizenz CC BY-NC-SA 4.0.
* **Few-Shot-Defektsegmentierung:** GFDS/FM-SAM (Liu et al. 2025) – Meta-Learning schwach, VFMs
  vielversprechend; **SAM2-Video-Modus behandelt Bildserien als Video** (47,9 mIoU). DefectSAM (Prototyp-
  Prompts + LoRA), SAID (Szenen-Prompt aus einem Paar), Sam-Industry; Survey FM-basierte
  Defektdetektion (arXiv 2502.19106).
* **UniDG** (arXiv 2604.08915, 2026): universelles Defekt-Generierungsmodell, 300K Quadrupel
  (normal, defekt, Maske, Caption) – potenzielle Alternative/Ergänzung zu Residual-Copy-Paste
  (Zahlen nur aus Sekundärquellen).

**Bewertung.** Es gibt (in dieser Recherche) **kein** etabliertes, auf vielen Datensätzen supervised
vortrainiertes Defekt-Segmentierungsmodell, das sich mit 5–30 Masken adaptieren lässt und für feine Defekte
belegt besser ist. Das ist eine echte Lücke und für uns ein lohnendes Experiment (Idee 4). Risiken:
negative Transfer (andere Oberflächen), Lizenzen (MVTec AD 2: CC BY-NC-SA 4.0; MVTec AD vermutlich ebenso,
**unverified**; DINOv3/SAM eigene Lizenzen) – für Produktionsmodelle relevant.

### 5.4 Test-Time Adaptation

Klassische TTA (z. B. TENT, Wang et al. ICLR 2021, **unverified**) adressiert Verteilungsverschiebung zur
Laufzeit. Unser Setting ist bereits *transduktiv*: Normalitätsmodell und Self-Training sehen U. Ein
weiterer TTA-Schritt bringt vermutlich wenig; wichtiger ist **Drift-Erkennung** (Verteilung der
Anomalie-Scores/Selbsteinschätzung neuer Chargen vs. Kalibrierpool), weil Drift die Garantien bricht.

---

## 6. Tooling und Formate

| Tool | Model-in-the-Loop | Relevante Details (verifiziert) | Empfehlung |
|---|---|---|---|
| **CVAT** | Nuclio-Funktionen (Detector/Interactor) im eigenen Deployment; **SDK-Auto-Annotation-Funktionen** (`spec` + `detect`, Masken via `mask`/`encode_mask`); **Agent-Modus** (`function create-native`, `function run-agent`) – Modell läuft auf eigener Maschine; Interaktoren liefern im UI nur `mask` | XML „CVAT for images 1.1“: `<mask label rle left top width height source z_order>`; RLE = Lauflängen über die BBox (Beispiel summiert exakt auf width·height; Startwert-Konvention per Round-Trip prüfen) | **Primärziel** für Korrektur-Workflow; Agent-Modus erlaubt unser U-Net als Interaktor auf dem Mac |
| **Label Studio** | ML-Backend: `predict(tasks, context)` (auch interaktiv via `context`), `fit(event, data)` auf Webhooks (`ANNOTATION_CREATED/UPDATED`); SAM/SAM2-Beispiel-Backends nur interaktiv | Brush-Masken: **eigenes bit-gepacktes RLE über RGBA** (`brush.mask2rle`, Wert ×4 Kanäle), Ergebnis `type: brushlabels`, `value.format: rle`; mit `model_version` + `score` als *Prediction* | Zweitziel; `score` = unser Risiko → Sortierung im UI = Review-Queue |
| **X-AnyLabeling** | SAM 1/2/3, SAM-HQ, EdgeSAM, MobileSAM; U-Net-Semantiksegmentierung; X-AnyLabeling-Server für eigene Modelle; Batch-Vorhersage | Export COCO, VOC, YOLO, MASK u. a. | Leichtgewichtig lokal; MASK/COCO-Export genügt |
| **FiftyOne** | `annotate()` schickt Samples **inkl. bestehender Segmentierungsfelder** zu CVAT/Label Studio; `load_annotations()` merged zurück; Rechte (`allow_additions/deletions/spatial_edits`) | `label_type="segmentation"` mit `mask_targets` | Bequemer Brückenkopf: Review-Queue als FiftyOne-View (sortiert nach Risiko), Rundreise ohne eigenes Formatparsing |
| **MONAI Label** | DeepEdit, DeepGrow, SAM2; Viewer inkl. CVAT; AL-Workflow (Endoskopie) | Strategien im README nicht spezifiziert | Vorbild für Server-Architektur, für uns nicht nötig |

**Mindestumfang Export/Import (Vorschlag):**

1. `export --format coco-rle`: Instanz = Defektkomponente, mit `score`, `risk`, `p_missed_defect`,
   Komponentenwahrscheinlichkeit; RLE nach pycocotools-Konvention (spaltenweise; per Round-Trip-Test prüfen).
2. `export --format cvat-xml` (bzw. Upload über `cvat-sdk`) mit `source="auto"` und Bildattributen
   `risk`, `rank`, `selection` ∈ {review, audit, coverage}, `pi` (Inklusionswahrscheinlichkeit).
3. `export --format labelstudio`: Tasks mit `predictions` (Brush-RLE via `label_studio_sdk.converter.brush`),
   `model_version` = Run-ID, `score` = Risiko.
4. `import`: korrigierte Masken + **Status** (accepted / corrected / flagged_missing / flagged_fp / skipped)
   + **Zeitstempel** (Start/Ende je Bild) + Annotator-ID. Ohne Zeiten kein Kostenmodell, ohne π keine
   gültige Schätzung.

---

## 7. Konkretes Human-in-the-Loop-Protokoll für eine neue Aufgabe

Budgetangaben in Menschminuten; Zeiten pro Aktion sind *Annahmen* aus Abschn. 3.1 und werden ab Runde 1
durch gemessene Werte ersetzt. Beispiel: |U| ≈ 2000 Bilder, Defektrate unbekannt (5–30 %).

**Phase 0 – Automatisch, 0 min.**
1. Fingerprint berechnen (Abschn. 5.2) aus vorhandenen Masken; falls noch keine existieren, nach Phase 1.
2. DINOv3-Embeddings (global + Patch) für alle Bilder; Normalitätsmodell auf U; Anomalie-Strata
   S1 = Top 5 %, S2 = 5–20 %, S3 = Rest.
3. Konfiguration nach Regeln; keine HPO.

**Phase 1 – Triage & Start-Labels, ≈ 30–40 min** (entfällt teilweise, wenn 5–30 Masken schon da sind).
1. *Triage (≈ 10 min):* ~150 Bilder Ja/Nein „enthält Defekt?“ (≈ 4 s/Bild). Auswahl per Poisson-Sampling
   mit **protokollierten π** über die Strata (z. B. π = 0,6 / 0,1 / 0,02), innerhalb S1 zusätzlich
   MaxHerding-Coverage auf Patch-Embeddings, um verschiedene Defektarten zu treffen.
   Ergebnis: Bildlabels (für MIL/Typklassifikator/Bild-Level-Kalibrierung) und Schätzung der Defektrate.
2. *Masken (≈ 20–30 min):* 10–20 positive Bilder maskieren – im **Korrekturmodus** (Vorschlag aus
   Normalitätskarte bzw. Vorgängermodell + Klick-Tool), Zeichnen nur als Rückfall. Davon ≥ 50 % per π
   zufällig aus den Triage-Positiven → **L_cal**; der Rest coverage-gewählt → nur L_train.
   *Wenn bestehende 5–30 Masken „die schönsten Beispiele“ sind:* als L_train behandeln und ein kleines
   L_cal (≥ 9 Defekte für α = 0,1) gemäß diesem Schritt ergänzen, sonst ist die Recall-Garantie nicht belegt.

**Phase 2 – Lauf 1, 0 min** (Rechenzeit). Ausgabe: Labels, Risiko, `p_missed_defect`, Exporte (Abschn. 6).

**Phase 3 – Runde 1, ≈ 45 min.**
1. *Verify-Durchgang (≈ 10 min):* ~120 Bilder à ~5 s: Top-Risiko (≈ 70 %), uniform gezogene Bilder mit
   bekanntem π (≈ 20 %; Untergrenze ε), Coverage-Kandidaten aus Region-Embeddings mit hoher „unerklärter
   Anomalie“ (≈ 10 %). Antworten: **OK / Defekt fehlt (+ optional 1 Klick) / Fehlalarm (Komponente
   anklicken) / Grenze grob falsch**.
2. *Korrektur (≈ 20 min):* geflaggte Bilder nach erwartetem Gewinn pro erwarteter Sekunde ordnen
   (Gewinn ≈ erwartete FN-Komponenten; Zeit ≈ Komponentenzahl × gemessene s/Komponente); ~15–25 Bilder.
   Nur Topologie korrigieren, Grenzen nur außerhalb der Toleranz.
3. *Audit (≈ 15 min):* n ≈ 20 Bilder per Poisson-Sampling mit π ∝ √(geschätzte Fehlervarianz) + Untergrenze
   (Active Inference), **aus den noch nicht reviewten Bildern**, vollständige Prüfung inkl. Suche nach
   übersehenen Defekten, Vorlage optional ausgeblendet. → `autolabel estimate` mit Strata
   {reviewt (π = 1), Rest (IPW-PPI)} → CIs für Defekt-Recall, Precision, Dice des **ausgelieferten** Pools.

**Retrain-Trigger** (einer genügt): ≥ 5 neue Defektmasken *oder* L wächst um ≥ 20 %; ≥ 1 neue Defektart
(Typklassifikator/Triage); Audit zeigt CI-Obergrenze der Miss-Rate über Ziel; ≥ 10 „Defekt fehlt“-Flags
(MIL-Signal). Audit-Bilder gehen erst *nach* der Schätzung in L ein; danach frisches Audit.

**Runde 2+ (je ≈ 30 min):** gleiche Aufteilung (≈ 1/4 Verify, 1/2 Korrektur, 1/4 Audit). Ab Runde 3
optional Bandit (ALBL-artig) über die Akquisitionsanteile, Belohnung = gemessener Recall-Gewinn pro Minute.

**Stopp-Regel** (in dieser Reihenfolge prüfen):
1. *Qualitätsziel erreicht:* obere 90-%-CI-Grenze der Defekt-Miss-Rate ≤ Ziel (z. B. 10 %) **und** untere
   Grenze der Precision ≥ Ziel → stoppen, Produktivmodell trainieren.
2. *Grenznutzen erschöpft:* Recall-Gewinn (Punktschätzer) der letzten Runde < 1 Punkt pro 10 min **und**
   kein neuer Defekttyp → keine weiteren Trainingslabels; ggf. nur noch Audit zur CI-Verengung.
3. *CIs zu breit, Punktschätzer gut:* Budget von Korrektur zu Audit verschieben (Messproblem, kein
   Modellproblem).
4. *Budget erschöpft:* Bericht mit aktuellen CIs, ehrlich „Ziel nicht belegt“.

**Gesamt erste Aufgabe:** ≈ 90–120 min bis zum ersten vermessenen Label-Pool; jede weitere Runde ≈ 30 min.

**Automatisches Tuning im Protokoll:** Regeln aus Phase 0; nach Phase 2 die 4–8 empirischen Kandidaten
per wiederholtem Cross-Fitting (nur Rechenzeit), 1-SE-Regel, sonst Ensemble; Schwellenziel aus dem einen
Nutzer-Knopf FN:FP. Nach Runde 1 wird das Kostenmodell (s/Aktion, Accept-Rate) aus den Logs geschätzt und
steuert die Anteile Verify/Korrektur/Audit (Weak/Strong-Rater-Logik nach Angelopoulos et al. 2025).

---

## 8. Priorisierte Empfehlungen

| # | Empfehlung | Erwarteter Gewinn | Aufwand | Risiko |
|---|---|---|---|---|
| 1 | **Selektions-Buchhaltung:** jedes menschlich bearbeitete Bild mit Grund (review/audit/coverage/triage) und π speichern; L_cal vs. L_train trennen | Hoch (Gültigkeit von Conformal/PPI bei jedem aktiven Loop) | Klein | Gering |
| 2 | **Stratifizierte Schätzung:** Review-Top-k als Zensus-Stratum + PPI auf dem Rest; später IPW-PPI mit π ∝ Fehler und Untergrenze | Hoch (Messung „gratis“ aus Review; schmalere CIs) | Klein–mittel (`ppi.py` erweitern) | Gering bei ε-Untergrenze; Varianz bei zu kleinem π |
| 3 | **Verify-first-Feedback** (OK / fehlt / Fehlalarm / Grenze) im Export/Import + Verwertung im Training (MIL-Loss, Hard Negatives, Accept = hartes Label) | Hoch für Recall pro Minute | Mittel | Mittel: MIL-Loss kann Fehlalarme fördern → nur mit Ignore-Ring und kleinem Gewicht |
| 4 | **Export/Import CVAT + Label Studio + COCO-RLE** inkl. Zeitstempel/Status | Hoch (reale Nutzbarkeit; Kostenmodell) | Mittel | Gering; RLE-Konventionen per Round-Trip-Tests absichern |
| 5 | **Fingerprint-Regeln + Mini-Kandidatenmenge mit 1-SE-Regel** statt HPO | Mittel–hoch (Robustheit über Aufgaben) | Klein–mittel | Regeln können für Ausreißeraufgaben falsch sein → LODO-Validierung (Idee 3) |
| 6 | **Startauswahl** über Anomalie-Strata + MaxHerding auf DINOv3-Region-Embeddings (mit π) | Mittel (Varianzreduktion, schnelleres Finden seltener Typen) | Klein | Gering |
| 7 | **Normalitätsmodell auf DINOv2/v3-Patch-kNN** (AnomalyDINO-Stil) | Mittel–hoch für „übersehene Defekte“-Signal (bereits in HANDOFF als Idee) | Klein | Lizenz/Download DINOv3 |
| 8 | **Eigenes U-Net als Klick-Interaktor** (Klick → lokale Hysterese-Absenkung), SAM2/3 nur für großflächige Defekte über X-AnyLabeling/CVAT | Mittel | Mittel | Bei dünnen Defekten SAM eher hinderlich |
| 9 | **Gemischte Akquisition** (Risiko/Coverage/Zufall ≈ 40/40/20), Bandit erst ab Runde 3 | Unsicher (AL vs. Random oft knapp) | Klein | Kein Gewinn ggü. Random möglich → immer mitmessen |
| 10 | **Cross-Task-Defekt-Vortraining** (Idee 4) | Unsicher, potenziell hoch bei n = 5 | Hoch | Negative Transfer, Lizenzen |

**Nicht empfohlen:** BO/HPO über viele Parameter auf 5–30 Labels; rein unsicherheitsbasierte Top-k-Auswahl
für Training bei Kleinstbudget; Review-Bilder ohne Stratifizierung ins Audit mischen; Superpixel-Queries für
dünne Defekte; Label-Studio-Brush-RLE mit COCO-RLE verwechseln.

---

## 9. Neue Forschungsideen (über die Literatur hinaus)

### Idee 1 – „Review = Audit“: probabilistische Review-Queue mit gemeinsamer Optimierung von Qualität und Messung

*Begründung.* Active Inference (Zrnic & Candès) und Cost-Optimal Evaluation (Angelopoulos et al.)
optimieren nur die **Messung**; Review-Queues optimieren nur die **Qualität**. Bei uns verbessert jede
Messung zugleich das Produkt (korrigierte Labels). Vorschlag: π_i = (1−ε)·π_i^Qualität + ε·n/N mit
π^Qualität ∝ erwartete Anzahl übersehener Defekte; Schätzung per IPW-PPI.
*Hypothesen.* Bei gleicher Menschzeit (H1) Abdeckung der 90-%-CIs ≥ 88 %; (H2) CI-Breite des Defekt-Recalls
≤ 0,7× der Variante „Top-k-Review + separates Zufallsaudit“; (H3) Pool-Recall nach Review ≥ 95 % des
deterministischen Top-k.
*Experiment.* MT, BSData, KSDD2, MVTec AD 2 (öffentlicher Test); GT als simulierter Mensch; Kosten aus
Abschn. 3.1; ε ∈ {0; 0,1; 0,3; 1}; 500 Wiederholungen des Samplings pro Lauf. *Falsifiziert*, wenn für
ε ≥ 0,1 die Abdeckung < 88 % liegt oder H2 verfehlt wird.

### Idee 2 – Ein-Bit-Feedback als Constraint-Training („Missed-Defect-MIL“)

*Begründung.* Mixed Supervision (Božič et al.) nutzt Bildlabels *vor* dem Training; One-bit Supervision
nutzt Ja/Nein für Klassifikation. Für Segmentierung mit Pseudo-Labels fehlt die Kombination: Feedback auf
das *eigene* Pseudo-Label als Constraint (Accept → harte Maske; „fehlt“ → Top-k-MIL außerhalb der Maske
+ lokale Schwellenabsenkung; Fehlalarm → Hard Negative).
*Hypothese.* Recall-Gewinn pro Menschminute auf U und T ist im Bereich 0–60 min höher als mit
Vollkorrekturen weniger Bilder.
*Experiment.* MT, BSData, KSDD2, DeepCrack; simulierte Feedbackkosten 5 s (Flag), 9 s (Flag + Klick),
60–120 s (Korrektur); Kurven Defekt-Recall@fixe FP/Bild vs. Minuten, 3 Seeds. *Falsifiziert*, wenn die
Flag-Kurve im gesamten Bereich unter der Korrekturkurve liegt oder die FP-Rate um > 20 % steigt.

### Idee 3 – Fingerprint-Regeln mit Leave-One-Dataset-Out validieren („tuning-free by construction“)

*Begründung.* nnU-Net zeigt, dass Regeln aus Datenstatistiken Tuning ersetzen können – aber für Organe, nicht
für 1–3 px Defekte bei 10 Labels. Perez et al. zeigen, dass Few-Shot-Auswahl versagt.
*Hypothese.* Regelbasierte Konfiguration erreicht auf dem gehaltenen Datensatz ≥ 90 % der Testqualität der
Oracle-Konfiguration (beste von ~30 Konfigurationen auf T) und ist (a) besser als ein fixer Default und
(b) mindestens gleich gut wie OOF-HPO über 30 Konfigurationen.
*Experiment.* LODO über MT, BSData, DeepCrack, CFD, KSDD2, MVTec-AD-Kategorien (screw, metal_nut, tile,
leather …), VisA; n_label ∈ {5, 10, 30}; Metrik: Defekt-Recall@FP/Bild und tol-F1 auf T. *Falsifiziert*,
wenn OOF-HPO die Regeln signifikant schlägt (gepaarter Test über Datensätze×Seeds) – dann ist unsere
These „Tuning auf L ist Rauschen“ für dieses Setting falsch.

### Idee 4 – Klassenagnostischer „Defekt-Prior“ durch Cross-Task-Vortraining

*Begründung.* AnomalyCLIP zeigt Transfer eines Normal/Anomal-Begriffs über 17 Datensätze; Defect Spectrum
liefert pixelgenaue Masken; supervised Vortraining für Segmentierung mit Few-Shot-Adaption ist kaum
untersucht. Unser Residual-Copy-Paste skaliert natürlich auf viele Quell-Datensätze.
*Hypothese.* Vortraining (U-Net bzw. Adapter auf DINOv3) auf der Vereinigung der übrigen Datensätze steigert
bei n = 5 den Defekt-Recall auf T um ≥ 5 Punkte; der Effekt schrumpft bei n = 30; insbesondere steigt der
Recall auf **Defekttypen, die in L fehlen**.
*Experiment.* LODO wie Idee 3; Vergleich ImageNet-Init vs. Defekt-Prior-Init vs. Defekt-Prior + Anomalie-
kanal; zusätzlich MT mit in L zurückgehaltenen Klassen. *Falsifiziert*, wenn der Gewinn bei n = 5 < 2 Punkte
oder in ≥ 2 von 7 Datensätzen negativ ist (negativer Transfer).

### Idee 5 – Entdeckung unbekannter Defekttypen als eigenes Akquisitionsziel

*Begründung.* Die schlimmsten FN sind Typen, die in L nicht vorkommen (METHOD.md, Grenzen). AL-Literatur
optimiert Modellgenauigkeit, nicht Typ-Entdeckung. Signal: „unerklärte Anomalie“ (Normalitätsmodell hoch,
Ensemble niedrig), darauf Coverage (MaxHerding) im Region-Embedding-Raum, damit nicht 20× derselbe
Reflexionstyp gezeigt wird.
*Hypothese.* Nach 20/50 Verify-Queries werden ≥ 2× so viele neue Defekttypen gefunden wie mit Random und
≥ 1,5× wie mit Entropie oder TypiClust.
*Experiment.* MT (5 Klassen), MVTec AD (mehrere Typen je Kategorie), VisA, MVTec AD 2; L enthält nur 1–2
Typen; Metrik: Anzahl entdeckter Typen vs. Queries, plus FP-Last (Anteil Normalbilder unter den Queries).
*Falsifiziert*, wenn der Faktor gegenüber Random < 1,5 bleibt.

### Idee 6 (Messinfrastruktur) – Gelerntes Zeitmodell für kostensensitive Akquisition

Logge Zeiten pro Bild/Aktion; regressiere Korrekturzeit auf vorhergesagte Komplexität (Komponentenzahl,
Randlänge, Fläche unsicherer Pixel). Akquisition nach erwartetem Gewinn / erwarteter Zeit. *Test:* kleine
Nutzerstudie (2 Annotatoren × 1 h, KSDD2 + MT), Gewinn pro Minute vs. reine Risikosortierung;
*falsifiziert*, wenn das Zeitmodell R² < 0,3 hat oder der Gewinn pro Minute nicht steigt.

---

## 10. Offene Risiken

* **Automation Bias** bei Korrekturen (Abschn. 3.2) – messen, nicht annehmen.
* **Abhängige Defekte** innerhalb eines Bildes: PPI/Conformal auf Bildebene clustern (Bild = Einheit).
* **Simulierte Menschen** (GT als Oracle) überschätzen Konsistenz; Grenzsubjektivität erst in echter
  Nutzerstudie sichtbar.
* **Lizenzen** von MVTec AD/AD 2, DINOv3, SAM 3 für Produktivmodelle prüfen.

---

## 11. Referenzen

Verifiziert = über Websuche (Abstract/Auszug) oder GitHub gefunden. **(unverified)** = aus Vorwissen,
in dieser Recherche nicht bestätigt.

**Aktives Lernen**
- Hacohen, G. et al. *Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets.* ICML 2022. arXiv:2202.02794. Code: github.com/avihu111/TypiClust
- Yehuda, O. et al. *Active Learning Through a Covering Lens* (ProbCover). NeurIPS 2022. arXiv:2205.11320
- Bae, W. et al. *Generalized Coverage for More Robust Low-Budget Active Learning* (MaxHerding). ECCV 2024. arXiv:2407.12212
- Mishal, I. & Weinshall, D. *DCoM: Active Learning for All Learners.* 2024 (über TypiClust-Repo gefunden; arXiv-ID nicht verifiziert)
- Mittal, S. et al. *Parting with Illusions about Deep Active Learning.* 2019. arXiv:1912.05361
- Mittal, S., Niemeijer, J., Schäfer, J. P., Brox, T. *Best Practices in Active Learning for Semantic Segmentation.* GCPR 2023. arXiv:2302.04075; erweitert: *Realistic Evaluation of Deep Active Learning for Image Classification and Semantic Segmentation.* IJCV 2025
- Lüth, C. et al. *Navigating the Pitfalls of Active Learning Evaluation.* NeurIPS 2023. arXiv:2301.10625. Code: github.com/IML-DKFZ/realistic-al
- Gupte, S. R. et al. *Revisiting Active Learning in the Era of Vision Foundation Models.* TMLR 2024. arXiv:2401.14555
- Chen, L. et al. *Making Your First Choice: To Address Cold Start Problem in Vision/Medical Active Learning* (HaCon). NeurIPS-Workshop 2022 / MIDL 2023. arXiv:2210.02442
- COLosSAL: Cold-start AL benchmark for 3D medical segmentation. MICCAI 2023 (Autoren nicht verifiziert)
- Levy et al. *From Cold Start to Active Learning: Embedding-Based Scan Selection for Medical Image Segmentation.* arXiv:2601.18532, 2026
- Cold-start AL mit SAM-Pseudo-Masken. MICCAI 2025, Paper 0877 (Titel/Autoren nicht verifiziert)
- Combes, J., Derville, A., Coeurjolly, J.-F. *When Imbalance Comes Twice: Active Learning under Simulated Class Imbalance and Label Shift in Binary Semantic Segmentation.* arXiv:2601.06209, 2026
- Cai, L., Pahwa, R. S. et al. *Exploring Active Learning for Semiconductor Defect Segmentation.* ICIP 2022; arXiv:2507.17359
- *Active learning for industrial defect detection: a study on hybrid sampling strategies.* Int. J. Adv. Manuf. Technol. 2025, doi:10.1007/s00170-025-17378-7 (Autoren nicht verifiziert)
- *Active Learning for Imbalanced Civil Infrastructure Data.* arXiv:2210.10586 (Autoren nicht verifiziert)
- Cai, L. et al. *Revisiting Superpixels for Active Learning in Semantic Segmentation with Realistic Annotation Costs.* CVPR 2021
- Kim, H. et al. *Adaptive Superpixel for Active Learning in Semantic Segmentation.* ICCV 2023
- *ESA: Annotation-Efficient Active Learning for Semantic Segmentation.* arXiv:2408.13491 (Autoren nicht verifiziert)
- Hwang, S. et al. *Active Learning for Semantic Segmentation with Multi-class Label Query.* NeurIPS 2023. arXiv:2309.09319
- Kim, H., Hwang, S., Kwak, S., Ok, J. *Active Label Correction for Semantic Segmentation with Foundation Models.* ICML 2024. arXiv:2403.10820
- *A²LC: Active and Automated Label Correction for Semantic Segmentation.* AAAI 2026. arXiv:2506.11599 (Autoren nicht verifiziert)
- *Diffusion-Driven Two-Stage Active Learning for Low-Budget Semantic Segmentation.* NeurIPS 2025 (Autoren nicht verifiziert)
- Hsu, W.-N. & Lin, H.-T. *Active Learning by Learning* (ALBL). AAAI 2015
- *Active Learning Enabled Low-cost Cell Image Segmentation Using Bounding Box Annotation.* arXiv:2405.01701 (Autoren nicht verifiziert)

**Interaktion, Annotationskosten**
- Kirillov, A. et al. *Segment Anything.* ICCV 2023. arXiv:2304.02643
- Ravi, N. et al. *SAM 2: Segment Anything in Images and Videos.* 2024. arXiv:2408.00714
- Carion, N. et al. *SAM 3: Segment Anything with Concepts.* ICLR 2026. arXiv:2511.16719. Code: github.com/facebookresearch/sam3
- Liu, Q. et al. *SimpleClick: Interactive Image Segmentation with Simple Vision Transformers.* ICCV 2023. arXiv:2210.11006
- Chen, X. et al. *FocalClick: Towards Practical Interactive Image Segmentation.* CVPR 2022. arXiv:2204.02574; *FocalClick-XL.* arXiv:2506.14686
- Huang, Y. et al. *FocSAM: Delving Deeply into Focused Objects in Segmenting Anything.* CVPR 2024. arXiv:2405.18706
- *RClicks: Realistic Click Simulation for Benchmarking Interactive Segmentation.* NeurIPS 2024. arXiv:2410.11722 (AIRI; Autoren nicht verifiziert)
- Papadopoulos, D. P. et al. *We Don't Need No Bounding-Boxes: Training Object Class Detectors Using Only Human Verification.* CVPR 2016. arXiv:1602.08405
- Papadopoulos, D. P. et al. *Extreme Clicking for Efficient Object Annotation.* ICCV 2017. arXiv:1708.02750
- Konyushkova, K. et al. *Learning Intelligent Dialogs for Bounding Box Annotation.* CVPR 2018. arXiv:1712.08087
- Benenson, R., Popov, S., Ferrari, V. *Large-Scale Interactive Object Segmentation with Human Annotators.* CVPR 2019. arXiv:1903.10830
- Cordts, M. et al. *The Cityscapes Dataset for Semantic Urban Scene Understanding.* CVPR 2016. arXiv:1604.01685
- Lin, T.-Y. et al. *Microsoft COCO.* ECCV 2014 (79,2 s/Instanz über Sekundärquelle gefunden)
- Hu, H. et al. *One-bit Supervision for Image Classification.* NeurIPS 2020. arXiv:2009.06168
- Božič, J., Tabernik, D., Skočaj, D. *Mixed Supervision for Surface-Defect Detection: From Weakly to Fully Supervised Learning.* Computers in Industry 2021. arXiv:2104.06064
- Palazzo, G. et al. Real-world validation of AI-based contouring (Prostata). Physics and Imaging in Radiation Oncology 2023
- ESTRO 2020 Poster: Evaluation of AI-based contouring tools in prostate cancer (MVision)
- *Reducing manual workload in CT and MRI annotation with the Segment Anything Model 2.* BMC Medical Imaging 2026 (TU Dresden; Autoren nicht verifiziert)
- *Weakly-Supervised Learning for Tree Instances Segmentation in Airborne Lidar Point Clouds.* arXiv:2508.15646 (Autoren nicht verifiziert)
- Bearman, A. et al. *What's the Point: Semantic Segmentation with Point Supervision.* ECCV 2016 **(unverified)**
- Guo, Y. et al. *Segment anything model-based crack segmentation using low-rank adaption fine-tuning.* Structural Health Monitoring 2025
- *Fine-tuning vision foundation model for crack segmentation in civil infrastructures* (CrackSAM). arXiv:2312.04233

**Budgetallokation, Evaluation**
- Kossen, J. et al. *Active Testing: Sample-Efficient Model Evaluation.* ICML 2021. arXiv:2103.05331
- Kossen, J. et al. *Active Surrogate Estimators.* NeurIPS 2022 (nur als Zitat gefunden)
- Zrnic, T. & Candès, E. *Active Statistical Inference.* ICML 2024. arXiv:2403.03208
- Angelopoulos, A. N. et al. *Cost-Optimal Active AI Model Evaluation.* arXiv:2506.07949, 2025
- Fisch, A. et al. *Stratified Prediction-Powered Inference for Hybrid Language Model Evaluation.* NeurIPS 2024. arXiv:2406.04291
- Yu, Shi, Yu. *Actively Testing while Learning.* NeurIPS 2023 (nur Folien gefunden; **unverified**)
- Zrnic, T. & Candès, E. *Cross-Prediction-Powered Inference.* PNAS 2024 **(unverified)**
- Tibshirani, R. J. et al. *Conformal Prediction Under Covariate Shift.* NeurIPS 2019 **(unverified)**
- Ratner, A. et al. *Snorkel: Rapid Training Data Creation with Weak Supervision.* VLDB 2017 **(unverified)**
- Karpathy, A. Tesla „Data Engine“ (Vorträge, 2019–2021) **(unverified)**

**Aufgabenadaption, Foundation-/Anomaliemodelle**
- Isensee, F. et al. *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods 18:203–211, 2021
- Isensee, F. et al. *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation.* MICCAI 2024. arXiv:2404.09556
- Perez, E., Kiela, D., Cho, K. *True Few-Shot Learning with Language Models.* NeurIPS 2021. arXiv:2105.11447
- Heckler-Kram, L. et al. *The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection.* IJCV 2026. arXiv:2503.21622
- *RoBiS: Robust Binary Segmentation for High-Resolution Industrial Images.* arXiv:2505.21152
- Damm, S. et al. *AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2.* WACV 2025. arXiv:2405.14529
- Guo, J. et al. *Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection.* CVPR 2025. arXiv:2405.14325; *Dinomaly2.* arXiv:2510.17611
- Zhou, Q. et al. *AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection.* ICLR 2024. arXiv:2310.18961
- Yang, S. et al. *Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics.* ECCV 2024. arXiv:2310.17316
- Liu, T. et al. *Exploring Few-Shot Defect Segmentation in General Industrial Scenarios with Metric Learning and Vision Foundation Models.* arXiv:2502.01216, 2025. Code: github.com/liutongkun/GFDS
- *DefectSAM: Prototype Prompt Guided SAM for Few-Shot Defect Segmentation.* IEEE 2025, ieeexplore 10912073 (Autoren nicht verifiziert)
- *SAID: Segment All Industrial Defects with Scene Prompts.* Sensors 25(16):4929, 2025
- *A Survey on Foundation-Model-Based Industrial Defect Detection.* arXiv:2502.19106
- *Large-Scale Universal Defect Generation: Foundation Models and Datasets* (UniDG). arXiv:2604.08915, 2026
- Meta AI. DINOv3 (Backbones, Lizenz, timm/HF-Support). github.com/facebookresearch/dinov3 (Repo verifiziert; Paper und Erstautor – vermutlich Siméoni, O. et al., arXiv 2508.10104 – **unverified**)
- Wang, D. et al. *Tent: Fully Test-Time Adaptation by Entropy Minimization.* ICLR 2021 **(unverified)**
- Shit, S. et al. *clDice – a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation.* CVPR 2021 **(unverified)**

**Tools (GitHub/Doku verifiziert)**
- CVAT: XML-Format „CVAT for images 1.1“ (github.com/cvat-ai/cvat, `site/content/en/docs/dataset_management/formats/format-cvat.md`); SDK Auto-Annotation (`.../api_sdk/sdk/auto-annotation.md`)
- Label Studio ML Backend: github.com/HumanSignal/label-studio-ml-backend; Brush-RLE: github.com/HumanSignal/label-studio-sdk (`converter/brush.py`)
- X-AnyLabeling: github.com/CVHub520/X-AnyLabeling
- FiftyOne CVAT-Integration: github.com/voxel51/fiftyone (`docs/source/integrations/cvat.rst`)
- MONAI Label: github.com/Project-MONAI/MONAILabel
