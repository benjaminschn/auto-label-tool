# 01 – Foundation Models & Zero-/Few-Shot-Anomaliesegmentierung für Defekte aller Größen

Literaturreview für `auto-label-tool` · Stand 18.10.2026 · Bezug: [`../METHOD.md`](../METHOD.md), [`../HANDOFF.md`](../HANDOFF.md)

## 0. Quellenlage und Lesehilfe

* **Zugriff:** In dieser Umgebung waren arXiv, CVF Open Access, OpenReview und Hugging Face *nicht direkt*
  abrufbar. Volltexte konnten deshalb nicht gelesen werden. Belege stammen aus (a) GitHub-READMEs und
  veröffentlichten Ergebnisdateien der offiziellen Repositories, (b) Abstracts und Suchauszügen. Gegen Ende
  war außerdem das gemeinsame Web-Suchbudget erschöpft; offene Punkte stehen in Abschnitt 12.
* **Verifikationsstufen** hinter jeder Zahl:
  **[R]** = direkt aus Repository-README oder Ergebnisdatei (CSV/JSON) gelesen;
  **[A]** = aus Abstract oder Suchauszug (Volltext nicht eingesehen);
  **(unverified)** = aus Erinnerung oder nur indirekt, nicht geprüft. Zahlen ohne Marker gibt es nicht.
* **Metrik-Kürzel:** I-AUROC (Bild), P-AUROC (Pixel), P-AP (Pixel-Average-Precision), P-F1(-max)
  (F1 bei der *besten* Schwelle, also ein Orakelwert), AUPRO / AU-PRO@x (Per-Region-Overlap bis FPR x;
  MVTec AD nutzt x = 0,3, MVTec AD 2 offiziell x = 0,05), SegF1 (Pixel-F1 bei *einer* festen, vorab
  gewählten Schwelle; offizielle Metrik der VAND-3.0/4.0-Challenges).
* Zahlen aus verschiedenen Papers sind fast nie direkt vergleichbar (Auflösung, Maskenbinarisierung,
  Protokoll). Das ADer-Benchmark-Paper nennt genau diese Verzerrungen als Grund für eine Standardisierung
  (Zhang et al., arXiv 2406.03262) [A].

---

## 1. Executive Summary

1. **P-AUROC ist für unsere Frage wertlos.** Er bleibt hoch, auch wenn die Lokalisierung praktisch
   versagt: MuSc-V2 auf Eyecandies/CandyCane hat P-AUROC 97,3, aber P-AP 1,8 [R]. MambaAD auf Real-IAD
   hat P-AUROC 98,5, aber P-AP 33,0 [R]. Für Labelqualität zählen **SegF1 bei fester Schwelle,
   AU-PRO@0,05, P-AP und Recall auf Defektebene, getrennt nach Größenklassen**.
2. **Auf harten Daten reicht unüberwachte Normalität allein nicht für Produktionslabels.** Auf MVTec AD 2
   erreicht der beste Beitrag zur CVPR-2026-Challenge (SuperADD: DINOv3 + kNN, trainingsfrei)
   **57,4 % SegF1** auf dem privaten Testset [A]. Zero-Shot-Verfahren liegen rund 15 Punkte darunter [A].
   Einzelne Kategorien brechen ein (RoBiS: *Can* 1,9 %, *Wall plugs* 14,4 % SegF1) [R]. Das bestätigt
   unser Design: ein überwachtes Ensemble aus wenigen Masken, die Normalität nur als Prior bzw.
   Review-Signal.
3. **DINOv2/DINOv3 haben ImageNet-CNNs als Merkmalsextraktor abgelöst**, und zwar in praktisch allen
   Spitzenverfahren 2025/26. DINOv3 verbessert vor allem die Pixelebene: Dinomaly mit DINOv3-L erreicht
   auf MVTec AD P-AP 74,2 und AUPRO 95,8 [R]. Laut VAND-4.0-Bericht dominierten DINOv3-Backbones den
   Industrie-Track [A]. **Der günstigste große Hebel ist, den Backbone unseres Normalitätsmodells
   (WideResNet50) auszutauschen.**
4. **Auflösung schlägt Architektur bei kleinen Defekten.** MuSc mit DINOv2-B/14 steigt von 336 auf 518 px
   von P-AP 59,5 auf 65,3 (MVTec AD) [R]. Die Top-Lösungen auf MVTec AD 2 arbeiten alle mit
   überlappenden Kacheln bzw. Crops (SuperADD, RoBiS) [A/R]. Gelernte Feature-Upsampler (LoftUp, JAFAR,
   AnyUp) sind für Anomalie-*Segmentierung* kaum evaluiert. Sie sind ein Experiment, kein Default.
5. **Batch- bzw. transduktive Verfahren entsprechen unserem Setting.** Gemeint sind MuSc (ICLR 2024),
   CoDeGraph (TMLR 2025) und DeMuS (ECCV 2026), die den unlabeled Pool selbst nutzen. Unser
   kNN-Ausschluss des eigenen Bildes ist im Kern ein Mutual-Scoring-Verfahren. Es hat aber dieselbe
   bekannte Schwäche: **„konsistente Anomalien“** (wiederkehrende Defekttypen erklären sich gegenseitig;
   „neighbor burnout“). Genau das ist bei uns zu erwarten (MT: 301 defekte Bilder in U).
6. **CLIP-/VLM-Segmentierung ist auf Pixelebene schwach.** Auf MVTec AD zero-shot erreicht WinCLIP
   F1-max 31,7 und APRIL-GAN P-AP 40,8 [R]. Der reine Vision-Ansatz AnomalyVFM (CVPR 2026) erreicht
   P-AP ≈ 50 [R], liegt damit aber *unter* dem trainingsfreien Batch-Verfahren MuSc (P-AP 62,7–67,4) [R].
   Multimodale LLMs (MLLMs) erreichen im MMAD-Benchmark (ICLR 2025) höchstens 74,9 % Accuracy
   (GPT-4o) und liefern keine verlässlichen Masken [A]. **Die Teamannahme ist damit gestützt: VLMs
   gehören nicht in den Labelpfad.** Sinnvoll sind sie höchstens als unabhängiges Zusatzsignal oder zur
   Benennung von Klassen.
7. **Wenige Pixelmasken bringen messbar viel.** Auf KolektorSDD2 steigt der Pixel-AP von SuperSimpleNet
   von 45,8 (unüberwacht) auf 59,1 (16 Masken plus Bildlabels) und auf 81,3 (alle 246 Masken) [R].
   Open-Set-supervised-Methoden (DevNet, DRA, BGAD, PRN, AHL, DPDL, IDEAL) sind überwiegend auf Bildebene
   evaluiert. Für Pixelsegmentierung mit ≈ 10 Anomalien und mehreren, teils ungesehenen Typen gibt es
   keinen klaren Sieger. Die übertragbaren Designprinzipien sind: **Residuen-Features gegenüber
   Normalität, Pseudo-/synthetische Anomalien, ein getrennter Kanal für ungesehene Typen.**
8. **Universelle Modelle (Dinomaly2, INP-Former, ArcAD) zeigen: Ein Modell für viele Produkte kostet
   kaum Genauigkeit.** Dinomaly2 meldet I-AUROC 99,9 auf MVTec AD, 99,3 auf VisA und 92,1 auf Real-IAD [A].
   Für ein *aufgabenagnostisches* Tool ist aber ein **gefrorener Foundation-Backbone mit einer
   Memory-Bank pro Aufgabe** attraktiver: Er braucht null Training und keine Hyperparameter-Anpassung pro
   Aufgabe. SuperADD tritt mit *einem* Hyperparametersatz für alle Klassen an und gewinnt [A].
9. **Empfehlungen (Details in Abschnitt 10):**
   (1) DINOv3- bzw. DINOv2-Normalitätsmodell, mehrschichtig, in nativer Auflösung mit Kacheln, ohne
   greedy Coreset;
   (2) Schutz gegen konsistente Anomalien;
   (3) ein kalibrierter Late-Fusion-Stacker vor dem 4. Eingangskanal;
   (4) Evaluationsprotokoll mit MVTec AD 2 und KSDD2 sowie Recall nach Größenklassen;
   (5) AnomalyVFM als billiges, unabhängiges drittes Review-Signal.
10. **Neue Ideen (Abschnitt 11):**
    (a) labelgeführte Burnout-Korrektur der Memory-Bank;
    (b) multiskalige konforme p-Wert-Karten mit Recall-Nachweis pro Größenklasse;
    (c) Capture-Recapture-Schätzung der von *beiden* Detektoren übersehenen Defekte als Vorhersage im
    PPI-Audit;
    (d) aufgabenspezifische LoRA-Adaption von DINOv3 mit unserer Residual-Synthese
    („AnomalyVFM pro Aufgabe“).

---

## 2. Vorbemerkung: Welche Metrik misst *unsere* Labelqualität?

| Metrik | Verhalten | Relevanz für Auto-Labels |
|---|---|---|
| P-AUROC | Wird von der riesigen Zahl leichter Normalpixel dominiert und ist saturiert. CandyCane: 97,3 bei P-AP 1,8 [R] | Nur Diagnose |
| AUPRO@0,3 | Mittelt pro Region, damit zählen kleine Defekte gleich viel. Integriert aber bis 30 % FPR, ein für Labels absurder Bereich | Gering |
| AU-PRO@0,05 | Wie oben, aber im relevanten Niedrig-FPR-Bereich (offiziell bei MVTec AD 2) | Hoch für „findet man alle Defekte?“ |
| P-AP / P-F1-max | Pixelgewichtet, große Defekte dominieren. F1-max nutzt eine Orakel-Schwelle | Mittel; optimistisch |
| SegF1 (feste Schwelle) | Schwelle muss ohne Test-GT gewählt werden, wie bei uns | **Am nächsten an unserem Use-Case** |
| Recall auf Defektebene je Größenklasse + FP-Pixel mit Toleranz | Bei uns schon implementiert (`metrics.py`) | **Primärmetrik** |

Daraus folgt: Jede Variante (Backbone, Upsampler, Fusion) wird auf **Recall auf Defektebene bei
festem Pixel-FPR, aufgeteilt nach Flächenklassen**, plus AU-PRO@0,05 und SegF1 verglichen. P-AUROC
wird nur ergänzend berichtet.

---

## 3. Self-supervised dichte Merkmale für Industrie-Anomaliesegmentierung (Thema 1)

### 3.1 DINOv3 als Backbone

* **DINOv3** (Siméoni et al., arXiv 2508.10104, Meta 2025). Patchgröße 16. Varianten [R]:
  * ViT-S/16 (21 M), S+/16 (29 M), B/16 (86 M), L/16 (300 M), H+/16 (840 M), 7B/16 (6,7 Mrd.); bis auf
    7B destilliert;
  * ConvNeXt-T/S/B/L (29–198 M), aus dem 7B-Modell destilliert;
  * Satelliten-Varianten.

  Kernidee laut Sekundärquellen ist „Gram anchoring“: Die Gram-Matrix der Patch-Features wird an einen
  früheren Lehrer gebunden, damit dichte Features bei langem Training nicht degradieren [A].
* **Lizenz:** eigene „DINOv3 License“; Gewichte nur nach Freigabe herunterladbar [R]. DINOv2 steht unter
  Apache 2.0 [R, laut AnomalyDINO-README]. Für ein Tool, das auch kommerziell eingesetzt werden könnte,
  muss die DINOv3-Lizenz **juristisch geprüft** werden; DINOv2 ist der Fallback.
* **Software:** DINOv3 braucht Python > 3.10 und PyTorch > 2.7 (Hinweis im Dinomaly-README) [R].
* **Wirkung in der AD-Literatur:**
  * Dinomaly-Autor, 08.2025: mit DINOv3 „pixel-level performance much better, slightly lower
    image-level“. DINOv3-L auf MVTec AD: I-AUROC 99,70, P-AUROC 98,78, **P-AP 74,22, P-F1 71,84,
    AUPRO 95,80** [R].
  * Ein Community-Fork mit DINOv3-B/16 kommt auf P-AP 72,9–73,4 und AUPRO 94,7–94,9 [R].
  * In Dinomaly2 erreicht DINOv3 „71.9/94.9 % P-AP/AUPRO“; die Zuordnung zu MVTec AD ist wahrscheinlich,
    aber nicht bestätigt [A].
  * AnomalyVFM (zero-shot) zeigt: DINOv3 ist auf KSDD2 besser als RADIO und DINOv2
    (P-AP 74,3 vs. 67,6/71,6), auf MPDD aber deutlich schlechter (26,1 vs. 35,7/28,7) [R].
    **DINOv3 ist also kein garantierter Gewinn pro Datensatz.**

### 3.2 Trainingsfreie Patch-kNN-Verfahren

* **AnomalyDINO** (Damm et al., WACV 2025, arXiv 2405.14529). Es nutzt DINOv2-Patch-Features und
  Nearest-Neighbor-Distanz zu einer Memory-Bank aus wenigen Normalbildern, mit optionaler Maskierung des
  Vordergrunds (PCA) und Rotations-Augmentierung. Ergebnisse:
  * 1-shot-I-AUROC auf MVTec AD steigt von 93,1 (bisher bester Wert) auf 96,6 [A];
  * die Lokalisierung verbessert sich mit der Auflösung (448 → 672 px), die Detektion kaum [A];
  * ein „batched zero-shot“-Modus ist implementiert [R].

  Relevanz: Das ist praktisch unser Normalitätsmodell mit besserem Backbone. Es bestätigt, dass
  Auflösung die Lokalisierung treibt.
* **SuperAD** (Zhang et al., arXiv 2505.19750, VAND 3.0). DINOv2, mehrstufige Features und nur
  **16 Referenzbilder pro Kategorie**, ausgewählt per Coreset auf den CLS-Embeddings. Die
  kNN-Karten pro Ebene werden gemittelt. SegF1 auf MVTec AD 2: 47,18 % (privat) und 42,51 %
  (privat, gemischte Beleuchtung) [A].
* **SuperADD** (arXiv 2605.14808, Sieger VAND 4.0, Industrie-Track) [A]:
  * DINOv3 als Extraktor mit mehrschichtigen Memory-Banks;
  * „refined subsampling“ beim Aufbau der Bank;
  * **überlappende, kachelweise Verarbeitung**;
  * morphologisches Closing als Postprocessing;
  * Intensitäts-Augmentierung gegen Beleuchtungswechsel;
  * **Schwelle = 95. Perzentil der Scores anomaliefreier Trainingsbilder**;
  * Challenge-Regel: eine Architektur und ein Hyperparametersatz für alle Klassen.

  SegF1 auf MVTec AD 2: **62,61 / 57,42 / 54,35 %** (public / private / private-mixed). Zum Vergleich der
  beste Vorjahresbeitrag (ISVL): 53,81 / 51,43 % [A].

  Relevanz: Das ist der stärkste Beleg, dass **trainingsfreies DINOv3-kNN mit Kacheln** die beste
  aufgabenagnostische Normalitätskomponente ist. Unbekannt ist, wie das „refined subsampling“ genau
  aussieht.
* **Weitere 2025/26-Arbeiten** (nur Titel bzw. Abstract gesehen):
  * „Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors“ (arXiv 2510.01934);
  * AD-DINOv3 (arXiv 2509.14084, zero-shot mit „anomaly-aware calibration“);
  * DuoAD (arXiv 2607.23924, CLS-basiert, few-shot);
  * RAID (arXiv 2602.19611, retrieval-augmentiert);
  * SubspaceAD (arXiv 2602.23013, siehe 4.3).

  Inhalte und Zahlen sind unverified.

### 3.3 Rekonstruktionsbasierte und vereinheitlichte Modelle

* **Dinomaly** (Guo et al., CVPR 2025, arXiv 2405.14325). Ein reiner Transformer-Rekonstruktor auf
  gefrorenen DINO-Features mit Noisy Bottleneck, linearer Attention und „loose reconstruction“. Ein Modell
  für alle Klassen. I-AUROC 99,6 / 98,7 / 89,3 (MVTec AD / VisA / Real-IAD); mit ViT-L 99,8 / 98,9 /
  90,1 [R]. Pixelzahlen der DINOv2-Version wurden nicht im Text gefunden. Trainiert auf einer RTX 3090
  mit 20 000 Iterationen (MVTec AD) [R].
* **Dinomaly2** (Guo et al., arXiv 2510.17611) [A, Code R]:
  * „Full-spectrum“-Framework für 2D, Multi-View, RGB-3D und RGB-IR sowie für Single-, Multi- und
    Few-Shot-Einsatz;
  * I-AUROC 99,9 (MVTec AD), 99,3 (VisA), 92,1 (Real-IAD);
  * mit 8 Normalbildern pro Klasse: 98,7 (MVTec AD) und 97,4 (VisA);
  * auf Objektebene 94,6 % AUROC auf MANTA-Tiny;
  * Preview-Code (05.2026) mit DINOv3-Unterstützung, Real-IAD Variety (160 Klassen) und ADNet
    (380 Klassen).
* **INP-Former** (Luo et al., CVPR 2025, arXiv 2503.02424). Normalprototypen werden *aus dem Testbild
  selbst* gewonnen („Intrinsic Normal Prototypes“) und steuern die Rekonstruktion; der Encoder ist
  DINOv2-reg ViT-B/14 [R]. Die Ergebnistabellen liegen nur als Bilder vor; Zahlen nicht gefunden.
  INP-Former++ (arXiv 2506.03660) erweitert das auf ein **semi-supervised** Setting [R]. Die Kernidee
  „normale Information desselben Bildes nutzen“ steht im Spannungsfeld zu unserem Ausschluss des eigenen
  Bildes (siehe 4.5).
* **MambaAD** (NeurIPS 2024, arXiv 2404.06564) [R]:

  | Datensatz | P-AUROC | P-AP | AUPRO |
  |---|---|---|---|
  | MVTec AD | 97,7 | 56,3 | 93,1 |
  | VisA | 98,5 | 39,4 | 91,0 |
  | Real-IAD | 98,5 | 33,0 | 90,5 |

  Ein lehrreiches Beispiel für die Diskrepanz zwischen P-AUROC und P-AP.
* **UniNet** (Wei et al., CVPR 2025). Kontrastiv geführte, domänenübergreifende Feature-Selektion;
  evaluiert auch auf dem *überwachten* VAD-Datensatz [A]. Zahlen nicht gefunden.
* **ArcAD** (ECCV 2026, arXiv 2607.02252), siehe 6.2: ein Kaltstart-Setting mit wenigen Normal- und
  Anomaliebildern, aufgesetzt auf Dinomaly.

**Bewertung für uns:**
* Vorteile: Rekonstruktionsmodelle haben globale Kontextinformation. Das hilft bei großflächigen oder
  logischen Anomalien (MT: *uneven*, *fray*) und liefert ein strukturell anderes Signal als kNN, also
  Diversität fürs Review.
* Kosten: Training pro Pool (laut Papers Stunden auf einer 24-GB-GPU; auf dem M4 nicht gemessen).
* Kontamination: Ein Rekonstruktor, der auf dem kontaminierten U trainiert wird, lernt wiederkehrende
  Defekte mit. Dafür gibt es keine eingebaute Robustheit wie unseren kNN-Ausschluss.
* Fazit: Option für später (GPU), nicht Priorität.

### 3.4 Auflösung, Kacheln, Feature-Upsampler

* **Primärdaten zur Auflösung:** MuSc auf MVTec AD [R]:
  * DINOv2-B/14 bei 336 → 518 px: P-AP 59,5 → 65,3, AUPRO 93,8 → 95,6;
  * DINOv2-L/14 bei 518 px: P-AP 67,4, AUPRO 96,2.

  Auf VisA (kleinere Defekte) entsprechend: DINOv2-B/14 37,1 → 41,2 P-AP [R].
* **MVTec AD 2:** RoBiS kombiniert Swin-artiges Cropping hochaufgelöster Bilder, INP-Former pro
  Kategorie, adaptive Binarisierung (mean + 3σ kombiniert mit MEBin) und SAM-Verfeinerung der Masken.
  Ergebnis: SegF1 51,00 / 46,52 % [R]. Eine Community-Studie meldet für PatchCore bei 768 statt
  640 px +0,10 AU-PRO (*can*) bzw. +0,16 (*wallplugs*) [R; nicht begutachtet].
* **Gelernte Upsampler:**
  * **FeatUp** (Fu et al., ICLR 2024);
  * **LoftUp** (Huang et al., ICCV 2025 oral, arXiv 2504.14032): koordinatenbasiert, trainiert auf
    SA-1B, per torch.hub für DINOv2-S/B, CLIP und SigLIP(2); DINOv3 laut README nicht unterstützt [R];
  * **JAFAR** (Couairon et al., arXiv 2506.11136): attentionbasiert, beliebige Zielauflösung [A];
  * **AnyUp** (Wimmer et al., ICLR 2026 oral, arXiv 2510.12764): „feature-agnostisch“, per torch.hub,
    mit Multi-Backbone-Variante [R]. UPLiFT (Walmer et al., CVPR 2026) merkt an, dass AnyUp auf einem
    Backbone trainiert ist und auf fremden schlechter abschneidet [A].
* **Evidenz für AD:**
  * SAM-LAD nutzt FeatUp auf DINOv2 und berichtet Gewinne auf MVTec LOCO. 16×/32×-Upsampling sei aber
    zu teuer [A]. Das betrifft logische Anomalien, nicht kleine Defekte.
  * Eine Materialmikroskopie-Studie (arXiv 2508.21529) findet Vorteile komplexer Upsampler genau dann,
    wenn Strukturen *kleiner als ein Patch* (z. B. Risse) unterschieden werden müssen. Bilineares
    Upsampling erzeugt Artefakte, FeatUp unscharfe Ergebnisse [A].
  * Einen direkten Vergleich auf MVTec, VisA oder KSDD2 mit Pixelmetriken haben wir **nicht gefunden**.
* **Risiko:** Bildgeführte Upsampler übernehmen Kanten aus dem RGB-Bild. Bei reflektierenden Oberflächen
  können Reflexionskanten „scharf“ in die Anomaliekarte wandern. Außerdem vergrößert die höhere
  Feature-Auflösung die Memory-Bank (×4 bis ×16) und damit die kNN-Kosten.
* **Billige Alternative (eigener Vorschlag, nicht aus der Literatur belegt):** Shift-TTA. Das Eingabebild
  wird um ½ Patch (8 px) in x und y verschoben; die vier Karten werden ausgerichtet und gemittelt. Das
  ergibt effektiv Stride 8 ohne gelernten Upsampler.

### 3.5 Größenabhängigkeit

| Größenklasse | Literatur und Mechanismus |
|---|---|
| Kleine Defekte (wenige Pixel) | Patch 14/16 begrenzt die Auflösung: Ein 2-px-Riss belegt 1/8 der Breite eines Tokens. Gegenmittel: Eingabe vergrößern oder kacheln, zusätzlich flache Schichten. Unser U-Net mit Full-Res-Stem bleibt hier unersetzlich; die Normalitätskarte dient als grober Prior |
| Große Defekte | Werden per Kachel-kNN teils übersehen, wenn die ganze Kachel betroffen ist. Diffuse, niederfrequente Abweichungen (MT *uneven*) brauchen globalen Kontext. Rekonstruktionsmodelle und herunterskalierte Ganzbildscores helfen. SuperAD adressiert explizit „large-scale anomalies“ von MVTec AD 2 [A] |
| Beides | Multiskalig mit getrennter Kalibrierung pro Skala fusionieren (Idee 11.2) |

---

## 4. Den unlabeled (kontaminierten) Pool selbst nutzen (Thema 2)

### 4.1 MuSc und MuSc-V2

* **MuSc** (Li et al., ICLR 2024, arXiv 2401.16753). Ungelabelte Testbilder bewerten sich gegenseitig
  („Mutual Scoring“). Normale Patches finden in *anderen* Bildern viele ähnliche Patches, Anomalien
  nicht. Auf Bildebene kommt RsCIN dazu, ein Re-Scoring über ein beschränktes Bildnachbarschaftsmaß.
  Ergebnisse [R]:

  | Datensatz | P-AUROC | F1-max | P-AP | PRO |
  |---|---|---|---|---|
  | MVTec AD (zero-shot) | 97,3 | 62,6 | 62,7 | 93,8 |
  | VisA | 98,8 | 48,8 | 45,1 | 92,7 |

  Mit DINOv2-L/14 bei 518 px auf MVTec AD: P-AP 67,4, PRO 96,2 [R]. Standard sind Sets aus 200 Bildern;
  ≈ 1 s/Bild mit ViT-L bei 518 px auf einer RTX 3090 [R]. Der Aufwand wächst quadratisch mit der
  Setgröße.
* **MuSc-V2** (Li et al., arXiv 2511.10047, TPAMI 2026). Erweiterung auf 3D und Multimodalität mit
  Iterative Point Grouping gegen Fehlalarme an Diskontinuitäten. Auf 2D-Datensätzen meist besser als
  MuSc [A]. Die per-Kategorie-Tabellen [R] zeigen das Metrikproblem aus Abschnitt 2 drastisch (CandyCane).

### 4.2 Konsistente Anomalien: die Achillesferse aller Mutual-Scoring-Verfahren

* **CoDeGraph** (Le Gia & Ahn, TMLR 2025, arXiv 2510.10456; Code DumBringer/CoDeGraph) [R]:
  * Problem: Wiederkehrende, ähnliche Defekte sind gegenseitig nächste Nachbarn und erscheinen normal.
  * Beobachtung „neighbor burnout“: Normale Patches gewinnen mit wachsendem k gleichmäßig Ähnlichkeit;
    konsistente Anomalien zeigen einen **abrupten Sprung**, sobald die wenigen Bilder mit ähnlichem
    Defekt „verbraucht“ sind.
  * Lösung: Graph auf Bildebene, Community Detection, Herausfiltern der betroffenen Patches aus der
    Ähnlichkeitsberechnung; theoretisch über Extremwerttheorie motiviert.
  * MVTec AD: mit ViT-L-14-336 Seg-F1 66,8 (+4,2) und AP 68,1 (+5,4) gegenüber dem besten
    Zero-Shot-Stand; mit DINOv2 **F1 69,1 (+6,5) und AP 71,9 (+9,2)**.
* **DeMuS** (Zhao et al., ECCV 2026; Code evoLonation/DeMuS) [R]:
  * Batch-Zero-Shot mit kleinen Batches (4–8 Bilder) und Posenvariation;
  * benennt den „anomaly-to-anomaly shortcut“ beim Matching;
  * gefrorenes DINOv3-L/16 plus zwei getrennt trainierte Adapter („where to match“, „how to score“),
    trainiert auf einem Hilfsdatensatz (VisA ↔ MVTec);
  * MVTec AD: P-AUROC 96,15, AUPRO 93,50; VisA: 96,94 / 92,23.
  * Bei kleinen Batches bewusst ein Trade-off; wir haben dagegen Tausende Bilder.

### 4.3 Weitere Batch- und Transduktiv-Varianten

* **AnomalyDINO, batched zero-shot** [R]: implementiert, Zahlen nicht notiert.
* **SubspaceAD** (arXiv 2602.23013): PCA-Unterraum der ungelabelten Test-Patches, Score = Residuum.
  VisA I-AUROC 94,1, MVTec 96,6; Fokus Bildebene [A].
* **Zero-Shot AD via Batch Normalization** (Li et al., NeurIPS 2023, arXiv 2302.07849) [Titel/Repo].
* **ACR** (NeurIPS 2023): MVTec zero-shot PRO 72,7 [R, laut MuSc-Tabelle].

### 4.4 Training auf kontaminierten Daten („fully unsupervised“)

* **SoftPatch** (Jiang et al., NeurIPS 2022) und **SoftPatch+** (Pattern Recognition 2025,
  arXiv 2412.20870). Patchweise Rauschdiskriminierung (LOF, Nearest oder Gauß) *vor* dem Coreset, plus
  weiche Gewichtung. MVTec AD mit 10 % zusätzlichen Rauschbildern: I-AUROC 98,65, P-AUROC 98,05 [R].
* **InReaCh** (McIntosh & Albu, ICCV 2023). Extrahiert hochkonfidente Normalpatches über „Kanäle“
  zwischen Bildrealisierungen: 99,9 % Präzision der Normalpatches auf MVTec, **robust bis 40 %
  korrumpierter Trainingsdaten**; Lokalisierungs-AUROC 0,968 [R]. Nachfolger Online-InReaCh
  (ECCV 2024) [R].
* **FUN-AD** (Im et al., WACV 2025, arXiv 2411.16110). Pseudo-Labels aus einer iterativ neu aufgebauten
  Memory-Bank plus „mutual smoothness loss“. Annahme: wechselseitig nächste Feature-Paare sind meist
  klassenhomogen [R].
* „Towards Real Unsupervised Anomaly Detection via Confident Meta-Learning“ (ICCV 2025,
  arXiv 2508.02293) [Titel].

### 4.5 Vergleich mit unserem Own-Image-Exclusion-kNN

Unser Score ist die Distanz zum k-ten nächsten Bankpatch aus einem *anderen* Bild, mit k = 3 und
64 zufälligen Patches pro Bild aus WRN50, Layer 2+3 (`anomaly.py`). Das ist strukturell ein
Mutual-Scoring über den ganzen Pool. Unterschiede und Konsequenzen:

1. **Konsistente Anomalien.**
   * Was passiert: Gibt es ≥ k andere Bilder mit ähnlichem Defekt, wird der Defekt „erklärt“. Das
     betrifft genau die *häufigen* Defekttypen. Die kennt zwar meist auch das überwachte Modell, aber
     damit verliert „unerklärte Anomalie“ seine Unabhängigkeit gerade dort, wo das U-Net schwach ist,
     etwa bei schwach sichtbaren Varianten eines häufigen Typs.
   * Gegenmittel nach Aufwand: k-Profil-Statistik (Burnout-Sprung, siehe CoDeGraph), Community-Filter,
     labelgeführte Bereinigung (Idee 11.1).
2. **Coreset-Falle.** Der übliche greedy-k-Center-Coreset (PatchCore) wählt bevorzugt *Ausreißer* aus,
   in einem kontaminierten Pool also Defektpatches. Die aktuelle Zufallsauswahl ist hier robuster.
   SoftPatch filtert deshalb *vor* dem Coreset [R]. Empfehlung: kein greedy Coreset ohne vorgeschaltete
   Rauschfilterung.
3. **Gruppenabhängigkeit.** Bei BSData (Verschleißsequenzen) oder bei Serienbildern desselben Bauteils
   erklären Nachbarbilder einen Defekt, weil es *derselbe* Defekt ist. Der Ausschluss sollte deshalb auf
   **Gruppen** (Sequenz, Bauteil-ID, zeitliche Nachbarn) erweitert werden.
4. **INP-Former-Gedanke.** Die normale Umgebung *im selben Bild* ist oft die beste Referenz (Textur,
   Beleuchtung). Ein Kompromiss (eigener Vorschlag): Patches des eigenen Bildes zulassen, aber nur mit
   räumlichem Mindestabstand r zum Query-Patch. Ein Defekt kann sich dann nicht selbst erklären, außer bei
   mehreren gleichartigen Defekten im Bild.
5. **Merkmalsqualität.** WRN50 Layer 2+3 mit 8-px-Raster ist der schwächste Teil. Siehe Empfehlung R1.

---

## 5. CLIP-, VLM- und MLLM-basierte Anomaliesegmentierung (Thema 3)

### 5.1 CLIP-Familie: echte Zahlen

Zero-shot auf MVTec AD bzw. VisA (aus der Vergleichstabelle im MuSc-README [R]; Training jeweils auf
dem *anderen* Datensatz):

| Methode | MVTec P-AUROC | MVTec F1-max | MVTec P-AP | MVTec PRO | VisA P-AUROC | VisA P-AP | VisA PRO |
|---|---|---|---|---|---|---|---|
| WinCLIP (CVPR 2023) | 85,1 | 31,7 | – | 64,6 | 79,6 | – | 56,8 |
| APRIL-GAN (VAND 2023) | 87,6 | 43,3 | 40,8 | 44,0 | 94,2 | 25,7 | 86,8 |
| AnomalyCLIP (ICLR 2024) | 91,1 | – | – | 81,4 | 95,5 | – | 87,0 |
| ACR (NeurIPS 2023) | 92,5 | 44,2 | 38,9 | 72,7 | – | – | – |
| MuSc (Batch, kein Text) | 97,3 | 62,6 | 62,7 | 93,8 | 98,8 | 45,1 | 92,7 |

* **AdaCLIP** (Cao et al., ECCV 2024). Hybride, also statische und pro Bild dynamische, lernbare Prompts.
  Laut Autoren sind vielfältige Hilfsdaten entscheidend [R]. Pixelzahlen nicht gefunden.
* **VCP-CLIP** (Qu et al., ECCV 2024). Visuelle Kontext-Prompts ohne produktspezifischen Text. „SOTA auf
  10 Industriedatensätzen“ [R]. Zahlen nicht gefunden.
* **AA-CLIP** (Ma et al., CVPR 2025, arXiv 2503.06661). Anomaliebewusste Text-Anker, dann Ausrichtung
  der Patch-Features. Pixel-AUROC 93,4 mit vollem Hilfsdatensatz, Bild-AUROC 83,1 mit 64 Shots [A].
  Das README warnt selbst: Überanpassung an die Anomaliemerkmale des Trainingssets, z. B. „runde
  Defekte“, führt zu oberflächlichen Merkmalen [R].
* **DictAS** (Qu et al., ICCV 2025, arXiv 2508.13560). Few-shot per „Dictionary Lookup“: Normale Patches
  sind aus Support-Features rekonstruierbar. Das README berichtet offen, dass prompt-gelernte
  CLIP-Modelle auf Texturen (*wood*) gut, auf Objekten (*cable*) deutlich schlechter sind [R].
* **MultiADS** (ICCV 2025, arXiv 2504.06740; Bosch). Zero-shot-Segmentierung **mehrerer Defekttypen**
  mit eigenem Benchmark; Erweiterung DAPO (ECCV 2026, arXiv 2512.09446) [R]. Zahlen nicht gefunden.
  Relevant für die Benennung von Typen, nicht für die Maskenqualität.

### 5.2 Reine Vision-Foundation-Modelle zero-shot: AnomalyVFM

**AnomalyVFM** (Fučka, Zavrtanik, Skočaj, CVPR 2026, arXiv 2601.20524):
* Synthetische Daten: FLUX erzeugt anomaliefreie Objekte, Inpainting erzeugt lokale Defekte, ein
  Feature-Differenzfilter liefert die Masken.
* Modell: LoRA in den Attention-Schichten des Backbones, leichter Conv-Decoder und ein
  konfidenzgewichteter Pixel-Loss gegen verrauschte synthetische Labels.
* Paper: Bild-AUROC 94,1 % im Mittel über 9 Datensätze, +3,3 Punkte [A].
* Repository („neue Konfiguration“, weicht vom Paper ab), Mittel über 9 Industriedatensätze [R]:

  | Backbone | P-AUROC | P-F1 | P-AP | AUPRO |
  |---|---|---|---|---|
  | RADIO | 96,3 | 44,8 | 44,1 | 89,3 |
  | DINOv3 | 95,8 | 43,9 | 43,5 | 86,9 |
  | DINOv2 | 95,4 | 42,4 | 41,4 | 85,7 |
  | CLIP | 94,6 | 41,3 | 38,8 | 86,9 |

  Pro Datensatz, P-AP mit RADIO / DINOv3 [R, `results/*/industrial/*/result.csv`]:

  | Datensatz | RADIO | DINOv3 |
  |---|---|---|
  | KSDD2 | 67,6 | **74,3** |
  | MVTec AD | 49,9 | 50,2 |
  | VisA | 26,7 | 26,3 |
  | MPDD | 35,7 | 26,1 |
  | Real-IAD | 36,5 | 34,5 |
  | DAGM | 66,6 | 64,5 |

* Einordnung: AnomalyVFM ist der stärkste *textfreie* Zero-Shot-Ansatz, liegt auf MVTec und VisA aber
  klar unter dem Batch-Verfahren MuSc (62,7 / 45,1). Der KSDD2-Wert ist bemerkenswert hoch. Er ist mit
  SuperSimpleNet (Abschnitt 6.2) aber wegen anderer Protokolle **nicht direkt vergleichbar**.
* Nutzen für uns: ohne Training per torch.hub nutzbar [R]. Die Fehler sind vermutlich wenig korreliert
  mit unserem Ensemble, weil Trainingsdaten und Prinzip andere sind. Das macht es zu einem Kandidaten
  für ein drittes, unabhängiges Review-Signal. Die Lizenz (RADIO, FLUX-Daten) ist ungeprüft.

### 5.3 Multimodale LLMs

* **AnomalyGPT** (Gu et al., AAAI 2024). PandaGPT/Vicuna-7B, simulierte Anomalien, leichter Decoder
  für die Lokalisierung. Lizenz CC BY-NC-SA 4.0 [R]. Zahlen nicht im Text gefunden.
* **Myriad** (2023/24): Experten-Prior aus Vision-AD-Modellen im MLLM (unverified, Repo nicht
  erreichbar).
* **MMAD** (Jiang et al., ICLR 2025, arXiv 2410.09453): 7 Teilaufgaben, 39 672 Fragen, 8 366 Bilder.
  Bestes Modell GPT-4o mit **74,9 %** Accuracy, Gemini-1.5-pro ≈ 73 %, InternVL2-76B 70,8 %.
  AD-Forscher sind ≈ 12 Punkte besser als die Modelle, die größte Lücke liegt bei Anomalie- und
  Defektfragen [A].
* **Reasoning-MLLMs 2025/26:**
  * AnomalyR1 (arXiv 2504.11914): Qwen2.5-VL-3B mit GRPO. Lokalisierung nur indirekt: Bounding Box,
    dann **SAM** für die Maske [A].
  * OmniAD (arXiv 2505.22039): GRPO mit Lokalisierung auf Patch-Ebene [A].
  * IAD-R1 (arXiv 2508.09178, AAAI 2026): PA-SFT plus SC-GRPO; Detektion bzw. Erklärung, keine
    Segmentierung [A].
  * „Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision“ (CVPR 2026)
    stellt fest, dass reine MLLM-Ansätze bei präziser Pixel-Lokalisierung weiterhin scheitern [A].
* **VAND 4.0 (Retail, Kaputt 2):** Das beste Standard-VLM liegt ≈ 28 AP hinter spezialisierten
  Detektoren. Seltene Defekte: Spillage ≈ 53 AP, fehlende Einheiten ≈ 27 AP [A].

### 5.4 SAM-Familie

* **SAM 3** (ICLR 2026, arXiv 2511.16719): „Promptable Concept Segmentation“, SA-Co mit 207 000
  Konzepten [A].
  * Meta selbst: auf Roboflow100-VL zero-shot *unter* SOTA, few-shot und nach Fine-Tuning darüber [A].
  * Eine Drittstudie (Fernerkundung) nennt 56,4 AP auf COCO gegenüber 15,2 AP zero-shot auf
    RF100-VL [A].
  * Eine Evaluation auf industriellen Defekten haben wir nicht gefunden.
* **Praxis:** RoBiS nutzt SAM nur zur *Verfeinerung* schon detektierter Masken [R]. AnomalyR1 nutzt
  SAM für Box → Maske [A].

### 5.5 Fazit zu Fehlermodi und Nutzen

* **Wo VLMs scheitern:**
  * kleine, kontrastarme Defekte, weil CLIP-Patch-Tokens grob sind und Text für Mikrodefekte kaum
    trennscharf ist;
  * Objektkategorien mit Strukturvariation (*cable*);
  * Domänen ohne Trainingsabdeckung.
  * Hinzu kommt: Schwellen sind nicht kalibriert, ohne Pixel-Kalibrierung entsteht kein verwertbarer
    SegF1.
* **Wo sie helfen:**
  1. großflächige, „semantische“ Defekte (fehlende Teile, Fremdkörper);
  2. **Benennung** neu entdeckter Defekt-Cluster und Textberichte im Review;
  3. ein unabhängiges, billiges Zusatzsignal (AnomalyVFM).

---

## 6. Few-Shot-*supervised* und Open-Set-supervised AD (Thema 4)

### 6.1 Klassische Open-Set-Verfahren (OSAD)

* **DevNet** (Pang et al., KDD 2019; arXiv 2108.00462): Deviation-Loss auf einem Gauß-Prior. Definiert
  das Protokoll „trainiere auf einer Anomalieklasse, teste auf den übrigen“ [A].
* **DRA** (Ding, Pang, Shen, CVPR 2022): getrennte Köpfe für gesehene Anomalien, Pseudo-Anomalien und
  latente Residuen-Anomalien [A].
* **BGAD** (Yao et al., CVPR 2023, arXiv 2207.01463): Normalverteilung per Normalizing Flow, explizite
  Grenze, „semi-push-pull“-Loss. Code mit `--num_anomalies 10` [R]. Pixelzahlen nicht gefunden.
* **PRN** (Zhang et al., CVPR 2023, arXiv 2212.02031): multiskalige **Prototypen der Residuen-Features**
  (Testfeature minus Normalprototyp) und Multi-Size-Self-Attention. Behauptet auf MVTec PRO
  +2,2 Punkte gegenüber dem unüberwachten und +22,8 gegenüber dem überwachten Stand [A].
* **AHL** (Zhu et al., CVPR 2024, arXiv 2310.12790): simuliert heterogene Anomalieverteilungen, als
  Plug-in für OSAD-Modelle; 9 Datensätze [A].
* **DPDL** (CVPR 2025, arXiv 2502.20981) und **MPFM** (ICML 2026, arXiv 2605.02438) [Titel].
* **IDEAL** (Wang, Shen, Yan, Pang, arXiv 2605.23231): „Discriminative Few-Shot AD“ mit normalen *und*
  anomalen Referenzen zur Inferenzzeit; Normal Variation Eraser plus Intrinsic Deviation Encoder;
  8 Datensätze, Code angekündigt [A].
* **Abdali et al.** (arXiv 2507.23712, IJCNN-Workshop 2025): Anomalie-Samples in Multi-Score-Fusion mit
  augmentierungsbasierter Validierung der Gewichte [A].

Gemeinsamer Befund: fast ausschließlich Bild-AUROC/AP, meist mit Bildlabels. **Belastbare
Pixelsegmentierung mit ≈ 10 Anomalien und mehreren Typen ist kaum untersucht.** VisA fasst alle
Defekte als „bad“ zusammen und taugt daher nicht für Experimente mit ungesehenen Typen [A].

### 6.2 Gemischte Überwachung und Segmentierung mit wenigen Masken

* **Božič et al.** (Computers in Industry 2021, arXiv 2104.06064), KSDD2, Bild-AP: 73,3 nur mit
  Bildlabels, **83,2 mit 16 pixelgenauen Masken**, 95,4 mit allen [A].
* **SuperSimpleNet** (Rolih et al., ICPR 2024, arXiv 2408.03143; Erweiterung „No Label Left Behind“,
  JIMS 2025, arXiv 2508.19060). Eine Architektur für alle Überwachungsregime mit synthetischen
  Anomalien. KSDD2 aus `paper_results/jims/results_avg_std.json` [R]:

  | Regime | Pixel-AP | AUPRO | P-AUROC | Bild-AP |
  |---|---|---|---|---|
  | unüberwacht | 45,8 | 91,7 | 94,3 | 55,8 |
  | nur Bildlabels | 47,6 | 87,8 | 89,1 | 97,2 |
  | 16 Masken + Bildlabels | 59,1 | 91,2 | 92,1 | 97,3 |
  | 53 Masken | 68,0 | 96,3 | 96,0 | 97,7 |
  | 126 Masken | 76,5 | 98,2 | 98,7 | 97,4 |
  | alle 246 Masken | **81,3** | 98,8 | 99,4 | 97,8 |

  Zum Vergleich unüberwacht auf MVTec AD: P-AP 49,6, AUPRO 91,2; auf VisA: 35,2 / 87,4 [R].

  **Lehre für uns:**
  1. Pixelmasken steigern P-AP monoton und deutlich; der Abstand von unüberwacht zu voll überwacht
     beträgt ≈ 35 Punkte.
  2. Auch die Detektion profitiert stark von echten Anomalien (Bild-AP 55,8 → 97,2).
  3. Unser Regime (5–30 Masken, *keine* Bildlabels für U) liegt dazwischen. Self-Training und
     konforme Recall-Schwellen sind genau die Mittel, um die fehlenden Bildlabels zu ersetzen.
* **SegAD** (Baitieva et al., CVPR 2024, arXiv 2405.04953). Valeo-Datensatz VAD: 5 000 Bilder,
  ≈ 2 000 defekt, > 20 Subklassen, ungesehene Defekte im Test. Ansatz: **Anomaliekarten plus
  Segmentierungskarten → lokale Statistiken → Boosted Random Forest**. +2,1 % AUROC auf VAD, nur
  Bildebene [A]. Das ist ein Beleg für *Late Fusion* von Anomaliekarten mit überwachten Signalen.
* **ArcAD** (ECCV 2026, arXiv 2607.02252) [R]. Kaltstart: wenige Normalbilder plus wenige Anomalien
  *mit Masken*, auf MVTec 1 089 Normal- und 121 Anomaliebilder, also ≈ 8 pro Kategorie. Aufgesetzt auf
  Dinomaly mit DINOv2-reg-B/14 als Plug-in: Hypersphären-Prototypen plus defektgeführte kontrastive
  Kalibrierung. Multi-Klasse (I-AUROC / P-AUROC / P-F1-max):

  | Datensatz | I-AUROC | P-AUROC | P-F1-max |
  |---|---|---|---|
  | MVTec AD | 99,7 | 99,2 | 68,9 |
  | VisA | 98,9 | 99,0 | 54,9 |
  | Real-IAD | 92,5 | 99,0 | 49,8 |
  | MANTA | 93,3 | 95,5 | 48,5 |

  Eine Vergleichszahl ohne Anomalien im selben Kaltstart-Split fehlt im README, der Gewinn durch die
  Anomalien ist also nicht bezifferbar.

### 6.3 Mehrere Defekttypen und ungesehene Typen

* **AnomalyNCD** (Huang et al., CVPR 2025, arXiv 2410.14379) [R]. „Novel class discovery“ für
  Anomalien. MEBin (Main Element Binarization) liefert anomaliezentrierte Ausschnitte,
  maskengeführtes Repräsentationslernen und korrigierte Pseudo-Labels folgen. Klassifikation auf
  Regions- und Bildebene, kombinierbar mit beliebigen AD-Methoden. Direkt relevant als Erweiterung
  unseres Few-Shot-Typklassifikators: **Kandidaten ohne Zuordnung clustern, um neue Klassen
  vorzuschlagen.**
* **MultiADS / DAPO:** Typzuordnung zero-shot per defektbewusster Prompts, siehe 5.1.

### 6.4 Was funktioniert mit ≈ 10 Anomalien? Synthese

Die Literatur konvergiert auf drei Komponenten. Wir haben zwei davon schon:

1. **Gesehene Typen:** überwachter Segmentierer mit realistischer Synthese aus den echten Defekten.
   Bei uns: Residual-Copy-Paste.
2. **Ungesehene Typen:** ein Normalitäts- bzw. Residuensignal (DRA-„latent residual“, PRN-Residuen,
   BGAD-Grenze). Bei uns: Normalitätsmodell, bisher nur als Review-Signal.
3. **Fehlt noch:** eine **gelernte Fusion**, die das Residuensignal dort einsetzt, wo der Segmentierer
   unsicher ist, ohne seine Präzision auf gesehenen Typen zu verschlechtern. PRN und DRA tun das im
   Netz, SegAD per Booster. Für uns ist die Booster- bzw. Stacker-Variante attraktiver: Sie lässt sich
   auf OOF-Daten ehrlich kalibrieren (R3).

---

## 7. Universelle Multi-Klassen-Modelle und Aufgaben-Adaption (Thema 5)

* **Ein Modell für viele Klassen** ist 2025/26 Stand der Technik:
  * Dinomaly mit I-AUROC 99,6 als Multi-Klasse [R];
  * Dinomaly2 mit 160- bzw. 380-Klassen-Datensätzen [R];
  * INP-Former „super multi-class“ [R];
  * UniVAD (Gu et al., CVPR 2025, arXiv 2412.03342): trainingsfrei, few-shot, mit
    Komponenten-Clustering, Komponenten-Patch-Matching und Graph-Modellierung über Industrie, Logik und
    Medizin [A];
  * ArcAD als Multi-Klassen-Kaltstart [R].
* **Für unser Tool entscheidend** ist weniger „ein Modell für alle Produkte“ als **Anpassungsaufwand
  nahe null pro neuer Aufgabe**. Das leisten:
  * trainingsfreie kNN-Verfahren auf gefrorenem Backbone (AnomalyDINO, SuperAD/SuperADD): Bank pro
    Aufgabe aufbauen, fertig;
  * eine einheitliche Hyperparametrierung über Klassen (VAND-4.0-Regel; SuperADD gewinnt so) [A];
  * Schwellen aus der Normalverteilung (95. Perzentil) [A], die sich bei uns konform über den
    unlabeled Pool und die OOF-Daten kalibrieren lassen.
* Kontinuierliche bzw. inkrementelle Multi-Task-AD (UCAD, CDAD u. a.) haben wir aus Budgetgründen nicht
  geprüft (unverified).
* **Domänenadaption pro Aufgabe:** AnomalyVFM zeigt, dass LoRA auf einem VFM mit synthetischen
  Defekten generisch funktioniert [R]. Die naheliegende Weiterentwicklung (Idee 11.4) ist dieselbe
  Adaption, aber mit **aufgabeneigener** Synthese, also unseren Residuen echter Defekte auf echten
  Hintergründen.

---

## 8. Benchmarks mit harten Defekten (Thema 6)

| Benchmark | Charakter | Beste gefundene Zahlen (Familie) |
|---|---|---|
| **MVTec AD 2** (Heckler-Kram et al., arXiv 2503.21622) | 8 Szenarien, > 8 000 hochaufgelöste Bilder; Transparenz, Reflexion, Beleuchtungswechsel, sehr kleine und große Defekte. Privates Testset, SegF1 bei einer Schwelle | Unüberwachtes DINOv3-kNN: SuperADD **57,42** SegF1 privat [A]. INP-Former + SAM: RoBiS 51,00 [R]. DINOv2-kNN: SuperAD 47,18 [A]. Baselines AU-PRO@0,30: EfficientAD 58,7, RD++ 54,3, PatchCore 53,8, RD 53,0, MSFlow 52,7, SimpleNet 46,4 [A, Spaltenzuordnung aus Kontext]. Zero-shot ≈ 15 SegF1 unter dem besten Beitrag [A] |
| **Real-IAD** (Wang et al., CVPR 2024, arXiv 2403.12580) | 30 Klassen, 151 050 Bilder, 2 000–5 000 px, Defektflächenanteil 0,01–6,75 % [R, ADer-README]; dazu D3- und Variety-Versionen | Dinomaly2 I-AUROC 92,1 [A]. MambaAD P-AP 33,0 [R]. AnomalyVFM zero-shot P-AP 36,5 [R]. ArcAD P-F1-max 49,8 [R] |
| **VisA** | 12 Klassen, kleine Defekte, Typen zu „bad“ zusammengefasst | MuSc P-AP 45,1 / PRO 92,7 [R]. MambaAD P-AP 39,4 [R]. AnomalyVFM 26,7 [R]. APRIL-GAN 25,7 [R]. ArcAD P-F1-max 54,9 [R] |
| **KolektorSDD2** | 3 335 Bilder, kleine Defekte, Produktionsdaten | SuperSimpleNet P-AP: 45,8 unüberwacht → 59,1 mit 16 Masken → 81,3 voll [R]. AnomalyVFM zero-shot 67,6–74,3 (anderes Protokoll) [R] |
| **MPDD** | Metallteile, Pose und Beleuchtung | AnomalyVFM P-AP 35,7 (RADIO) bzw. 26,1 (DINOv3) [R] |
| **MANTA** (CVPR 2025, arXiv 2412.04867) | sehr kleine Objekte, Multi-View, Text | ArcAD P-F1-max 48,5 [R]. Dinomaly2 94,6 Objekt-AUROC (MANTA-Tiny) [A] |
| **3CAD** (AAAI 2025, arXiv 2502.05761) | 3C-Produkte, reale Defekte | Keine Zahlen gefunden |
| **MVTec AD** (Kontrollwert, saturiert) | – | Dinomaly-DINOv3-L P-AP 74,2 [R]. CoDeGraph (Batch, DINOv2) P-AP 71,9 [R]. MuSc DINOv2-L 67,4 [R]. ArcAD P-F1-max 68,9 [R] |

Für **Magnetic Tile, BSData, DeepCrack und CrackForest** haben wir keine Foundation-Model- bzw.
AD-Ergebnisse mit Pixelmetriken gefunden. Wir müssen sie selbst erzeugen.

---

## 9. Vergleichstabelle der relevantesten Methoden

| Methode (Jahr) | Familie | Nutzt U-Pool? | Robust gegen Kontamination | Kleine Defekte | Große Defekte | Mehrere Typen | Kosten / Aufwand | Kennzahl (Datensatz, Metrik) | Relevanz für uns |
|---|---|---|---|---|---|---|---|---|---|
| Unser kNN (WRN50, Ausschluss eigenes Bild) | kNN | ja (ganzer Pool) | mittel (k = 3; konsistente Anomalien offen) | schwach (8-px-Raster, CNN) | mittel | nein | gering | MT: nicht gemessen | Basis |
| AnomalyDINO (2025) | kNN, DINOv2 | optional (batched) | nein | gut mit hoher Auflösung | mittel | nein | trainingsfrei | MVTec 1-shot I-AUROC 96,6 [A] | Vorlage für R1 |
| SuperADD (2026) | kNN, DINOv3, Kacheln | nein (Normalbilder) | nein | **gut** (Kacheln) | gut | nein | trainingsfrei; ein Hyperparametersatz | AD2 SegF1 57,4 [A] | **Vorlage für R1** |
| Dinomaly(-2) (2025) | Rekonstruktion, DINOv2/3 | Training auf Normalbildern | gering | gut | gut (global) | nein | Training (GPU) | MVTec P-AP 74,2 (v3-L) [R] | zweites Normalitätssignal (R11) |
| INP-Former(++) (2025) | Rekonstruktion, Prototypen aus dem Bild | ++: semi-supervised | gering | gut | gut | nein | Training | AD2 via RoBiS 51,0 [R] | Idee Self-Reference |
| MambaAD (2024) | Rekonstruktion, SSM | Training | gering | mittel | gut | nein | Training | Real-IAD P-AP 33,0 [R] | gering |
| MuSc / MuSc-V2 (2024/26) | Batch-Mutual-Scoring | **ja** | mittel | gut (518 px) | mittel | nein | O(N²) pro Set | MVTec P-AP 62,7–67,4 [R] | entspricht unserem Ansatz |
| CoDeGraph (2025) | Batch + Community-Filter | **ja** | **hoch** (konsistente Anomalien) | gut | mittel | nein | moderat | MVTec P-AP 71,9 [R] | **R2 / Idee 11.1** |
| DeMuS (2026) | Batch + Adapter, DINOv3 | ja (kleine Batches) | mittel | gut | mittel | nein | Adapter-Training auf Hilfsdaten | MVTec AUPRO 93,5 [R] | gering (wir haben große Pools) |
| SoftPatch(+), InReaCh, FUN-AD | Rauschrobustes Training | ja | **hoch** (bis 40 % [R]) | mittel | mittel | nein | gering bis moderat | MVTec P-AUROC 98,05 bei 10 % Rauschen [R] | Filter vor der Bank (R1) |
| WinCLIP / APRIL-GAN / AnomalyCLIP / AA-CLIP | CLIP + Text | nein | – | **schwach** | mittel | über Text | Hilfsdaten-Training | MVTec P-AP 40,8 (APRIL-GAN) [R] | nicht für Labels |
| AnomalyVFM (2026) | VFM + LoRA, synthetische Daten | nein | – | mittel | mittel | nein | torch.hub, kein Training | 9 Datensätze P-AP 44,1; KSDD2 74,3 [R] | drittes Review-Signal (R6) |
| MLLMs (AnomalyGPT, AnomalyR1, OmniAD, IAD-R1) | Sprachmodell | nein | – | schwach | mittel | ja (Text) | groß | MMAD Acc ≤ 74,9 % (GPT-4o) [A] | nur Benennung und Berichte |
| SAM 3 (2026) | promptbare Segmentierung | nein | – | schwach ohne Prompt | gut als Refiner | Konzepte | groß | keine Defekt-Evaluation gefunden | Review-UI (R9) |
| SuperSimpleNet (2024/25) | überwacht + synthetisch, alle Regime | Bildlabels | – | gut | gut | nein | Training | KSDD2 P-AP 59,1 (16 Masken) [R] | Referenz-Baseline |
| SegAD (2024) | Late Fusion mit Booster | – | – | – | – | Subklassen | gering | VAD +2,1 AUROC [A] | **Vorlage für R3** |
| DRA / BGAD / PRN / AHL / IDEAL | Open-Set-supervised | teils | – | PRN: Residuen multiskalig | – | ungesehene Typen | Training | meist nur Bildebene | Designprinzipien |
| ArcAD (2026) | Kaltstart-supervised + Dinomaly | nein | – | gut | gut | – | Training | MVTec P-F1-max 68,9 [R] | Benchmark-Protokoll |
| AnomalyNCD (2025) | Clustering von Anomalietypen | ja | – | – | – | **ja** | gering | Zahlen nicht notiert | Typentdeckung (R10) |

---

## 10. Konkrete, priorisierte Empfehlungen für unsere Pipeline

Die erwarteten Gewinne sind **Hypothesen aus der Literatur**, keine gemessenen Ergebnisse auf unseren
Daten. Jede Maßnahme wird mit dem Protokoll aus R5 gemessen.

| Rang | Maßnahme | Erwarteter Gewinn | Aufwand | Risiko |
|---|---|---|---|---|
| **R1** | **Normalitätsmodell auf DINOv3 umstellen** (Fallback DINOv2-reg). ViT-B/16 als Standard, ViT-S/16 für schnelle Läufe. 2–3 Zwischenschichten L2-normalisiert und konkateniert, Kosinusdistanz. **Native Auflösung mit überlappenden Kacheln** (z. B. 448–672-px-Kacheln, 50 % Überlappung) plus optional Shift-TTA für effektiven Stride 8. Ausschluss des eigenen Bildes und k beibehalten. Bank mit mehr Patches pro Bild, aber *Zufalls*- oder SoftPatch-gefiltertes Subsampling statt greedy Coreset. FAISS für kNN | Größter Hebel für das Signal „übersehener Defekt“ und für die Auswahl sauberer Hintergründe. Literatur: DINO-kNN mit Kacheln dominiert MVTec AD 2 [A]; DINOv2-L statt CLIP-L in MuSc +4,7 bis +5,1 P-AP [R] | 2–3 Tage (`anomaly.py`, `create_encoder`) | DINOv3-Lizenz und Gating; Laufzeit auf MPS ungemessen; Bankgröße. Gegenmaßnahmen: DINOv2-Fallback, ViT-S, Patch-Budget |
| **R2** | **Schutz gegen konsistente Anomalien:** (a) Ausschluss auf *Gruppen* erweitern (Sequenz, Bauteil, zeitliche Nachbarn); (b) k-Profil speichern (d₁…d_K, K ≈ 16) und einen Burnout-Score (größter relativer Sprung) als zweite Statistik ausgeben; (c) optional der CoDeGraph-Community-Filter | Höherer Recall des Normalitätssignals für *häufige* Defekttypen; besseres „P(übersehener Defekt)“ | 1 Tag für (a)+(b), 3 Tage für (c) | Gering. Burnout-Statistik kann bei sehr homogenen Pools rauschen |
| **R3** | **Late-Fusion-Stacker** auf Regionsebene vor dem 4. Kanal. Merkmale: max/mean der U-Net-Wahrscheinlichkeit, Normalitäts-p-Wert (konform aus U kalibriert), Fläche, Kontrast, Burnout-Score, Ensemble-Uneinigkeit. Isotone bzw. logistische Regression oder kleiner GBM. Training auf OOF-Kandidaten plus synthetischen Pastes mit bekanntem Label (SegAD-Prinzip [A]) | Recall bei festem FP-Budget, bessere Kalibrierung von P(echter Defekt), mehr Recall für ungesehene Typen | 1–2 Tage (`quality.py`, `pipeline.py`) | Gering. Bleibt OOF-ehrlich, die U-Net-Gewichte bleiben unberührt |
| **R4** | **4. Eingangskanal bzw. Residuen-Features (PRN-artig)** ins U-Net. Bedingungen: Karte für Synthese-Crops *nach* dem Einfügen neu berechnen, Channel-Dropout (p ≈ 0,3), DINO-Karte auf Crop-Auflösung | Möglicher Gewinn bei kontrastarmen Defekten. Keine direkte Pixel-Ablation in der Literatur gefunden, also Experiment | 3–5 Tage | **Mittel bis hoch:** Das U-Net und „unerklärte Anomalie“ werden *abhängig*, das Review-Signal verliert Wert. Abhilfe: Für das Review ein *zweites*, nicht eingespeistes Normalitätsmodell nutzen (anderer Backbone) |
| **R5** | **Evaluationsprotokoll:** Loader für MVTec AD 2 (SegF1, AU-PRO@0,05; Testset `test_public` lokal) und KSDD2. Alle Metriken zusätzlich **nach Defektfläche geschichtet** (z. B. < 0,01 %, 0,01–0,1 %, 0,1–1 %, > 1 % der Bildfläche). P-AUROC nur als Diagnose | Vergleichbarkeit mit der Literatur; ehrliche Aussagen für „alle Größen“ | 1–2 Tage (`data.py`, `metrics.py`) | Gering |
| **R6** | **AnomalyVFM als drittes, unabhängiges Review-Signal** (torch.hub, kein Training) im Rang-Mittel der Review-Priorität bzw. im Stacker (R3) | Höherer Recall in den ersten 5–20 % der Review-Queue, vor allem bei ungesehenen Typen | 0,5–1 Tag | Lizenz (RADIO/FLUX) ungeprüft; Domänenlücke (VisA P-AP nur 27 [R]) |
| **R7** | **Schwellen für Normalitätskandidaten:** konforme p-Werte gegen das Patch-Score-Pool des sauberen U-Subsets (wie die Hintergrundauswahl); alternativ MEBin (AnomalyNCD/RoBiS) [R] oder Normal-Quantil (SuperADD: 95 %) [A] | Kandidatengenerierung mit kontrollierter FPR, aufgabenagnostisch | 0,5 Tag | Kontaminiertes Kalibrierset: Quantile nur aus Bildern mit niedrigem Score nehmen (zirkulär, Effekt messen) |
| **R8** | **Feature-Upsampler** (AnyUp Multi-Backbone, LoftUp für DINOv2) *erst nach R1*, nur für die Kleindefekt-Klassen bewerten | Unklar; Literatur für AD dünn [A] | 1–2 Tage | Reflexionskanten werden „scharf“; Bank ×4–16 |
| **R9** | **SAM 2/3 als Kantenvorschlag** für große Defekte nur im Review-UI bzw. für Ignore-Ringe, nie direkt als Label | Weniger Korrekturaufwand bei großen Defekten | 1–2 Tage | Domänenschwäche zero-shot [A] |
| **R10** | **Typentdeckung:** AnomalyNCD-artiges Clustern der „unerklärten“ Kandidaten über den DINO-Features; Cluster dem Reviewer als mögliche neue Klasse anbieten; optional MLLM für einen Namensvorschlag | Bessere Mehrklassen-Labels, frühes Erkennen neuer Typen | 2–3 Tage | Gering, rein beratend |
| **R11** | **Zweites, strukturell anderes Normalitätsmodell** (Dinomaly2 oder INP-Former auf den saubersten U-Bildern) erst bei GPU-Verfügbarkeit | Diversität im Review-Signal, große und logische Defekte | 3–5 Tage plus Rechenzeit | Lernt wiederkehrende Defekte aus U mit; Kosten |

**Ausdrücklich nicht empfohlen:**
* CLIP-Prompt-basierte Zero-Shot-Segmentierung als Labelquelle;
* MLLM-Masken oder MLLM-Entscheidungen über Labels;
* greedy Coreset auf dem kontaminierten Pool;
* Modellvergleiche nur über P-AUROC.

---

## 11. Neue Forschungsideen (über die Literatur hinaus)

### 11.1 Labelgeführte Burnout-Korrektur der Memory-Bank

* **Idee:** CoDeGraph und SoftPatch erkennen Kontamination *ohne* Labels. Wir haben aber 5–30 echte
  Defektmasken. Aus deren Patches entsteht eine Defektbank D (plus Augmentierungen, plus
  Residual-Pastes auf U-Hintergründen). Jeder Bankpatch b aus U bekommt den Score
  s(b) = d_D(b) / d_N(b), wobei d_N die k-NN-Distanz zur Normalbank ohne b's eigenes Bild ist. Patches
  mit kleinem s, die also näher an bekannten Defekten liegen als an Normalität, werden iterativ aus der
  Normalbank entfernt. Dazu kommt der Burnout-Score aus R2.
* **Warum das bei uns besser sein könnte:** Die Lücke betrifft konsistente Anomalien *bekannter* Typen.
  Genau die können wir gezielt entfernen. Unbekannte Typen bleiben über die unüberwachte
  Burnout-Statistik abgedeckt. Labelfreie Verfahren müssen dagegen Communities raten. Bei schwachen,
  aber häufigen Defekten (MT *blowhole*, BSData Pitting) sollte der gezielte Filter mehr Bankkontamination
  entfernen.
* **Falsifizierbares Experiment:** MT und BSData, 3 Seeds, gleiche Splits wie im Benchmark-Runner.
  * Varianten: (i) aktuelles kNN mit k = 3, (ii) + Burnout-Score, (iii) + CoDeGraph-artiger Filter,
    (iv) + labelgeführte Bereinigung, (v) (iii)+(iv).
  * Metriken: AU-PRO@0,05 und Recall auf Defektebene bei Pixel-FPR 1 % auf U, **getrennt nach
    Defekttypen in L und Typen nicht in L** (für MT einen Typ aus L entfernen).
  * Hypothese verworfen, wenn (iv) gegenüber (iii) bei Typen aus L weniger als +2 Punkte Recall bringt
    oder bei Typen außerhalb von L mehr als 1 Punkt verliert.

### 11.2 Multiskalige konforme p-Wert-Karten mit Recall-Nachweis pro Größenklasse

* **Idee:** Pro Skala s (z. B. Kacheln in nativer Auflösung für kleine Defekte, ganzes Bild auf 1/2 bzw.
  1/4 skaliert für große) wird der rohe kNN-Score in einen **konformen p-Wert** umgerechnet. Basis ist
  die empirische Verteilung der Patch-Scores ohne eigenes Bild auf dem saubersten U-Subset. Die Skalen
  werden mit einer validen Kombinationsregel fusioniert (Bonferroni-Minimum oder
  Cauchy-Kombinationstest). Das Ergebnis ist eine Pixelkarte mit *interpretierbarer* Fehlalarmrate statt
  willkürlicher Score-Einheiten. Unsere Split-Conformal-Recall-Schwelle wird zusätzlich **pro
  Größenklasse** ausgewiesen, also eine Mondrian-konforme Garantie, soweit die Defektzahl reicht.
* **Warum das bei uns besser sein könnte:** Maximum oder Mittelwert roher Multiskalen-Scores
  bevorzugt die Skala mit der größten Score-Varianz. Kleine Defekte werden dann von der großen Skala
  überstimmt und umgekehrt (genau das MT-Problem: dünne Risse plus *uneven*). Auf p-Wert-Ebene fusioniert
  dagegen jede Skala im gleichen Maßstab. Und „Recall pro Größenklasse“ beantwortet die
  Anwenderfrage, ob auch kleine Defekte gefunden werden, statt eines Mittelwerts, den große Defekte
  dominieren.
* **Falsifizierbares Experiment:** MT, BSData, KSDD2, MVTec AD 2 (`test_public`).
  * Varianten: eine Skala (Kachel), eine Skala (global), Maximum der z-normierten Rohscores,
    p-Wert-Fusion.
  * Metriken: Recall auf Defektebene bei Pixel-FPR 1 % pro Flächenklasse sowie die Kalibrierung der
    p-Werte (Anteil normaler GT-Pixel mit p < α soll ≤ α + Toleranz sein).
  * Hypothese verworfen, wenn die p-Wert-Fusion nicht gleichzeitig in der kleinsten *und* der größten
    Flächenklasse mindestens so gut ist wie die beste Einzelskala bzw. das Rohmaximum oder wenn die
    p-Wert-Kalibrierung auf normalen Pixeln um mehr als 2 Prozentpunkte verfehlt wird.

### 11.3 Capture-Recapture-Schätzung der übersehenen Defekte

* **Idee:** Das überwachte Ensemble (Detektor A) und das Normalitätsmodell (Detektor B) sind zwei
  „Fänge“ derselben Defektpopulation in U. Mit n_A, n_B und n_AB (Defekte bzw. Regionen, die von A, B
  bzw. beiden gefunden werden; gematcht über Überlapp) schätzt der Lincoln-Petersen- bzw.
  Chapman-Schätzer N̂ = (n_A+1)(n_B+1)/(n_AB+1) − 1 die Gesamtzahl der Defekte. Damit ist
  N̂ − |A∪B| die Zahl der Defekte, die **beide** übersehen.
  * Die klassische Verletzung der Annahmen ist Heterogenität (kleine, kontrastarme Defekte sind für
    beide schwer).
  * Sie wird durch Stratifizierung nach Größe und Kontrast korrigiert. Die Fangwahrscheinlichkeiten pro
    Stratum werden **mit unserer Residual-Copy-Paste-Synthese** gemessen: Defekte mit bekannter Größe
    und Gain auf saubere U-Hintergründe einfügen, dann erfassen, welche A bzw. B findet.

  Die Schätzung „Defekte, die beide übersehen“ geht als Vorhersage f_i in den PPI-Differenzschätzer
  ein. Capture-Recapture ist z. B. in Software-Inspektionen etabliert. Eine Anwendung auf
  Pixel-Auto-Labeling mit zwei heterogenen Detektoren haben wir nicht gefunden.
* **Warum das bei uns besser sein könnte:** Unsere jetzige Selbsteinschätzung leitet erwartete FN nur
  aus *A's eigenen* kalibrierten Wahrscheinlichkeiten ab. Defekte, die A gar nicht „sieht“, tragen
  dort fast nichts bei. Ein zweiter, strukturell anderer Detektor liefert genau diese fehlende
  Information. Bessere f_i verengen die PPI-Intervalle. **[Korr. 07 F5]** Die PPI-Intervalle sind nur approximativ
  (t-Quantil, Delta-Methode); bei kleinen Audits und seltenen Defekten senkt ein schlechtes f_i auch ihre
  Abdeckung. Gültig im designbasierten Sinn ist nur ein Audit mit exakter Schranke oder Betting mit
  prädiktablen Größen unter den Bedingungen aus 07 §7 Nr. 7.
* **Falsifizierbares Experiment:** Auf MT, KSDD2 und BSData (volle GT für U vorhanden) wird die wahre
  Zahl der von A *und* B übersehenen Defekte verglichen mit (i) dem jetzigen erwarteten FN aus A,
  (ii) dem unstratifizierten Chapman-Schätzer und (iii) dem stratifizierten Schätzer mit synthetischer
  Kalibrierung.
  * Metriken: Bias und RMSE über 3 Seeds × 3 Labelbudgets sowie die Breite der PPI-Intervalle für
    Recall bei n = 20/40 Audit-Bildern.
  * Hypothese verworfen, wenn (iii) gegenüber (i) den RMSE nicht um mindestens 25 % senkt oder die
    PPI-Breite nicht messbar reduziert.
  * Wichtig: Gilt R4 (Anomaliekarte als Eingang), sind A und B abhängig. Das Experiment muss dann mit
    dem *zweiten*, nicht eingespeisten Normalitätsmodell laufen.

### 11.4 Aufgabenspezifische LoRA-Adaption von DINOv3 mit eigener Residual-Synthese („AnomalyVFM pro Aufgabe“)

* **Idee:** AnomalyVFM adaptiert ein VFM per LoRA auf *generische* synthetische Defekte (FLUX) und
  erreicht damit zero-shot P-AP ≈ 44 im Mittel [R]. Wir haben pro Aufgabe etwas Besseres: echte
  Defekt-Residuen und Tausende echter Hintergründe. Vorgehen:
  * DINOv3 per LoRA mit einem leichten Decoder und konfidenzgewichtetem Loss trainieren (wie
    AnomalyVFM), auf Residual-Pastes der eigenen Aufgabe plus Fold-konsistenten Pseudo-Labels;
  * Cross-Fitting über die Folds wie beim U-Net;
  * die adaptierten Features sowohl als Segmentierer (5. Ensemble-Mitglied mit anderem induktivem
    Bias) als auch fürs kNN nutzen.
* **Warum das bei uns besser sein könnte:** Es schließt die Domänenlücke, die generische Zero-Shot-VFMs
  haben (VisA nur 27 P-AP [R]). Es nutzt die semantische Stärke des Foundation-Backbones für große und
  variable Defekte, wo unser ImageNet-resnet18d-U-Net schwächer sein dürfte, und gewinnt
  Ensemble-Diversität. Der aufgabenagnostische Charakter bleibt erhalten (LoRA-Rang und Iterationen
  fest).
* **Falsifizierbares Experiment:** KSDD2, MT, BSData, 15 gelabelte Bilder, 3 Seeds. Verglichen werden:
  (i) unser U-Net-Ensemble (Runde 0/1), (ii) AnomalyVFM zero-shot, (iii) DINOv3-LoRA mit eigener
  Synthese, (iv) Ensemble (i)+(iii).
  * Metriken: SegF1 bei OOF-gewählter Schwelle auf U und T, Recall auf Defektebene pro Flächenklasse.
  * Hypothese verworfen, wenn (iii) nicht besser ist als (ii) **und** (iv) nicht mindestens
    +2 Punkte SegF1 gegenüber (i) bringt. Die Laufzeit auf dem M4 wird mitprotokolliert.

---

## 12. Offene Punkte und nicht Gefundenes

* Nicht im Text gefunden: Pixelzahlen von Dinomaly (DINOv2) und INP-Former (nur Bildtabellen),
  AdaCLIP, VCP-CLIP, UniVAD, UniNet; Pixeltabellen von BGAD, PRN, AHL, DRA.
* Unbekannt: das genaue „refined subsampling“ von SuperADD, ebenso die Spaltenzuordnung der Baselines im
  MVTec-AD-2-Paper. Volltexte waren nicht abrufbar.
* Nicht geprüft: MANTA- und 3CAD-Ergebnisse, kontinuierliche Multi-Task-AD (UCAD/CDAD), Myriad, die
  Lizenzen von AnomalyVFM bzw. RADIO.
* Nicht gemessen: alle Laufzeitaussagen für den Mac mini M4. Literaturangaben beziehen sich auf
  RTX 3090/4090.
* Das Web-Suchbudget war am Ende erschöpft. Eine Folgesitzung könnte gezielt die Volltexte von
  MVTec AD 2, SuperADD, Dinomaly2 und CoDeGraph prüfen.

---

## 13. Referenzen

Marker: [R] = Repository gelesen, [A] = Abstract/Suchauszug, (unverified) = Details ungeprüft.

**Backbones und Upsampler**
1. Siméoni et al., *DINOv3*, arXiv 2508.10104 (2025). Code und Lizenz: github.com/facebookresearch/dinov3 [R]
2. Fu et al., *FeatUp: A Model-Agnostic Framework for Features at Any Resolution*, ICLR 2024 (unverified, nur zitiert)
3. Huang et al., *LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models*, ICCV 2025, arXiv 2504.14032 [R]
4. Couairon et al., *JAFAR: Jack up Any Feature at Any Resolution*, arXiv 2506.11136 (2025) [A]
5. Wimmer et al., *AnyUp: Universal Feature Upsampling*, ICLR 2026, arXiv 2510.12764 [R]
6. Walmer et al., *UPLiFT: Efficient Pixel-Dense …*, CVPR 2026 (Supplement) [A]
7. *SAM-LAD: Segment Anything Model Meets Zero-Shot Logic Anomaly Detection*, arXiv 2406.00625 [A]
8. *Maybe you don't need a U-Net: convolutional feature upsampling for materials micrograph segmentation*, arXiv 2508.21529 [A]

**Patch-kNN, Rekonstruktion, vereinheitlichte Modelle**
9. Roth et al., *Towards Total Recall in Industrial Anomaly Detection (PatchCore)*, CVPR 2022 (unverified, nur zitiert)
10. Damm et al., *AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2*, WACV 2025, arXiv 2405.14529 [A/R]
11. Zhang et al., *SuperAD: A Training-free Anomaly Classification and Segmentation Method for CVPR 2025 VAND 3.0 Track 1*, arXiv 2505.19750 [A]
12. *SuperADD: Training-free Class-agnostic Anomaly Segmentation – CVPR 2026 VAND 4.0 Industrial Track*, arXiv 2605.14808; Code github.com/LukasRoom/SuperADD [A/R]
13. *RoBiS: Robust Binary Segmentation for High-Resolution Industrial Images*, arXiv 2505.21152; Code github.com/xrli-U/RoBiS [R]
14. Guo et al., *Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection*, CVPR 2025, arXiv 2405.14325 [R]
15. Guo et al., *One Dinomaly2 Detect Them All: A Unified Framework for Full-Spectrum Unsupervised Anomaly Detection*, arXiv 2510.17611 [A/R]
16. Luo et al., *Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection (INP-Former)*, CVPR 2025, arXiv 2503.02424; INP-Former++ arXiv 2506.03660 [R]
17. He et al. (Erstautor unverified), *MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection*, NeurIPS 2024, arXiv 2404.06564 [R]
18. Wei et al., *UniNet: A Contrastive Learning-guided Unified Framework with Feature Selection for Anomaly Detection*, CVPR 2025 [A]
19. Gu et al., *UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection*, CVPR 2025, arXiv 2412.03342 [A]
20. Zhang et al., *ADer / A Comprehensive Library for Benchmarking Multi-class Visual Anomaly Detection*, arXiv 2406.03262 [A/R]
21. *Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors*, arXiv 2510.01934 (unverified)
22. *AD-DINOv3: Enhancing DINOv3 for Zero-Shot Anomaly Detection with Anomaly-Aware Calibration*, arXiv 2509.14084 (unverified)
23. *DuoAD*, arXiv 2607.23924; *RAID: Retrieval-Augmented Anomaly Detection*, arXiv 2602.19611 (unverified)

**Batch, transduktiv, Kontamination**
24. Li et al., *MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images*, ICLR 2024, arXiv 2401.16753 [R]
25. Li et al., *MuSc-V2*, arXiv 2511.10047, TPAMI 2026 [R]
26. Le Gia & Ahn, *On the Problem of Consistent Anomalies in Zero-Shot Industrial Anomaly Detection (CoDeGraph)*, TMLR 2025, arXiv 2510.10456 [R]
27. Zhao et al., *DeMuS: Learning Decoupled Matching and Scoring for Batch Zero-Shot Industrial Anomaly Detection*, ECCV 2026 [R]
28. *SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling*, arXiv 2602.23013 [A]
29. Li et al., *Zero-Shot Anomaly Detection via Batch Normalization*, NeurIPS 2023, arXiv 2302.07849 (unverified)
30. Jiang et al., *SoftPatch: Unsupervised Anomaly Detection with Noisy Data*, NeurIPS 2022; SoftPatch+ arXiv 2412.20870 [R]
31. McIntosh & Albu, *Inter-Realization Channels (InReaCh)*, ICCV 2023; Online InReaCh ECCV 2024 [R]
32. Im et al., *FUN-AD: Fully Unsupervised Learning for Anomaly Detection with Noisy Training Data*, WACV 2025, arXiv 2411.16110 [R]
33. *Towards Real Unsupervised Anomaly Detection via Confident Meta-Learning*, ICCV 2025, arXiv 2508.02293 (unverified)

**CLIP, VLM, MLLM, SAM**
34. Jeong et al., *WinCLIP*, CVPR 2023 [R, Zahlen via MuSc-Tabelle]
35. Chen et al., *APRIL-GAN*, arXiv 2305.17382 (VAND 2023) [R, via MuSc-Tabelle]
36. Zhou et al., *AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection*, ICLR 2024, arXiv 2310.18961 [R]
37. Cao et al., *AdaCLIP*, ECCV 2024, arXiv 2407.15795 [R]
38. Qu et al., *VCP-CLIP*, ECCV 2024 [R]
39. Ma et al., *AA-CLIP: Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP*, CVPR 2025, arXiv 2503.06661 [A/R]
40. Qu et al., *DictAS*, ICCV 2025, arXiv 2508.13560 [R]
41. *MultiADS: Defect-aware Supervision for Multi-type Anomaly Detection and Segmentation in Zero-Shot Learning*, ICCV 2025, arXiv 2504.06740; DAPO arXiv 2512.09446 [R]
42. Fučka, Zavrtanik, Skočaj, *AnomalyVFM – Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors*, CVPR 2026, arXiv 2601.20524 [R]
43. Gu et al., *AnomalyGPT*, AAAI 2024 [R]
44. Jiang et al., *MMAD: A Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection*, ICLR 2025, arXiv 2410.09453 [A]
45. *AnomalyR1: A GRPO-based End-to-end MLLM for Industrial Anomaly Detection*, arXiv 2504.11914 [A]
46. *OmniAD*, arXiv 2505.22039 [A]
47. *IAD-R1: Reinforcing Consistent Reasoning in Industrial Anomaly Detection*, AAAI 2026, arXiv 2508.09178 [A]
48. Jin et al., *Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision*, CVPR 2026, arXiv 2603.27179 [A]
49. *SAM 3: Segment Anything with Concepts*, ICLR 2026, arXiv 2511.16719 [A]

**Supervised und Open-Set**
50. Pang et al., *Explainable Deep Few-shot Anomaly Detection with Deviation Networks*, arXiv 2108.00462 (DevNet, KDD 2019) [A]
51. Ding, Pang, Shen, *Catching Both Gray and Black Swans: Open-set Supervised Anomaly Detection (DRA)*, CVPR 2022 [A]
52. Yao et al., *Explicit Boundary Guided Semi-Push-Pull Contrastive Learning for Supervised Anomaly Detection (BGAD)*, CVPR 2023, arXiv 2207.01463 [A/R]
53. Zhang et al., *Prototypical Residual Networks for Anomaly Detection and Localization (PRN)*, CVPR 2023, arXiv 2212.02031 [A]
54. Zhu et al., *Anomaly Heterogeneity Learning for Open-set Supervised Anomaly Detection (AHL)*, CVPR 2024, arXiv 2310.12790 [A]
55. *Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection (DPDL)*, CVPR 2025, arXiv 2502.20981 (unverified)
56. *Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection*, ICML 2026, arXiv 2605.02438 (unverified)
57. Wang, Shen, Yan, Pang, *Beyond Normal References: Discriminative Few-Shot Anomaly Detection (IDEAL)*, arXiv 2605.23231 [A]
58. Abdali et al., *Anomalous Samples for Few-Shot Anomaly Detection*, arXiv 2507.23712 [A]
59. *ArcAD: Anomaly-Rectified Calibration for Cold-Start Supervised Anomaly Detection*, ECCV 2026, arXiv 2607.02252 [R]
60. Baitieva et al., *Supervised Anomaly Detection for Complex Industrial Images (SegAD, VAD)*, CVPR 2024, arXiv 2405.04953 [A/R]
61. Božič et al., *Mixed supervision for surface-defect detection: from weakly to fully supervised learning*, arXiv 2104.06064 [A]
62. Rolih et al., *SuperSimpleNet*, ICPR 2024, arXiv 2408.03143; *No Label Left Behind*, JIMS 2025, arXiv 2508.19060 [R]
63. Huang et al., *AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios*, CVPR 2025, arXiv 2410.14379 [R]

**Benchmarks und Challenges**
64. Heckler-Kram et al., *The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection*, arXiv 2503.21622 [A] (Erstautor nicht verifiziert)
65. *From Benchmarks to Reality: Advancing Visual Anomaly Detection by the VAND 3.0 Challenge*, arXiv 2509.17615 [A]
66. *Detecting Defects that Matter: An Application-Driven Benchmark for Anomaly Detection in Manufacturing and Retail Logistics (VAND 4.0 Challenge)*, arXiv 2610.04392 [A]
67. Wang et al., *Real-IAD: A Real-World Multi-View Dataset for Benchmarking Versatile Industrial Anomaly Detection*, CVPR 2024, arXiv 2403.12580 [R, via ADer]
68. *MANTA: A Large-Scale Multi-View and Visual-Text Anomaly Detection Dataset for Tiny Objects*, CVPR 2025, arXiv 2412.04867 (unverified)
69. Yang et al., *3CAD: A Large-Scale Real-World 3C Product Dataset for Unsupervised Anomaly Detection*, AAAI 2025, arXiv 2502.05761 (unverified)
