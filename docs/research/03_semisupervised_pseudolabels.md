# 03 – Semi-Supervised Segmentierung, Pseudo-Label-Verfeinerung (inkl. SAM), verrauschte Labels, Multi-Skalen und label-freie Qualitätsschätzung

Stand: 18.10.2026 · Literatur-Review für `auto-label-tool` · Bezug: [`METHOD.md`](../METHOD.md), [`HANDOFF.md`](../HANDOFF.md)

> **Wie belastbar das ist.** Recherchiert wurde per Websuche. Volltexte von arXiv, CVF, Nature und
> HuggingFace ließen sich in dieser Umgebung nicht abrufen (DNS-Fehler); GitHub-READMEs ließen sich
> lesen. Zahlen stammen deshalb aus (a) offiziellen GitHub-READMEs (markiert **[README]**) oder
> (b) Abstracts und Such-Snippets (markiert **[Abstract/Snippet]**). Diese sollten vor einer Publikation
> am PDF geprüft werden. Bibliographische Angaben, die nur aus Vorwissen stammen, sind mit
> **(unverified)** markiert. Das Websuche-Budget war gegen Ende aufgebraucht. Zwei Fragen
> (Tiling vs. Downscaling, Details zu MVTec AD 2) ließen sich deshalb nicht mehr gezielt recherchieren
> und werden aus Grundprinzipien beantwortet (als *Einschätzung* gekennzeichnet).

---

## 0. Executive Summary

1. **Der Encoder ist der stärkste Hebel, nicht der SSL-Algorithmus.** Bei UniMatch V2 steigt Pascal VOC
   mit 92 Labels allein durch den Wechsel von ResNet-101 auf DINOv2-B von 75,2 auf 86,3 mIoU, bei
   Cityscapes mit 186 Labels von 76,6 auf 83,6 **[README]**. Methodische Neuerungen von 2023–2024 liegen
   bei gleichem Backbone dagegen bei etwa +1 mIoU (CorrMatch 76,4, AllSpark 76,1 **[README]**). Für die
   Industrie gibt es eine wichtige Einschränkung: Laut Gharbage et al. (CVPRW 2026) bringt DINOv3 mit
   *eingefrorenem* Backbone keinen konsistenten Vorteil gegenüber ImageNet. *Feinjustiert* ist DINOv3
   der beste Start für RGB-Defektsegmentierung, auf Röntgendaten aber schlechter **[Abstract/Snippet]**.
   → DINOv3-Encoder (ConvNeXt-T/S oder ViT-S/B mit Full-Res-Pfad) als **zusätzliche Ensemble-Member**
   testen, nicht als blinden Ersatz.
2. **Das bestehende fold-konsistente Self-Training ist konzeptionell solide.** Was ihm fehlt, ist eine
   *Online*-Weak-to-Strong-Konsistenz mit **Feature-Perturbation** (UniMatch V2 „Complementary Dropout“).
   Feature-Perturbation ist bei kontrastarmen Defekten sicherer als starke Intensitäts-Augmentierung,
   weil sie den Defekt im Eingangsbild nicht auslöscht.
3. **Feste Konfidenzschwellen benachteiligen seltene Klassen systematisch.** Das berichten CAFS,
   AEL, Supervised ClassMix und Arbeiten zu klassen-unbalancierter SSL. Unsere OOF-kalibrierten
   Hysterese-Schwellen und die Copy-Paste-Synthese sind dafür die richtige Antwort. Ergänzen sollte man
   **pro Klasse und pro Größenklasse** getrennte Schwellen und eine klassen-balancierte Pixel-Stichprobe
   in der Konsistenz-Loss.
4. **SAM/SAM2/SAM3 als Refiner helfen nur unter Bedingungen.** Bei kompakten Objekten mit klaren
   Kanten ist der Nutzen belegt: In „SAM Carries the Burden“ stieg der Dice im medizinischen Fall durch
   SAM-verfeinerte Pseudo-Labels von 66,6 auf 74,9 bzw. von 74,3 auf 84,2 **[Abstract/Snippet]**. Bei
   dünnen, kontrastarmen Strukturen ist es schlechter: Zero-Shot erreicht SAM3 auf Rissen im Mittel
   56,5 Crack-IoU bei 70,4 % Recall **[README SERD]**, und für dünne Risse ist schlechtes Abschneiden
   mehrfach berichtet. → SAM nur als **gegateter, pro Komponente validierter** Refiner für
   „blob-artige“ Defekte einsetzen, nie für Linien.
5. **In-Context-/Few-Shot-Foundation-Modelle taugen als Labeler für Defekte schlecht.** SegGPT kommt auf
   Defect Spectrum auf ca. 28–32 mIoU. PerSAM und Matcher schneiden laut Liu et al. 2025 „auf den
   meisten Industriedatensätzen schlecht“ ab. Trainingsfreies Feature-Matching erreicht ca. 35–37 mIoU
   im Mittel über 54 Defektkategorien **[Abstract/Snippet]**. Sinnvoll sind sie nur als *unabhängiger
   Zweitdetektor* für übersehene Defekte.
6. **Das Produktionsmodell sollte mit Soft-Targets plus Ignore plus Gewichten trainieren** und dafür eine
   Dice-Variante nutzen, die Soft-Labels korrekt behandelt (Dice Semimetric Loss, MICCAI 2023). Dazu
   kommen räumlich variierendes Label-Smoothing an Rändern (SVLS) und Early-Learning-Kontrolle pro
   Klasse (ADELE). Harte 0/1-Pseudo-Labels mit Soft-Dice sind die schlechteste Kombination.
7. **Bei Multi-Skalen braucht es komponenten-normierte Losses.** Die blob loss bringt ca. +2–5 % bei
   Instanz-Metriken. Für dünne Defekte eignet sich die Skeleton Recall Loss (mehr als 90 % weniger
   Overhead als clDice, multi-klassenfähig). Tversky- oder Focal-Tversky-Varianten verschieben die
   Balance gezielt Richtung Recall. Ohne Metriken pro Größenklasse bleibt der Nutzen allerdings
   unsichtbar.
8. **Für die label-freie Qualitätsschätzung ist auf Bildebene der paarweise Dice zwischen
   Ensemble-Membern der robusteste Schätzer** (Zenk et al., MedIA 2025, über fünf bzw. sechs
   Datensätze). Auf Komponentenebene sagen MetaSeg-artige Meta-Klassifikatoren False Positives mit
   AUROC ≈ 88–92 % voraus **[Abstract/Snippet]**. **Übersehene Defekte (FN) adressiert keine der
   etablierten Methoden direkt.** Genau hier liegt die Forschungslücke (Ideen N2 und N3).
9. **Top-3-Empfehlung.** (1) Heterogenes Ensemble mit DINOv3-Encoder. (2) Neues Trainingsrezept für
   das Produktionsmodell (Soft-Targets, DML, blob- und Skeleton-Recall-Terme). (3) Komponentenweiser
   Qualitäts-Meta-Klassifikator plus paarweiser Dice, kalibriert auf OOF und geprüft per PPI.
   SAM-Refinement erst auf Rang 5, gegatet.
10. **Vier neue Forschungsideen** (Abschnitt 5) mit falsifizierbaren Experimenten. Die wichtigste:
    *Recall-Schätzung durch synthetische Injektion plus Capture–Recapture zwischen unabhängigen
    Detektoren*.

---

## 1. Ausgangslage und Prüfkriterien

Unsere Lage unterscheidet sich von fast allen Benchmarks. Pro Task gibt es **5–30 Masken** (Pascal-VOC-SSL
startet bei 92, ACDC bei 1–7 Volumina mit großen Organen). Der Vordergrund ist **selten** (typisch < 2 %
der Pixel, viele Bilder defektfrei), die Defekte sind **sehr unterschiedlich groß** (1-px-Kratzer bis
flächige „uneven“-Regionen) und es gibt **mehrere Klassen**. Die einzige bisher gemessene Zahl unseres
Baselines (Magnetic Tile, L = 15, Runde 0, OOF) ist tol-F1 = 0,31 bei Defekt-Recall 0,73
([`HANDOFF.md`](../HANDOFF.md)). Es ist also noch viel Spielraum.

Jede Methode wird deshalb nach fünf Kriterien bewertet: **(W)** wenige Labels, **(S)** seltene und
kleine Defekte, **(M)** Multi-Klasse, **(T)** task-agnostisch, das heißt ohne Task-spezifisches
Engineering, und **(K)** Kosten auf dem Mac mini M4 (MPS).

---

## 2. Review nach Unterthemen

### 2.1 Semi-Supervised Semantic Segmentation (SSSS), 2022–2026

**Weak-to-Strong-Konsistenz (FixMatch → UniMatch).** UniMatch (Yang et al., CVPR 2023) erweitert
FixMatch um Feature-Perturbation und zwei starke Bildansichten.
*Ergebnisse* **[README/Snippet]**: Pascal VOC 92 Labels 75,2 mIoU (ResNet-101). Medizinisch auf ACDC
mit 1, 3 und 7 gelabelten Fällen 85,4, 88,9 und 89,9 Dice; der rein supervised Baseline liegt bei
28,5, 41,5 und 62,5.
*Übertragbarkeit*: Der ACDC-Sprung zeigt, dass Konsistenz bei *extrem* wenigen Labels enorm helfen kann,
wenn das Ziel groß und strukturell stereotyp ist (Herzstrukturen). Für seltene, kleine Defekte fehlt
dieser Beleg. Wenn die meisten ungelabelten Bilder defektfrei sind, verstärkt eine Konsistenz-Loss mit
fester Konfidenzschwelle vor allem „Hintergrund“ (Confirmation Bias Richtung FN).
*Kosten*: zwei bis drei Forward-Passes pro U-Bild und Schritt.

**UniMatch V2** (Yang, Zhao, Zhao, TPAMI 2025, arXiv 2410.10777) nutzt einen DINOv2-Encoder, einen
einzigen Augmentierungsstrom, in dem Bild- und Feature-Perturbation verschmolzen sind, und
„Complementary Dropout“, das zwei komplementäre Feature-Views erzeugt.
*Ergebnisse* **[README]** mit DINOv2-B, jeweils gegenüber UniMatch V1 mit ResNet-101:

| Datensatz / Split | UniMatch V1 (R101) | AllSpark (MiT-B5) | SemiVL (CLIP-B) | **UniMatch V2 (DINOv2-B)** |
|---|---|---|---|---|
| VOC 92 Labels | 75,2 | 76,1 | 84,0 | **86,3** |
| Cityscapes 186 | 76,6 | 78,3 | 77,9 | **83,6** |
| ADE20K 316 | 21,6 | – | 33,7 | **38,7** |
| COCO 232 (1/512) | 31,9 | 34,1 | **50,1** | 47,9 |

Die Autoren schreiben selbst, dass das Encoder-Upgrade mehr bringt als sorgfältiges Methodendesign
**[Abstract]**. Im Extremfall mit den wenigsten Labels (COCO 1/512) liegt allerdings SemiVL mit
Sprachprior vorn.
*Übertragbarkeit*: (W) +, (S) ?, (M) +, (T) ++. Der Kernbefund „starker vortrainierter Encoder vor
Methode“ ist das wichtigste Ergebnis für uns. ViT-Patch-Größe 14 (DINOv2) bzw. 16 (DINOv3) bedeutet
aber, dass 1–3-px-Defekte im Token-Raum nicht aufgelöst werden. Der bestehende Full-Res-Stem
beziehungsweise ein Feature-Upsampler (Abschnitt 2.3) ist dann Pflicht.
*Kosten*: Das README empfiehlt 4 GPUs zur Reproduktion. Für unsere kleinen Crops mit ViT-S/B ist das
auch auf MPS machbar (Einschätzung).

**CorrMatch** (Sun et al., CVPR 2024, arXiv 2306.04300) propagiert Labels über Korrelationskarten:
hochkonfidente Pixel per paarweiser Ähnlichkeit (Pixel-Propagation) und klassenagnostische Regionen aus
Korrelationskarten (Region-Propagation).
*Ergebnisse* **[README]** mit DeepLabV3+ und R101: VOC 92 Labels 76,4 gegenüber 75,2 (UniMatch),
Cityscapes 186 Labels 77,3 gegenüber 76,6.
*Übertragbarkeit*: Die Idee, Seeds über Affinitäten zu erweitern, passt zu unserer
Hysterese-Expansion. Bei dünnen Defekten sind Korrelationen auf Stride-8/16-Features aber zu grob.
Lohnend wäre eine Version auf Full-Res-Features. Es handelt sich um einen kleinen Gewinn.

**AllSpark** (Wang et al., CVPR 2024) „regeneriert“ Labeled-Features aus Unlabeled-Features in einem
Transformer-Kanal. Ergebnisse siehe Tabelle oben (+0,9 mIoU über V1). Für uns ist das
architektonisch aufwändig bei kleinem Gewinn: niedrige Priorität.

**SemiVL** (Hoyer et al., ECCV 2024, arXiv 2311.16241) nutzt CLIP-Vision-Language-Guidance und
Klassendefinitionen; laut Abstract +6,1 mIoU bei VOC 92 gegenüber dem damaligen SOTA. Für
Industriedefekte ist der Sprachprior schwach („scratch“ und „pit“ sind in CLIP kaum gut verankert).
Siehe auch die Befunde zu SAM3-Textprompts in Abschnitt 2.2. (T) −.

**PrevMatch** (Shin et al., WACV 2026, arXiv 2405.20610) nutzt frühere Checkpoints als zusätzliche
Pseudo-Label-Quelle („previous guidance“), mit einem stark randomisierten Ensemble daraus.
*Ergebnis* **[Abstract]**: +1,6 mIoU gegenüber Diverse Co-Training bei VOC 92 und 2,4× schnelleres
Training. Auf Cityscapes 1/2 sind andere Methoden besser.
*Übertragbarkeit*: **sehr gut und billig**. Innerhalb von Fold k sind Snapshots von Member k leckfrei,
also verträglich mit unserer Fold-Konsistenz. Das ergibt mehr Teacher-Diversität ohne zusätzliches
Training.

**Neuere Arbeiten 2025–2026 (nur Abstract-Niveau):**
- *BoundMatch* (Ishikawa & Aoki, arXiv 2503.23519) ergänzt Konsistenz auf semantischen *Rändern*
  zwischen Teacher und Student; laut Autoren SOTA auf Cityscapes mit DINOv2.
- *SemiOVS* (arXiv 2507.03302) nutzt Out-of-Distribution-Unlabeled-Bilder über ein
  Open-Vocabulary-Modell; +3,5 gegenüber PrevMatch und +3,0 gegenüber SemiVL bei VOC 92.
- *Knowledge Consultation* (arXiv 2503.10693, DINOv2-S).
- *S5* (arXiv 2508.12409) skaliert SSSS in der Fernerkundung über domänenspezifisches Pretraining.

Keine dieser Arbeiten adressiert seltenen Vordergrund.

**Klassen-Unbalance in SSL.**
- *AEL* (NeurIPS 2021, arXiv 2110.05474) zieht Pixel unterrepräsentierter Klassen häufiger
  (adaptive equalization sampling) und gewichtet Pseudo-Labels dynamisch.
- *CAFS* (arXiv 2303.11606) weist explizit darauf hin, dass U2PL und UniMatch die Klassen-Unbalance
  der Pseudo-Labels ignorieren.
- *Guo & Li* (ICML 2022) belegen für Klassifikation, dass feste Schwellen Minderheitsklassen
  benachteiligen, und schlagen adaptive Schwellen pro Klasse vor.
- *Supervised ClassMix* (arXiv 2604.07122) beschreibt den Mechanismus: Seltene Klassen haben früh
  geringe Konfidenz und werden deshalb kaum Pseudo-Label. Die Gegenmaßnahme ist, gelabelte
  Vordergrundregionen in Unlabeled-Bilder zu kleben. Das entspricht strukturell unserem
  Residual-Copy-Paste.
- *DHC* (Wang & Li, MICCAI 2023, „Dual-debiased Heterogeneous Co-training“; Details unverified) und
  ein CVPR-2025-Decoupling-Framework für klassen-unbalancierte medizinische SSL bestätigen, dass das
  Problem auch in der Medizin akut ist.

**Semi-supervised *Defekt*-Segmentierung.** Die Literatur ist schmal und meist mit hohen Labelquoten:
- Sci. Rep. 2024 (s41598-024-72579-6): Bild- und Feature-Perturbation plus Cross-Pseudo-Supervision
  für Oberflächendefekte.
- Adv. Eng. Informatics 2025 (Aluminiumband): drei Decoder mit Cross-View-Pseudo-Supervision und
  Self-Attention-Konsistenz.
- Expert Syst. Appl. 2025 (Stahl): leichte SSL-Distillation mit „dual-level adaptive thresholding“
  für die Pseudo-Label-Auswahl, im Einsatz auf einer Kaltwalzlinie.
- *SemiCrack* (Xiang et al., Measurement 2023): CPS plus Kontrastlernen.
- DeepCrack mit 20–80 % Labels (Frontiers 2022).
- Ein U-Net mit dynamischen Pseudo-Labels (Buildings 2025) erreicht F1 0,883 bei 50 % Labels gegenüber
  0,896 voll supervised **[Snippet]**.

Am nächsten an unserem Szenario ist **Božič et al. (Comput. Ind. 2021, KolektorSDD2)**. Ohne
Pixel-Labels (nur Bild-Labels) kommt das Modell auf 73,3 AP, mit **16 pixelgelabelten Bildern** plus
schwachen Labels auf 83,2 AP und mit allen Pixel-Labels auf 95,4 AP **[Snippet]**.
*Fazit*: Es gibt **keine publizierte Evaluation im Regime 5–30 Masken plus seltene Defekte in U**.
Unsere Benchmarks wären hier selbst ein Beitrag.

### 2.2 Pseudo-Label-Auswahl und -Verfeinerung

**Unsicherheitsbasierte Auswahl.**
- *UPS* (Rizve et al., ICLR 2021, arXiv 2101.06329) diagnostiziert, dass falsch-hochkonfidente
  Vorhersagen schlecht kalibrierter Modelle die Hauptquelle von Pseudo-Label-Rauschen sind. Es filtert
  deshalb mit Konfidenz *und* Unsicherheit und führt **negative Pseudo-Labels** ein.
- *U2PL* (Wang et al., CVPR 2022, arXiv 2203.03884; IJCV-Erweiterung U2PL+) trennt verlässliche und
  unverlässliche Pixel über die Entropie. Unverlässliche Pixel werden *nicht verworfen*, sondern als
  Negative für die unwahrscheinlichen Klassen genutzt („zwischen Motorrad und Person, aber sicher
  nicht Auto“).
- *Übertragung auf uns*: Bei Multi-Klasse ist das direkt nutzbar. Ein Pixel kann sicher „Defekt“ und
  zugleich unsicher in der Klasse sein. → **Hierarchische Pseudo-Labels**: Für den binären
  Defekt-Kopf ist so ein Pixel positiv, für den Klassen-Kopf ignoriert (oder negativ für die
  ausgeschlossenen Klassen). Das passt zur vorhandenen Trennung Segmentierung (binär) plus
  `classify.py` (Typzuordnung).

**Kalibrierung und Schwellen.** Unsere isotone Kalibrierung auf OOF-Vorhersagen plus Hysterese plus
konforme Recall-Schwelle ist methodisch weiter als alles, was SSSS-Papers nutzen (meist τ = 0,95 fest;
Wert aus Vorwissen). Offen ist laut [`HANDOFF.md`](../HANDOFF.md) nur der Schwellen-Transfer von
Einzel-Member-OOF auf den Ensemble-Mittelwert. FlexMatch- bzw. CAFS-artige Schwellen pro Klasse sind
bei uns **besser als OOF-Schwellen pro Klasse und pro Größenklasse** realisiert, nicht als
Lernfortschritts-Heuristik.

**Klassenagnostische Refiner.** *SegRefiner* (Wang et al., NeurIPS 2023, arXiv 2312.12425) behandelt
Mask-Refinement als diskreten Diffusionsprozess. Er ist modellagnostisch und soll laut Autoren sehr
feine Details bei hochauflösenden Bildern erfassen. Er ist aber auf Natur- und Dichotomous-Bildern
trainiert, ein Domänentransfer auf Defekte ist unbelegt. CRF und Guided Filter: In diesem Lauf wurde
keine neue Evidenz für Defekte gefunden. *Einschätzung*: Kantenbasierte Glättung verstärkt bei
kontrastarmen, diffusen Defekten eher Reflexions- und Texturkanten. Nur als Ablation sinnvoll.

**SAM-Familie als Refiner oder promptbarer Labeler.**

| Modell | Kernpunkt | Evidenz (Quelle) |
|---|---|---|
| SAM (Kirillov et al., ICCV 2023, arXiv 2304.02643, unverified ID) | promptbar (Punkt, Box, Maske) | Basis |
| HQ-SAM (Ke et al., NeurIPS 2023, arXiv 2306.01567) | HQ-Output-Token, Fusion früher und später ViT-Features, < 0,5 % neue Parameter, 44K Masken | adressiert explizit grobe Ränder und „dünne Strukturen“. Spätere Arbeiten berichten, dass es die Schwächen des Basismodells erbt **[Snippet]** |
| SAM 2 (Ravi et al., arXiv 2408.00714, unverified ID) | Video-Memory | Liu et al. 2025: „SAM2 kann im Video-Tracking-Modus Few-Shot-Defektsegmentierung end-to-end“ **[Abstract]** |
| SAM 3 (Meta, arXiv 2511.16719) | Promptable Concept Segmentation (Text und Bild-Exemplare), 848M Parameter, Detektor plus SAM-2-Tracker, interaktive Verfeinerung im Video. SAM 3.1 „Object Multiplex“ seit 27.03.2026 | SA-Co: 75–80 % menschlicher Leistung **[README]** |
| SERD (arXiv 2607.12292) | trainingsfreies Dekodieren interner SAM3-Antworten statt finaler Masken | 6 Rissdatensätze, Mittel: SAM3 56,51 Crack-IoU, R 70,38 / P 68,35. SERD 61,14 IoU, R 78,01 **[README]**. Die Autoren sagen, finale Textprompt-Masken seien für dünne, fragmentierte, kontrastarme Risse schlecht geeignet |
| SAMRefiner (Lin et al., ICLR 2025, arXiv 2502.06756) | rauschtolerante Prompt-Erzeugung *aus einer groben Maske*: distanzgeführte Punkte, kontextabhängige „elastische“ Boxen, gaußförmige Maskenprompts, Split-then-Merge für mehrere Objekte. SAMRefiner++ mit selbstüberwachter IoU-Adaption. Unterstützt HQ-SAM | Zahlen im README nicht angegeben |
| SAM Carries the Burden (Keuth et al., 2024, arXiv 2411.12602) | wenige Labels → U-Net → Box- und Punktprompts → SAM → Pseudo-Labels | Dice 74,29 → 84,17 (Handgelenk), 66,63 → 74,87 (Zähne). Unverfeinerte Pseudo-Labels brachten *keinen* Nutzen. 45 Labels erreichen das Niveau von 450 **[Snippet]** |
| SAMatch (arXiv 2411.16949) | hochkonfidente Teacher-Vorhersagen nur als *Prompts* für ein auf L feinjustiertes SAM | medizinisch |
| S⁴M (arXiv 2504.05301) | naive Prompts (Zentrum, Box) führen zu Über-Segmentierung oder degenerierten Masken → stochastische Mehrpunkt-Prompts | Instanz-SSL |
| SSL-MedSAM2 (arXiv 2512.11548) | SAM2-Few-Shot-Pseudo-Labels, iterativ mit nnU-Net verfeinert | Challenge-Daten |

*Industrie- und Riss-spezifisch:*
- „SAM Era“ (2024): automatische SAM-Masken auf Stahl, Fliesen und Schienen sind unzureichend.
- *DefectSAM* (IEEE 2025): Prototyp-Prompts aus Support-Paaren plus LoRA.
- *SAID* (Sensors 2025): Szenen-Prompts; +0,87 mIoU gegenüber SegGPT (≈ 29,1 mIoU) one-shot.
- *PA-SAM*: PConv-Finetuning plus automatischer Prompter.
- *P³-SAM* (arXiv 2609.21424): +12,00 % mIoU auf Surface Defects-4i.
- *DCM-SAM* (arXiv 2609.38811): Mixture of LoRA Experts.
- *SAM3-LoRA* (arXiv 2609.00469): PEFT-Studie für Multi-Klassen-Strukturdefekte.
- *VNS-SAM* (arXiv 2601.00537): „visually non-salient“, übertrifft SAM und HQ-SAM-F; Industriedefekte
  sind dort ein ungesehenes Szenario.
- Bei Rissen: *CrackSAM* (Ge et al., arXiv 2312.04233; Adapter bzw. LoRA, robust bei Rauschen und
  ungesehenen Daten laut Abstract), *Segment Any Crack* (arXiv 2504.14138; F1 61,22 / IoU 44,13 auf
  OmniCrack30k) und *Distribution-aware Noisy-label Crack Segmentation* (arXiv 2410.09409; SAM-Adapter
  gegen verrauschte Labels, Generalisierung auf CFD).
- Eine Zero-Shot-Studie zu dünnen Strukturen (Int. J. Inf. Technol. 2024) findet SAM visuell
  zufriedenstellend, aber numerisch unzureichend gegenüber 1-px-genauer GT.
- Ein medizinischer Zero-Shot-Test berichtet „komplettes Versagen“ bei Gefäßen.

**Synthese: Wann SAM-Ränder helfen und wann sie schaden.**

| hilft (belegt bzw. plausibel) | schadet oder unzuverlässig (belegt) |
|---|---|
| kompakte, flächige Defekte mit klarer Intensitäts- oder Texturkante (Ausbruch, Fleck, Delle mit Schatten) | 1–3-px-Linien, Haarrisse, verzweigte Strukturen (dünne Risse, Gefäße) |
| grobes Modell-Masken-Prior **plus** Box- und Punktprompts (SAM Carries the Burden, SAMRefiner) | Textprompts für Defektkonzepte (semantischer Drift, hohe Fehlalarme laut Prompt-Survey arXiv 2507.09562) |
| Feinjustierung auf wenigen Task-Masken (SAMatch, DefectSAM, CrackSAM) | diffuse oder subjektive Grenzen: SAM sucht „Objekt“-Kanten, die bei Verschleiß oder „uneven“ nicht existieren |
| Mehrdeutigkeit über mehrere Prompts bzw. Masken abschätzbar | Reflexionen und Bearbeitungsspuren mit starken Kanten ziehen Masken in den Hintergrund |

*Konsequenz*: SAM liefert **Kandidaten-Edits**, keine Labels. Jedes Edit wird pro Komponente gegen das
Ensemble und gegen OOF-Evidenz akzeptiert oder verworfen (Empfehlung R5, Idee N1).

### 2.3 Few-Shot- und In-Context-Segmentierung mit Foundation-Modellen

**Generalisten.** *SegGPT* (Wang et al., ICCV 2023), *PerSAM* (Zhang et al., arXiv 2305.03048) und
*Matcher* (Liu et al., arXiv 2305.13310; ICLR 2024, unverified). Matcher erreicht auf Naturbildern
52,7 mIoU (COCO-20i) bzw. 33,0 (LVIS-92i, +14,4 gegenüber SegGPT) **[Snippet]**. Auf Defekten:
- SegGPT (ViT-L) auf Defect Spectrum 1- bzw. 5-shot ca. **27,7–32,0 mIoU**, auf VISION V1 für PCB_1
  **0,1 IoU** (aus SOFS) **[Snippet]**.
- PerSAM und Matcher schneiden laut Liu et al. (2025) auf den meisten Industriedatensätzen schlecht ab.

**Defekt-spezifische Few-Shot-Arbeiten.**
- *Liu, T. et al. 2025* (arXiv 2502.01216; Optics & Laser Technology 192:114078) stellen einen
  FDS-Benchmark mit 54 Defektkategorien auf. Meta-Learning-FSS ist „generell ungeeignet“,
  Vision-Foundation-Modelle sind vielversprechend, und SAM2 im Video-Tracking-Modus ist überraschend
  effektiv. Ihr trainingsfreies **FM-SAM** (Feature-Matching plus FastSAM-Refinement) liegt in einer
  Ablation bei ca. **34,6–36,5 mIoU** (Mittel über 54 Kategorien) **[Snippet]**.
- *SOFS* (Zhang et al., arXiv 2407.21351; IEEE TII 2025) arbeitet mit DINOv2-B, **ohne Resizing** und
  mit einer „abnormal prior map“ gegen FP. +12,2 % mittlerer mIoU gegenüber SegGPT auf VISION V1, laut
  README Defekte bis < 0,03 % Flächenanteil. Defect Spectrum: 63 % der Typen haben < 2 % Fläche.
- *FS-SAM2* (arXiv 2509.12105): SAM2 plus LoRA für FSS (Naturbild-Benchmarks).
- *No Time to Train* (arXiv 2507.02798): trainingsfrei, SAM2 plus DINOv2-Memory-Bank; 36,8 nAP
  COCO-FSOD, 22,4 nAP Cross-Domain. Schwächen bei mehrdeutigen bzw. ähnlichen Klassen.
- Trainingsfrei mit DINOv3 plus SAM2 für Stahlrohre (Springer-Kapitel 2026, Zahlen nicht geprüft).
- *AnomalyDINO* (Damm et al., WACV 2025): DINOv2-Patch-kNN für Few-Shot-Anomaliedetektion. Ein
  direkter Verwandter unseres PatchCore-Normalitätsmodells; ein Encoder-Tausch dort ist billig.

**Backbones.**
- *DINOv3* (arXiv 2508.10104) bringt Gram Anchoring gegen die Degradation dichter Features bei langem
  Training und eine Hochauflösungsphase. Laut Sekundärquelle liegt es +6 mIoU über DINOv2 auf ADE20K
  (unverified). Varianten **[README]**: ViT-S/16 (21M), S+/16, B/16 (86M), L/16, H+/16, 7B sowie
  **ConvNeXt-T/S/B/L (29–198M), destilliert**. Die Gewichte gibt es erst nach Zugangsantrag, unter
  eigener „DINOv3 License“; für Produktion prüfen.
- *Gharbage et al. 2026* (arXiv 2605.23472; CVPRW 2026) ist die direkteste Evidenz für Industrie:
  ConvNeXt-DINOv3 gegen ImageNet, mit ResNet-50 als Referenz **[Snippet]**:
  - *eingefroren* kein konsistenter Vorteil (Rubber Rings: ImageNet 73,25 vs. DINOv3 72,32 mIoU);
  - *feinjustiert* ist DINOv3 der beste Start für Segmentierung und RGB-Instanzlokalisierung;
  - auf Röntgen ist ImageNet besser (GDXray, feinjustiert 29,74 vs. 27,84 mAP@50).
  Eine Sekundärquelle bemängelt fehlende Seeds und Fehlerbalken.
- Alternativ domänenspezifisches Pretraining: anomaliegeführtes Pretraining auf großem Metalldatensatz
  (arXiv 2509.18919, nicht im Detail geprüft).

**Feature-Upsampler für ViT-Features** (relevant für Patch-14/16-Encoder und kleine Defekte):
FeatUp, LoftUp, JAFAR (arXiv 2506.11136, unverified ID), *AnyUp* (Wimmer et al., ICLR 2026, arXiv
2510.12764; encoder-agnostisch, ohne Retraining pro Encoder), NAF (arXiv 2511.18452) und *Upsample
Anything* (arXiv 2511.16301, CVPR 2026). Laut arXiv 2605.17472 dauert die Inferenz ca. 59–135 ms.
*Einschätzung*: Für 1-px-Kratzer ersetzt kein Upsampler einen echten Full-Res-Pfad. Für kleine
kompakte Defekte (≥ 4–8 px) kann er aber den Decoder vereinfachen.

**Fazit 2.3.** Trainingsfreie und In-Context-Labeler liegen auf Defekten bei ≈ 30–40 mIoU. Das ist
für Produktions-Labels deutlich zu schwach. Ihr Wert für uns liegt woanders: (i) Die *Backbones*
(DINOv2/v3) gehören in die feinjustierten Member. (ii) Die Labeler selbst taugen als *unabhängige
Detektoren*, um übersehene Defekte zu finden, wo Unabhängigkeit wichtiger ist als Präzision
(Idee N2).

### 2.4 Lernen mit verrauschten und inkonsistenten Labels

- **ADELE** (Liu et al., CVPR 2022 Oral, arXiv 2110.03740): Segmentierungsnetze lernen erst saubere
  Pixel und memorieren dann falsche, und zwar **pro Klasse zu unterschiedlichen Zeitpunkten**. ADELE
  erkennt den Beginn der Memorierung pro Klasse, korrigiert Labels im Early-Learning-Fenster und fügt
  Multi-Scale-Konsistenz hinzu. Limitation laut Autoren: Bei stark *strukturiertem* Rauschen, etwa
  systematisch zu breiten Rändern, gibt es kein Early Learning. → Für Pseudo-Labels aus einem
  systematisch verzerrten Teacher ist das relevant.
- **SVLS** (Islam & Glocker, IPMI 2021, arXiv 2104.05788): Label-Smoothing mit Gauß-Kern über die
  Nachbarschaft, also Unsicherheit nur an Rändern. Liefert bessere Randvorhersage und Kalibrierung und
  integriert Multi-Rater natürlich. Kritik („Trust your neighbours“, Murugesan et al.; Venue
  unverified): SVLS fehlt ein Mechanismus, die Constraint gegen das Hauptziel zu gewichten. Die
  Alternative sind Penalty-basierte Constraints auf Logits.
- **Dice Semimetric Losses (DML)** (Wang Z. et al., MICCAI 2023, arXiv 2303.16296): Gängige
  Soft-Dice-Implementierungen sind mit Soft-Labels inkorrekt. DML ist bei harten Labels identisch und
  bei Soft-Labels korrekt. Auf QUBIQ, LiTS und KiTS liefern Soft-Labels (Mittelung,
  Label-Smoothing, KD) mit DML besseren Dice *und* bessere Kalibrierung **[Abstract]**. Code:
  `zifuwanggg/JDTLosses`. → **Für Ensemble-Soft-Targets unmittelbar anwendbar.**
- **T-Loss** (Gonzalez-Jimenez et al., MICCAI 2023 Best Paper, arXiv 2306.00753): NLL einer
  Student-t-Verteilung mit gelerntem Robustheitsparameter. Der Vorteil ist vor allem bei hohem
  Rauschanteil sichtbar (ISIC, Shenzhen).
- **Learning from Disagreement** (Uma et al., JAIR 2021): Soft-Labels schlagen aggregierte Labels,
  *wenn* es viele hochwertige Annotatoren gibt. Sonst ist die Kombination aus Gold- und Soft-Labels am
  besten. → Unsere „Annotatoren“ sind K×M Ensemble-Member. Gold (L) und Soft (U) zu kombinieren ist
  genau dieses Rezept.
- **Labelfehler-Erkennung.** *Rottmann & Reese* (WACV 2023, arXiv 2207.06104) heben Unsicherheit auf
  vorhergesagte Zusammenhangskomponenten und finden so die meisten injizierten Fehler.
  *Lad & Mueller* (ICML-Workshop 2023, arXiv 2307.05080) testen sieben Scores; am besten ist ein
  **Soft-Minimum der Modell-Likelihood für die annotierte Klasse über die Pixel**, als Ranking
  fehlerhafter Bilder. → Direkt auf unsere OOF-Vorhersagen für **L** anwendbar, um inkonsistente
  *Ground-Truth*-Masken zu finden. Bei 5–30 Masken zählt jede.
- **Multi-Annotator-Modelle** (z. B. „Disentangling Human Error from the Ground Truth“, Zhang et al.,
  NeurIPS 2020 – **unverified**) setzen mehrere Annotationen pro Bild voraus. Für uns ist das nur
  relevant, wenn gezielt doppelt annotiert wird; empfohlen für ca. 10 Audit-Bilder, um die
  Randkonvention zu quantifizieren (Idee N4).

**Rezept für das Produktionsmodell (abgeleitet).**
1. **Targets.** Für U: kalibrierte Ensemble-Wahrscheinlichkeit p̂ als Soft-Target in drei Zonen.
   Konfident positiv (Hysterese-Maske) bekommt Ziel p̂ (bzw. ≥ 0,9) und Gewicht 1. Konfident negativ
   bekommt Ziel p̂ (≈ 0) und Gewicht 1. Unsicher bekommt Gewicht w = 1 − u, wobei u die normierte
   Member-Uneinigkeit ist; bei u > u_max wird ignoriert. Für L: GT mit SVLS-Rand (σ an die gemessene
   Randunsicherheit angepasst) und Gewicht ≥ 2.
2. **Loss.** Gewichtete BCE (bzw. T-Loss nur auf U-Pixeln) plus **DML-Soft-Dice** plus ein
   komponenten-normierter Term (blob-artig, siehe 2.5) plus Skeleton Recall für Linien-Klassen.
3. **Ignore-Ring.** Den festen 2-px-Ring durch einen *gemessenen* Ring ersetzen (Idee N4).
4. **Early-Learning-Kontrolle.** Den Fit auf unsicheren Pseudo-Label-Pixeln pro Klasse überwachen.
   Steigt er sprunghaft (ADELE-Signal), stoppen oder Targets durch die aktuelle Vorhersage ersetzen.
5. **Sampling.** Bilder mit Pseudo-Komponenten, seltene Klassen und „harte Negative“ (hoher
   Anomaliescore ohne Label, also Reflexionen) überproportional ziehen.
6. **Validierung.** Das Produktionsmodell sieht ganz L. Ehrliche Schwellen gibt es deshalb nur über
   ein eigenes Cross-Fitting oder über das PPI-Audit-Set, nie über L-in-sample.

### 2.5 Multi-Skalen und Größen-Unbalance

- **Focal Tversky** (Abraham & Khan, ISBI 2019, arXiv 1810.07842): bessere Precision-Recall-Balance
  bei kleinen Läsionen; +25,7 % Dice (BUS 2017) bzw. +3,6 % (ISIC 2018) gegenüber U-Net mit Dice
  **[Snippet]**.
- **Loss-Survey** (arXiv 2312.05391): Dice und Focal schneiden bei kleinen Organen schlechter ab als
  Alternativen wie Lovász, Jaccard und Tversky.
- **Läsions- bzw. Komponenten-Losses.** *blob loss* (Kofler et al., arXiv 2205.08209; IPMI 2023,
  unverified) normiert die Loss pro Zusammenhangskomponente, sodass große Defekte kleine nicht
  überdecken. Gegenüber Soft-Dice ca. +5 % (MS-Läsionen), +3 % (Lebertumoren) und ≈ +2 % (Mikroskopie)
  in Instanz-Metriken. Nachteile: Hyperparameter-sensitiv und mehr Speicher. Eine 2026er
  nnU-Net-Studie (arXiv 2604.08015) zeigt den **Trade-off** deutlich: Small-Lesion-Recall 0,873 mit
  läsionsbasierter Loss gegenüber 0,796 (Dice+CE), 0,834 (Tversky) und 0,831 (Focal Tversky). Die
  Lesion-F1 ist dagegen bei Focal Tversky besser (0,846 vs. 0,757) **[Snippet]**. → Der
  Komponenten-Term erhöht Recall, kostet aber FP. Unsere nachgelagerte konforme Schwelle und das
  FP-Meta-Modell (R4) können das kompensieren.
- **Dünne Strukturen.** *Skeleton Recall Loss* (Kirchhoff et al., ECCV 2024, arXiv 2404.03010)
  berechnet Soft-Recall auf einem vorab berechneten „tubed skeleton“ der GT, auf der CPU im
  Dataloader. Mehr als 90 % weniger Overhead als clDice, **multi-klassenfähig** (clDice ist es aus
  Kostengründen nicht). Getestet auf Straßen, DRIVE, **Betonrissen**, ToothFairy und TopCoW; Basis
  mehrerer Challenge-Siege **[README]**. clDice (Shit et al., CVPR 2021, unverified) ist der
  Vorgänger. → Für DeepCrack, CFD und Kratzerklassen ist das erste Wahl. Mit verrauschten
  Pseudo-Labels sollte nur auf Skeletten konfident positiver Komponenten gearbeitet werden.
- **Lovász-Softmax** (Berman et al., CVPR 2018, unverified) und **Boundary Loss** (Kervadec et al.,
  unverified): In diesem Lauf gab es keine defektspezifische Evidenz; nur als Ablation.
- **Auflösung.** SOFS setzt explizit auf *Non-Resizing* für kleine Defekte, und unser Full-Res-Stem
  folgt demselben Prinzip. *Einschätzung* (nicht recherchiert): Für Bilder ≫ 1024 px ist Tiling mit
  Überlappung und Gauß-gewichtetem Stitching dem Downscaling für kleine Defekte überlegen. Großflächige
  Defekte brauchen aber Kontext. Ein zweiskaliger Ansatz verbindet beides: ein grober Kontextpfad auf
  dem herunterskalierten Bild, dessen Features in die Tile-Inferenz eingespeist werden, oder ein
  FPN/UPerNet-Decoder auf DINOv3-ConvNeXt (Stride 4–32). Das muss auf KolektorSDD2 und MVTec AD 2
  gemessen werden.
- **Metriken.** Pixel-Dice wird von großen Defekten dominiert ([`HANDOFF.md`](../HANDOFF.md), Risiko 4).
  Berichtet werden sollten: Komponenten-Recall und -Precision (Treffer bei Überlapp ≥ x %), **Recall
  pro Größenklasse** (z. B. < 0,01 %, 0,01–0,1 %, 0,1–1 %, > 1 % Bildfläche), clDice bzw.
  Skelett-Recall für Linien-Klassen, Bild-Recall und tol-F1.

### 2.6 Label-freie Performance- und Qualitätsschätzung

| Methode | Prinzip | Evidenz | Passung |
|---|---|---|---|
| **Paarweiser Dice zwischen Ensemble-Membern** (Roy et al. 2019; benchmarkt von Zenk et al., MedIA 2025, arXiv 2406.03323) | Mittelwert des Dice aller Member-Paare pro Bild | im großen Failure-Detection-Benchmark (5–6 Datensätze, realistische Shifts, AURC) **am besten**; Pixel-Konfidenz-*Aggregation* ist entscheidend **[Abstract]** | ++ (K Member vorhanden) |
| **MetaSeg** (Rottmann et al., arXiv 1811.00648; CVPRW 2019) | Meta-Klassifikation pro vorhergesagtem Segment (IoU = 0 vs. > 0) und Meta-Regression des IoU aus Dispersionsmerkmalen | Cityscapes: AUROC 87,7–89,9 %, R² 81,5–84,8 %; bis 91,6 % AUROC inkl. BraTS **[Snippet]** | ++ für FP-Komponenten |
| **RCA** (Valindria et al., TMI 2017, arXiv 1702.03407); In-Context-RCA 2025 (UniverSeg als Reverse-Classifier) | Vorhersage als Pseudo-GT, Reverse-Segmenter auf gelabelte Referenzen anwenden | Multi-Organ, Herz-MRT (4.800 Scans) | o: braucht gute Referenzen (unser L), teuer, FM-Reverse-Segmenter auf Defekten schwach |
| Direkte Dice-Regression (Robinson et al., MICCAI 2018); unsicherheitsbasierte Qualitätsvorhersage (arXiv 2508.01460) | CNN bzw. Bayes-Modell regressiert Dice | medizinisch | o: Trainingsdaten wären nur OOF-L (n klein) |
| **ATC** (Garg et al., ICLR 2022, arXiv 2201.04234) | Schwelle auf Konfidenz so wählen, dass der Anteil darüber auf Quelldaten der Accuracy entspricht | 2–4× genauer als Vorgänger, *Klassifikation* | + adaptierbar auf Komponenten-Precision |
| **Agreement-on-the-Line** (Baek et al., NeurIPS 2022, arXiv 2206.13089) | ID/OOD-Agreement linear mit gleicher Steigung wie die Accuracy | nur für NN-Klassifikatoren gezeigt, „nicht universell“ | ? für Segmentierung, Idee N3 |
| DoC (Guillory et al., ICCV 2021, unverified) | Konfidenzdifferenz Quelle/Ziel | Klassifikation | − (gleiche Verteilung L/U) |
| Konsistenz-Ranking (arXiv 2503.00450) | Stabilität unter Perturbation rankt Modelle quellfrei | biomedizinisch, korreliert stark mit wahrer Leistung | + zur **Modellauswahl** zwischen Runden und Backbones |
| Labelfehler-Scores (Lad & Mueller 2023; Rottmann & Reese 2023) | Soft-Min-Likelihood bzw. Komponenten-Unsicherheit | SYNTHIA, Cityscapes | ++ für Review-Priorisierung |

**Was korreliert am besten mit wahrem Dice bzw. übersehenen Defekten?**
- *Bild-Dice*: Die beste verfügbare Evidenz spricht für **paarweisen Dice zwischen Membern**, nicht
  für mittlere Konfidenz oder Entropie (Zenk et al.).
- *FP-Komponenten*: MetaSeg-artige Merkmale, also Größe, mittlere und maximale Wahrscheinlichkeit,
  Entropie, Member-Stimmenanteil und Randgradient. Bei uns kommen Anomaliescore und
  Copy-Paste-Ähnlichkeit hinzu.
- *Übersehene Defekte*: Hierfür gibt es **keine etablierte Methode**. Alle genannten Schätzer bewerten
  *vorhandene* Vorhersagen. Ein FN hat definitionsgemäß keine Komponente. Brauchbare Signale sind
  (a) unabhängige Detektoren wie Normalitätsmodell oder Few-Shot-FM (Abschnitt 2.3), (b)
  „Near-Miss“-Komponenten zwischen konformer und Hysterese-Schwelle, (c) Member, die als einzige
  feuern, und (d) synthetische Injektion mit bekannter Wahrheit (Idee N2).
- Alle Schätzer müssen **auf OOF kalibriert und per PPI-Audit geprüft** werden. Das ist bereits so
  angelegt.

---

## 3. Vergleichstabelle

Legende: ++ sehr geeignet, + geeignet, o neutral oder unklar, − ungeeignet. Kosten bezogen auf den
Mac mini M4 (MPS): € gering, €€ mittel, €€€ hoch.

| Methode (Jahr) | Kategorie | Belegtes Ergebnis (Datensatz, Metrik) | W | S | M | T | Kosten |
|---|---|---|---|---|---|---|---|
| UniMatch (2023) | SSL W2S | VOC92 75,2 mIoU; ACDC 1 Fall 85,4 Dice | + | o | + | + | €€ |
| UniMatch V2 (2025) | SSL + DINOv2 | VOC92 86,3; CS186 83,6 mIoU | + | o (Patch 14) | + | ++ | €€–€€€ |
| CorrMatch (2024) | Label-Propagation | VOC92 76,4 | + | − (grobe Features) | + | + | €€ |
| AllSpark (2024) | Feature-Regeneration | VOC92 76,1 | o | o | + | + | €€€ |
| SemiVL (2024) | VL-Guidance | VOC92 84,0; COCO 1/512 50,1 | + | − | + | − (Sprache) | €€ |
| PrevMatch (2024/26) | Checkpoint-Ensemble | +1,6 mIoU VOC92, 2,4× schneller | + | o | + | ++ | € |
| U2PL (2022) | unverlässliche Pixel als Negative | VOC und CS (Zahlen nicht geprüft) | + | o | ++ | + | €€ |
| AEL / CAFS | klassen-balanciert | CS, VOC | + | + | ++ | + | € |
| Božič et al. (2021) | Mixed Supervision | KSDD2: 73,3 → 83,2 AP mit 16 Masken | ++ | + | o | + | € |
| HQ-SAM (2023) | Refiner | qualitativ bessere Ränder | o | − dünn | o | + | €€ |
| SAM 3 (2025) | Konzept-Segmentierung | Risse zero-shot 56,5 IoU, R 70,4 % | o | − dünn | + | + | €€€ (848M) |
| SAMRefiner (2025) | Maske → Prompts → SAM | ICLR, Zahlen nicht geprüft | + | o | + (Split-Merge) | ++ | €€ |
| SAM Carries the Burden (2024) | SAM-Pseudo-Label-Refinement | Dice +8–10 Punkte (med.) | ++ | − dünn | o | + | €€ |
| SegGPT / PerSAM / Matcher | In-Context | Defekte ≈ 28–32 mIoU (SegGPT); „schlecht“ | + | − | o | ++ | €€ |
| FM-SAM, SOFS | Few-Shot-Defekt | ≈ 35–37 mIoU (54 Kat.); +12,2 % vs. SegGPT | ++ | + (SOFS) | + | ++ | €€ |
| DINOv3-ConvNeXt (2026) | Backbone | RGB feinjustiert bester Start; Röntgen schlechter | + | + (Stride 4) | + | + | €–€€ |
| ADELE (2022) | verrauschte Labels | VOC (WSSS) SOTA, med. synthetisch | + | o | ++ | + | € |
| SVLS (2021) | Rand-Smoothing | bessere Ränder und Kalibrierung | + | + | + | ++ | € |
| DML (2023) | Soft-Dice | besserer Dice und Kalibrierung | + | + | + | ++ | € |
| T-Loss (2023) | robuste Loss | ISIC, Shenzhen unter Rauschen | + | o | + | + | € |
| blob loss (2022/23) | Komponenten-Loss | +2–5 % Instanz-Metriken | + | ++ | + | + | €€ |
| Skeleton Recall (2024) | Topologie-Loss | 5 Datensätze inkl. Risse; −90 % Overhead | + | ++ (Linien) | ++ | + | € |
| Focal Tversky (2019) | Recall-Bias | +3,6–25,7 % Dice (2D) | + | + | + | + | € |
| Paarweiser Dice (2019/25) | Qualitätsschätzung | bester Failure-Detector (AURC) | ++ | o | + | ++ | € |
| MetaSeg (2019) | FP-Komponenten | AUROC ≈ 88–92 % | + | + | + | ++ | € |
| RCA (2017) | Qualitätsschätzung | Multi-Organ | o | − | o | o | €€€ |
| ATC / AoL (2022) | Accuracy-Schätzung | Klassifikation | o | ? | + | + | € |

---

## 4. Konkrete, priorisierte Empfehlungen für unsere Pipeline

Reihenfolge nach erwartetem Gewinn pro Aufwand und Risiko. Jede Empfehlung wird auf MT, BSData,
DeepCrack und CFD (sowie lokal KSDD2) mit L = 15, 3 Seeds, OOF **und** U-GT bewertet. Gemessen werden
tol-F1, Komponenten-Recall pro Größenklasse und Konformitäts-Check.

| Rang | Maßnahme | Erwarteter Gewinn | Aufwand | Risiko |
|---|---|---|---|---|
| **R1** | **Heterogenes Ensemble mit DINOv3-Encoder.** Pro Fold zwei Member: der vorhandene resnet18d-U-Net und ein DINOv3-ConvNeXt-T/S-U-Net mit Full-Res-Stem. Alternativ ViT-S/16 mit AnyUp bzw. Full-Res-Stem. Fold-Konsistenz bleibt erhalten: Teacher k mit beiden Backbones lehrt Student k. | **hoch** (UniMatch V2: +11 mIoU durch Encoder; DINOv3 laut Industriestudie bester feinjustierter RGB-Start). Zusätzlich echte Diversität für die Uneinigkeitssignale | mittel (Gewichte nach Zugangsantrag, Lizenz prüfen; MPS-Speicher) | Röntgen- bzw. Graubild-Domänen: ImageNet war in einer Studie besser → als *Member*, nicht als Ersatz; die Auswahl erfolgt per OOF |
| **R2** | **Trainingsrezept des Produktionsmodells** (Abschnitt 2.4): Soft-Targets mit Zonen, DML-Soft-Dice, SVLS auf L, Gewicht 1 − u, ADELE-artige Memorierungs-Überwachung, Hard-Negative-Sampling | mittel bis hoch für Labelqualität und Kalibrierung; reduziert systematisch zu breite oder zu schmale Ränder | gering bis mittel (Loss- und Target-Code) | gering. Validierung nur über Audit oder Cross-Fit, nie in-sample |
| **R3** | **Größen- und Topologie-bewusste Losses**: komponenten-normierter Term (blob-artig) und Skeleton Recall für Linienklassen. Tversky mit β > α nur, wenn die konforme Schwelle allein nicht reicht. Dazu Metriken pro Größenklasse | mittel (Recall kleiner Defekte, Konnektivität von Rissen) | gering (Skeletonisierung im Dataloader; CC-Labels gibt es schon) | FP steigen. Das wird über das FP-Meta-Modell (R4) und die Schwellen abgefangen |
| **R4** | **Qualitätsschätzer erweitern**: (a) paarweiser Dice der Member pro Bild, isoton auf OOF-Dice kalibriert. (b) MetaSeg-Komponenten-Meta-Klassifikator (TP/FP) auf OOF-Komponenten mit Merkmalen aus Größe, Konfidenz, Entropie, Stimmenanteil, Anomaliescore und Randkontrast. (c) Labelfehler-Score (Soft-Min-Likelihood) auf **L** zur GT-Prüfung | mittel bis hoch für die *Messbarkeit*; die Review-Queue wird schärfer | gering | n(OOF-Komponenten) ist klein → einfache Modelle (logistisch) und Bootstrap-CIs |
| **R5** | **SAM2- bzw. SAM3-Refiner, gegatet**: Prompts aus Hysterese-Komponenten SAMRefiner-artig (Box mit Kontextrand, distanzgeführte Punkte, Maskenprompt). Akzeptanz nur, wenn (i) die Komponente blob-artig ist (Skelett-Breite > w_min), (ii) IoU(Refined, Ensemble-High-Recall-Maske) ≥ τ, (iii) die Defektklasse auf OOF-L einen Gewinn zeigt. Abweichungsregion → Ignore | mittel für große, kantige Defekte (MT „blowhole“ und „break“, BSData-Pitting?), ≈ 0 oder negativ für Risse | mittel (SAM2-Hiera-S läuft lokal; SAM3 mit 848M offline akzeptabel) | **Ränder verschieben sich zur Objektkante statt zur Annotationskonvention** → messen mit tol-F1 *und* strikter Dice |
| **R6** | **Online-Weak-to-Strong-Konsistenz** innerhalb der Fold-Konsistenz: UniMatch-V2-artiges Complementary Dropout (Feature-Ebene), geometrische und Rausch-Augmentierung, *keine* starke Kontrast- und Helligkeitsveränderung auf Defektpixeln. Konsistenz-Pixel klassen-balanciert ziehen (AEL). PrevMatch-Snapshots von Member k als Zusatz-Teacher | mittel (bei UniMatch-artigen Setups der größte Methoden-Gewinn, bei uns teils schon durch Runde 2 abgedeckt) | mittel (mehrere Forward-Passes) | Confirmation Bias Richtung Hintergrund → Recall pro Runde überwachen und abbrechen, wenn der Konformitäts-Check auf U fällt |
| **R7** | **Hierarchische Pseudo-Labels für Multi-Klasse** (U2PL-Idee): „Defekt“ konfident, Klasse unsicher → positiv für den binären Kopf, ignoriert bzw. negativ für ausgeschlossene Klassen im Klassen-Kopf bzw. in `classify.py` | mittel bei vielen Klassen | gering | gering |
| **R8** | **Few-Shot-FM als unabhängiger Zweitdetektor** (SOFS bzw. FM-SAM mit DINOv3, SAM3 mit Bild-Exemplaren aus L) für die Missed-Defect-Suche und Idee N2. **Nicht** als Labelquelle | gering bis mittel (Recall-Messbarkeit) | mittel | FP-Flut → nur als Review-Signal |

**Ausdrücklich nicht empfohlen:**
- die Pipeline durch UniMatch V2 „as is“ zu ersetzen: Es gibt dort keine Kalibrierung, keine
  Recall-Garantie, und bei unserer Vordergrund-Seltenheit fehlt jeder Beleg;
- SegGPT, PerSAM oder Matcher als Labeler;
- Textprompts (SAM3, CLIP) als Defektdefinition;
- ungegatetes SAM-Refinement für Risse oder Kratzer;
- CRF-Glättung bei kontrastarmen Defekten (allenfalls als Ablation).

**Antwort auf die Backbone-Frage:** Kein vollständiger Ersatz. DINOv3 als zusätzlicher Member
(R1), die Auswahl und Gewichtung pro Task erfolgt über OOF-Leistung. Das ist task-agnostisch, weil
die Entscheidung datengetrieben fällt. Wenn ein Task Graubilder bzw. Röntgen ist, darf das Ensemble
den DINOv3-Member abwählen.

---

## 5. Neue Forschungsideen (über die Literatur hinaus)

### N1 – Konform gegatete Refinement-Edits („Refine, but certify recall“)
**Idee.** Jede Refinement-Operation (SAM, Feature-Propagation, CRF) wird als *Edit* einer Komponente
betrachtet: Pixel werden hinzugefügt oder entfernt. Auf OOF-L lässt sich für jede Edit-Art und jedes
Merkmalsbin (Größe, Elongation, Kontrast) messen, welcher Anteil echter Defektpixel durch das Edit
verloren geht (ΔRecall) und welcher Anteil FP-Pixel entfernt wird. Ein Edit wird auf U nur akzeptiert,
wenn eine **konforme bzw. Clopper-Pearson-Obergrenze des Recall-Verlusts** im Bin unter ε liegt.
Andernfalls bleibt die Originalmaske, und die strittige Differenzregion wird Ignore.
**Warum neu.** SAM-Refinement-Arbeiten (SAMRefiner, SAM Carries the Burden) akzeptieren global. Unsere
Pipeline hat bereits konforme Werkzeuge und OOF-Daten. Das Gating macht den Refiner
recall-zertifiziert und routet automatisch dünne Strukturen weg.
**Experiment.** MT, BSData, DeepCrack, CFD, KSDD2; L = 15, 3 Seeds. Varianten: (a) kein Refinement,
(b) SAM2 immer, (c) HQ-SAM immer, (d) gegatet (ε = 0,02). Metriken auf U-GT: Komponenten-Recall,
Pixel-Recall, strikter Dice, tol-F1.
**Falsifikation.** Die Hypothese gilt als widerlegt, wenn (d) beim realisierten Recall-Verlust > ε
liegt (Garantie verletzt) **oder** bei tol-F1 nicht ≥ max(a, b, c) − 0,01 erreicht.

### N2 – Recall-Schätzung auf U durch synthetische Injektion plus Capture–Recapture
**Idee.** Der Recall auf U ist die kritischste und am schlechtesten messbare Größe. Er wird aus zwei
Quellen geschätzt.
(i) *Injektion*: Residual-Copy-Paste-Defekte (vorhanden) werden mit bekannter Lage in eine Zufallsmenge
von U-Bildern eingefügt, über ein Raster aus Gain, Größe und Orientierung. Gemessen wird die
Detektionsrate d(Gain, Größe, Klasse) des *finalen* Labelers. Diese Rate wird per Importance-Weighting
auf die Merkmalsverteilung realer Defekte übertragen, geschätzt aus detektierten Defekten plus
Audit-Set.
(ii) *Capture–Recapture*: Zwei möglichst unabhängige Detektoren (Segmentierungs-Ensemble vs.
Normalitätsmodell bzw. Few-Shot-FM, R8) „fangen“ Defekte. Aus der Überlappung folgt eine
Lincoln-Petersen-Schätzung bzw. bei Abhängigkeit eine Chapman- oder log-lineare Schätzung der
Gesamtzahl. Die Abhängigkeit wird auf den injizierten Defekten gemessen, deren Wahrheit bekannt ist.
Beide Schätzer werden mit dem PPI-Audit kombiniert, das als Rektifizierer dient.
**Warum neu.** Etablierte Qualitätsschätzer (Abschnitt 2.6) bewerten nur vorhandene Komponenten.
Capture–Recapture ist in der Ökologie und Epidemiologie Standard, für Pixel-Auto-Labeling ist es
unseres Wissens nicht beschrieben (nicht erschöpfend geprüft).
**Experiment.** Auf allen Benchmarks mit vollständiger U-GT werden wahrer Komponenten-Recall und die
Schätzungen (i), (ii), (i+ii+PPI) verglichen, außerdem die Breite des PPI-Intervalls mit und ohne diese
Prädiktoren bei n_audit = 10/20/40.
**Falsifikation.** Widerlegt, wenn der mittlere absolute Fehler der Recall-Schätzung über
Datensätze und Seeds nicht kleiner ist als beim naiven Schätzer (OOF-Recall auf L) **oder** wenn die
PPI-Intervalle mit Prädiktor nicht schmaler werden bei gleicher Abdeckung (≥ 90 %).

### N3 – Cross-fitted Agreement-on-the-Line für Komponenten-Metriken
**Idee.** Agreement-on-the-Line (AoL) ist für Klassifikatoren belegt. Wir haben ein natürliches
Modell-Ensemble: K Folds × M Backbones × Runden, dazu PrevMatch-Snapshots. Für jedes Member-Paar wird
die *Komponenten-Agreement* (F1 zwischen den Komponentenmengen zweier Member) auf OOF-L und auf U
berechnet, ebenso die *Komponenten-F1 gegen GT* auf OOF-L. Liegen die Paare (Agreement_L,
Agreement_U) auf einer Geraden, schätzt deren Steigung bzw. Bias die F1 auf U aus der F1 auf L. Das
ist eine Domänen-Shift-Korrektur zwischen L und U, etwa wenn U andere Chargen oder Kameras enthält.
**Warum neu.** AoL ist in der Segmentierung und auf Komponentenebene nicht untersucht. Bei uns ist
der Shift zwischen L und U oft real (L wurde gezielt mit Defekten ausgewählt).
**Experiment.** Mindestens 16 Member pro Task (Backbones × Folds × Snapshots). Dabei wird der
Pearson-Wert der Linearität gemessen und der Schätzfehler der U-F1 mit ATC-Adaption und naiver
OOF-F1 verglichen. Shifts werden simuliert: L aus einer BSData-Sequenz, U aus anderen.
**Falsifikation.** Widerlegt, wenn R² der Agreement-Linie < 0,6 ist **oder** der Schätzfehler
nicht kleiner als bei naiver OOF-F1 ausfällt.

### N4 – Gemessene Randkonvention statt fester Ignore-Ring
**Idee.** Defektgrenzen sind subjektiv. Die Konvention des Annotators wird pro Task gelernt.
(i) Auf OOF-L wird der *signierte Randabstand* zwischen der kalibrierten 0,5-Konturlinie und der
GT-Kontur gemessen, als Funktion des lokalen Kontrasts und Gradienten. So entsteht ein
„Konventions-Offset“-Modell, z. B. „Annotator zeichnet ca. 1,5 px außerhalb der Gradientenkante“.
(ii) Pseudo-Label-Ränder auf U werden um diesen Offset korrigiert.
(iii) Die Ignore-Ring-Breite wird pro Pixel aus der Streuung dieses Offsets gesetzt statt fest auf
2 px.
(iv) Optional werden ca. 10 Audit-Bilder doppelt annotiert, um die Inter-Rater-Streuung zu
quantifizieren. Sie bildet die Untergrenze jeder sinnvollen Randgenauigkeit.
**Warum neu.** SVLS und Multi-Rater-Methoden *modellieren* Randunsicherheit, aber keine
*systematische, lernbare Konvention* aus wenigen Masken. Auch SAM-Ränder ließen sich damit auf die
Konvention „zurückbiegen“, als Kombination mit N1.
**Experiment.** Auf MT und DeepCrack werden die GT-Masken synthetisch dilatiert oder erodiert
(±1, ±2 px), um unterschiedliche Konventionen zu simulieren. Gemessen wird, ob (a) der gelernte
Offset die simulierte Verschiebung zurückgewinnt und (b) der strikte Dice auf U gegenüber festem Ring
steigt.
**Falsifikation.** Widerlegt, wenn der geschätzte Offset um mehr als 0,5 px vom simulierten
abweicht oder der strikte Dice nicht um ≥ 0,02 steigt.

---

## 6. Offene Lücken und Grenzen dieses Reviews

- **Keine** gefundene Arbeit evaluiert SSSS im Regime 5–30 Masken plus seltene Defekte in U. Alle
  Übertragungsaussagen sind deshalb Hypothesen, die unsere Benchmarks prüfen müssen.
- Volltexte waren nicht abrufbar. Zahlen mit **[Snippet]** sind vor einer Veröffentlichung zu prüfen,
  besonders FM-SAM (34,6–36,5 mIoU) und die SAM2-Werte auf Tile und Kolektor aus arXiv 2502.01216,
  deren Spaltenbedeutung unklar blieb und die deshalb nicht übernommen wurden.
- Nicht mehr recherchiert (Suchbudget erschöpft): Tiling vs. Downscaling, MVTec AD 2 (Metriken,
  Befunde), CascadePSP, CC-Metrics bzw. „Every Component Counts“, Lovász und Boundary Loss auf
  Defekten.
- Lizenzen: Für SAM 3 („SAM License“) und DINOv3 („DINOv3 License“) muss vor dem Produktiveinsatz
  geprüft werden, ob mit den Modellen erzeugte Labels kommerziell nutzbar sind.

---

## 7. Referenzen

Format: Erstautor et al., Titel, Venue/Jahr, ID/URL. „(unverified)“ heißt: bibliographische Angabe
aus Vorwissen, in diesem Lauf nicht bestätigt.

**Semi-supervised Segmentierung**
1. Yang, L. et al. – Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation (UniMatch). CVPR 2023. arXiv:2208.09910; github.com/LiheYoung/UniMatch
2. Yang, L., Zhao, Z., Zhao, H. – UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation. TPAMI 2025. arXiv:2410.10777; github.com/LiheYoung/UniMatch-V2
3. Sun, B. et al. – CorrMatch: Label Propagation via Correlation Matching for Semi-Supervised Semantic Segmentation. CVPR 2024. arXiv:2306.04300; github.com/BBBBchan/CorrMatch
4. Wang, H. et al. – AllSpark: Reborn Labeled Features from Unlabeled in Transformer for Semi-Supervised Semantic Segmentation. CVPR 2024 (Erstautor unverified; Zahlen aus dem UniMatch-V2-README)
5. Hoyer, L. et al. – SemiVL: Semi-Supervised Semantic Segmentation with Vision-Language Guidance. ECCV 2024. arXiv:2311.16241 (Erstautor unverified)
6. Shin, W. et al. – PrevMatch: Revisiting and Maximizing Temporal Knowledge in Semi-Supervised Semantic Segmentation. WACV 2026. arXiv:2405.20610
7. Ishikawa, H., Aoki, Y. – BoundMatch: Boundary detection applied to semi-supervised segmentation. arXiv:2503.23519 (2025)
8. (Autoren nicht geprüft) – Leveraging Out-of-Distribution Unlabeled Images: Semi-Supervised Semantic Segmentation with an Open-Vocabulary Model (SemiOVS). arXiv:2507.03302 (2025)
9. (Autoren nicht geprüft) – Knowledge Consultation for Semi-Supervised Semantic Segmentation. arXiv:2503.10693 (2025)
10. (Autoren nicht geprüft) – S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing. arXiv:2508.12409 (2025)
11. Wang, Y. et al. – Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels (U2PL). CVPR 2022. arXiv:2203.03884 (Erstautor-Initiale unverified)
12. Rizve, M. N. et al. – In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework for Semi-Supervised Learning (UPS). ICLR 2021. arXiv:2101.06329
13. Hu, H. et al. – Semi-Supervised Semantic Segmentation via Adaptive Equalization Learning (AEL). NeurIPS 2021. arXiv:2110.05474 (Erstautor unverified)
14. (Autoren nicht geprüft) – CAFS: Class Adaptive Framework for Semi-Supervised Semantic Segmentation. arXiv:2303.11606
15. Guo, L.-Z., Li, Y.-F. – Class-Imbalanced Semi-Supervised Learning with Adaptive Thresholding. ICML 2022, PMLR 162. proceedings.mlr.press/v162/guo22e
16. (Autoren nicht geprüft) – Accuracy Improvement of Semi-Supervised Segmentation Using Supervised ClassMix and Sup-Unsup Feature Discriminator. arXiv:2604.07122 (2026)
17. Wang, H., Li, X. – DHC: Dual-debiased Heterogeneous Co-training Framework for Class-imbalanced Semi-supervised Medical Image Segmentation. MICCAI 2023 (unverified)
18. (Autoren nicht geprüft) – A Semantic Knowledge Complementarity based Decoupling Framework for Semi-supervised Class-imbalanced Medical Image Segmentation. CVPR 2025. cvpr.thecvf.com/virtual/2025/poster/33219
19. Božič, J., Tabernik, D., Skočaj, D. – Mixed supervision for surface-defect detection: from weakly to fully supervised learning. Computers in Industry 2021. arXiv:2104.06064
20. (Autoren nicht geprüft) – Efficient and accurate semi-supervised semantic segmentation for industrial surface defects. Scientific Reports 2024. doi:10.1038/s41598-024-72579-6
21. (Autoren nicht geprüft) – Semi-supervised surface defect segmentation of aluminum strips based on self-attention consistency and cross-view pseudo supervision. Advanced Engineering Informatics 2025. sciencedirect S1474034625002824
22. (Autoren nicht geprüft) – A lightweight semi-supervised distillation framework for hard-to-detect surface defects in the steel industry. Expert Systems with Applications 2025. sciencedirect S0957417425031045
23. Xiang, C. et al. – Semi-supervised learning framework for crack segmentation based on contrastive learning and cross pseudo supervision (SemiCrack). Measurement 2023. sciencedirect S0263224123006553 (Initiale unverified)
24. (Autoren nicht geprüft) – Research on Concrete Crack Damage Assessment Method Based on Pseudo-Label Semi-Supervised Learning. Buildings 2025. doi:10.3390/buildings15152726
25. (Autoren nicht geprüft) – Deep Learning for Crack Detection: A Review of Learning Paradigms, Generalizability, and Datasets. arXiv:2508.10256 (2025)

**SAM-Familie und Refinement**
26. Kirillov, A. et al. – Segment Anything. ICCV 2023. arXiv:2304.02643 (unverified ID)
27. Ravi, N. et al. – SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714 (unverified ID)
28. Meta SAM-3-Team (Erstautor nicht geprüft) – SAM 3: Segment Anything with Concepts. arXiv:2511.16719 (2025); github.com/facebookresearch/sam3 (inkl. SAM 3.1, 03/2026)
29. Ke, L. et al. – Segment Anything in High Quality (HQ-SAM). NeurIPS 2023. arXiv:2306.01567
30. Lin, Y. et al. – SAMRefiner: Taming Segment Anything Model for Universal Mask Refinement. ICLR 2025. arXiv:2502.06756; github.com/linyq2117/SAMRefiner
31. Keuth, R. et al. – SAM Carries the Burden: A Semi-Supervised Approach Refining Pseudo Labels for Medical Segmentation. arXiv:2411.12602 (2024)
32. (Autoren nicht geprüft) – A SAM-guided and Match-based Semi-Supervised Segmentation Framework for Medical Imaging (SAMatch). arXiv:2411.16949
33. (Autoren nicht geprüft) – S⁴M: Boosting Semi-Supervised Instance Segmentation with SAM. arXiv:2504.05301
34. (Autoren nicht geprüft) – SSL-MedSAM2: A Semi-supervised Medical Image Segmentation Framework Powered by Few-shot Learning of SAM2. arXiv:2512.11548
35. Ma, J. et al. – Segment anything in medical images (MedSAM). Nature Communications 2024. doi:10.1038/s41467-024-44824-z (Erstautor unverified)
36. (Autoren nicht geprüft) – Generalist Vision Foundation Models for Medical Imaging: A Case Study of SAM on Zero-Shot Medical Segmentation. PMC10252742
37. (Autoren nicht geprüft) – Semantic-Edge Response Decoding of SAM3 for Zero-Shot Crack Segmentation (SERD). arXiv:2607.12292 (2026); github.com/xauat-liushipeng/SERD
38. (Autoren nicht geprüft) – SAM3-LoRA: Parameter-Efficient Adaptation of a Concept-Promptable Foundation Model for Multi-Class Structural Defect Segmentation. arXiv:2609.00469 (2026)
39. (Autoren nicht geprüft) – Prompt Engineering in Segment Anything Model: Methodologies, Applications, and Emerging Challenges. arXiv:2507.09562
40. (Autoren nicht geprüft) – Image segmentation for thin structures using a zero-shot learner. Int. J. Information Technology 2024. doi:10.1007/s41870-024-02215-z
41. (Autoren nicht geprüft) – SAM Era: Can It Segment Any Industrial Surface Defects? 2024 (ResearchGate 379334001)
42. (Autoren nicht geprüft) – DefectSAM: Prototype Prompt Guided SAM for Few-Shot Defect Segmentation. IEEE 2025 (ieeexplore 10912073)
43. (Autoren nicht geprüft) – SAID: Segment All Industrial Defects with Scene Prompts. Sensors 2025. doi:10.3390/s25164929
44. (Autoren nicht geprüft) – Leveraging Vision Foundation Model via PConv-Based Fine-Tuning with Automated Prompter for Defect Segmentation. PMC12031572
45. (Autoren nicht geprüft) – P³-SAM: SAM with Perceptual Parallel Prompt for Few-Shot Strip Steel Surface Defect Segmentation. arXiv:2609.21424 (2026)
46. (Autoren nicht geprüft) – DCM-SAM: Defect-Conditioned Mixture of LoRA Experts for NPU-Deployed AM Defect Segmentation. arXiv:2609.38811 (2026)
47. (Autoren nicht geprüft) – Boosting Segment Anything Model to Generalize Visually Non-Salient Scenarios (VNS-SAM). arXiv:2601.00537 (2026)
48. Ge, K. et al. – Fine-tuning vision foundation model for crack segmentation in civil infrastructures (CrackSAM). arXiv:2312.04233 (Erstautor-Initiale unverified)
49. (Autoren nicht geprüft) – Segment Any Crack: Deep Semantic Segmentation Adaptation for Crack Detection. arXiv:2504.14138
50. (Autoren nicht geprüft) – Distribution-aware Noisy-label Crack Segmentation. arXiv:2410.09409
51. Wang, M. et al. – SegRefiner: Towards Model-Agnostic Segmentation Refinement with Discrete Diffusion Process. NeurIPS 2023. arXiv:2312.12425
52. (Autoren nicht geprüft) – Inspiring the Next Generation of Segment Anything Models: Comprehensively Evaluate SAM and SAM 2 with Diverse Prompts Towards Context-Dependent Concepts under Different Scenes. arXiv:2412.01240

**Few-Shot, In-Context, Backbones**
53. Wang, X. et al. – SegGPT: Towards Segmenting Everything In Context. ICCV 2023 (unverified ID: arXiv:2304.03284)
54. Zhang, R. et al. – Personalize Segment Anything Model with One Shot (PerSAM). arXiv:2305.03048 (Venue ICLR 2024 unverified)
55. Liu, Y. et al. – Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching. arXiv:2305.13310 (Venue ICLR 2024 unverified)
56. Liu, T. et al. – Exploring Few-Shot Defect Segmentation in General Industrial Scenarios with Metric Learning and Vision Foundation Models. Optics & Laser Technology 192 (2025) 114078. arXiv:2502.01216
57. Zhang, Z. et al. – Small Object Few-shot Segmentation for Vision-based Industrial Inspection (SOFS). IEEE TII 2025. arXiv:2407.21351; github.com/zhangzilongc/SOFS
58. (Autoren nicht geprüft) – FS-SAM2: Adapting Segment Anything Model 2 for Few-Shot Semantic Segmentation via Low-Rank Adaptation. arXiv:2509.12105
59. Espinosa, M. et al. – No time to train! Training-Free Reference-Based Instance Segmentation. arXiv:2507.02798 (2025) (Erstautor aus Repo-Namen abgeleitet)
60. Damm, S. et al. – AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2. WACV 2025
61. Siméoni, O. et al. – DINOv3. arXiv:2508.10104 (2025); github.com/facebookresearch/dinov3 (Erstautor unverified)
62. Oquab, M. et al. – DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193 (unverified ID)
63. Gharbage, (Vorname nicht geprüft) et al. – Rethinking Transfer Learning for Industrial Inspection: DINOv3 vs. ImageNet Pretraining Across RGB and X-ray Tasks. CVPR Workshops 2026. arXiv:2605.23472
64. Wimmer, T. et al. – AnyUp: Universal Feature Upsampling. ICLR 2026. arXiv:2510.12764; github.com/wimmerth/anyup
65. (Autoren nicht geprüft) – JAFAR: Jack up Any Feature at Any Resolution. arXiv:2506.11136 (unverified ID)
66. (Autoren nicht geprüft) – NAF: Zero-Shot Feature Upsampling via Neighborhood Attention Filtering. arXiv:2511.18452
67. Seo, (Vorname nicht geprüft) et al. – Upsample Anything. CVPR 2026. arXiv:2511.16301
68. (Autoren nicht geprüft) – Effizienzvergleich von Feature-Upsamplern. arXiv:2605.17472 (2026)
69. (Autoren nicht geprüft) – Training-Free Few-Shot Defect Segmentation Framework for Steel Tubes Surfaces Based on DINOv3 and SAM2. Springer 2026 (doi:10.1007/978-981-95-9773-4_55)
70. (Autoren nicht geprüft) – Advancing Metallic Surface Defect Detection via Anomaly-Guided Pretraining on a Large Industrial Dataset. arXiv:2509.18919

**Verrauschte Labels**
71. Liu, S. et al. – Adaptive Early-Learning Correction for Segmentation from Noisy Annotations (ADELE). CVPR 2022. arXiv:2110.03740; github.com/Kangningthu/ADELE
72. Islam, M., Glocker, B. – Spatially Varying Label Smoothing: Capturing Uncertainty from Expert Annotations. IPMI 2021. arXiv:2104.05788
73. Murugesan, B. et al. – Trust your neighbours: Penalty-based constraints for model calibration (Venue unverified; MedIA-Kalibrierungsarbeit 2024)
74. Wang, Z. et al. – Dice Semimetric Losses: Optimizing the Dice Score with Soft Labels. MICCAI 2023. arXiv:2303.16296
75. Gonzalez-Jimenez, A. et al. – Robust T-Loss for Medical Image Segmentation. MICCAI 2023. arXiv:2306.00753
76. Uma, A. et al. – Learning from Disagreement: A Survey. JAIR 2021
77. Rottmann, M., Reese, M. – Automated Detection of Label Errors in Semantic Segmentation Datasets via Deep Learning and Uncertainty Quantification. WACV 2023. arXiv:2207.06104
78. Lad, V., Mueller, J. – Estimating label quality and errors in semantic segmentation data via any model. ICML 2023 DMLR Workshop. arXiv:2307.05080
79. Zhang, L. et al. – Disentangling Human Error from the Ground Truth in Segmentation of Medical Images. NeurIPS 2020 (unverified)

**Losses und Multi-Skalen**
80. Kirchhoff, Y. et al. – Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures. ECCV 2024. arXiv:2404.03010; github.com/MIC-DKFZ/Skeleton-Recall
81. Kofler, F. et al. – blob loss: instance imbalance aware loss functions for semantic segmentation. arXiv:2205.08209 (IPMI 2023 unverified)
82. Abraham, N., Khan, N. M. – A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation. ISBI 2019. arXiv:1810.07842
83. (Autoren nicht geprüft) – Loss Functions in the Era of Semantic Segmentation: A Survey and Outlook. arXiv:2312.05391
84. (Autoren nicht geprüft) – Component-Adaptive and Lesion-Level Supervision for Improved Small Structure Segmentation in Brain MRI. arXiv:2604.08015 (2026)
85. Shit, S. et al. – clDice: a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. CVPR 2021 (unverified)
86. Berman, M. et al. – The Lovász-Softmax Loss. CVPR 2018 (unverified)

**Qualitäts- und Performance-Schätzung**
87. Valindria, V. et al. – Reverse Classification Accuracy: Predicting Segmentation Performance in the Absence of Ground Truth. IEEE TMI 2017. arXiv:1702.03407
88. (Autoren nicht geprüft) – In-Context Reverse Classification Accuracy: Efficient Estimation of Segmentation Quality without Ground-Truth. 2025 (ResearchGate 389648345)
89. Robinson, R. et al. – Real-time Prediction of Segmentation Quality. MICCAI 2018
90. (Autoren nicht geprüft) – Uncertainty-Aware Segmentation Quality Prediction via Deep Learning Bayesian Modeling. arXiv:2508.01460
91. Zenk, M. et al. – Comparative Benchmarking of Failure Detection Methods in Medical Image Segmentation: Unveiling the Role of Confidence Aggregation. Medical Image Analysis 101 (2025) 103392. arXiv:2406.03323; github.com/MIC-DKFZ/segmentation_failures_benchmark
92. Rottmann, M. et al. – Prediction Error Meta Classification in Semantic Segmentation (MetaSeg). arXiv:1811.00648; Rottmann, M., Schubert, M. – Uncertainty Measures and Prediction Quality Rating for the Semantic Segmentation of Nested Multi Resolution Street Scene Images. CVPRW 2019, arXiv:1904.04516 (Titel von 1811.00648 unverified)
93. Garg, S. et al. – Leveraging Unlabeled Data to Predict Out-of-Distribution Performance (ATC). ICLR 2022. arXiv:2201.04234
94. Baek, C. et al. – Agreement-on-the-Line: Predicting the Performance of Neural Networks under Distribution Shift. NeurIPS 2022. arXiv:2206.13089
95. Guillory, D. et al. – Predicting with Confidence on Unseen Distributions (DoC). ICCV 2021 (unverified)
96. (Autoren nicht geprüft) – Unsupervised Source-Free Ranking of Biomedical Segmentation Models Under Distribution Shift. arXiv:2503.00450 (2025)
97. Xie, R. et al. – Leveraging Gradients for Unsupervised Accuracy Estimation Under Distribution Shift. TMLR 2025 (Initiale unverified)
98. Roy, A. G. et al. – Bayesian QuickNAT (paarweiser Dice als Qualitätsmaß). NeuroImage 2019 (unverified; zitiert nach Zenk et al.)
