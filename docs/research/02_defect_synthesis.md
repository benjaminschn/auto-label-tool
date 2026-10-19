# 02 – Defektsynthese und generative Augmentation für Few-Shot-Defektsegmentierung

Stand: 18.10.2026 · Literatur-Review für `auto-label-tool` · Thema: synthetische Defekte (alle Größen,
mehrere Klassen) für Training, Qualitätskontrolle und Kalibrierung.

> **Quellenlage und Markierungen.** In dieser Umgebung waren arxiv.org, thecvf.com, ojs.aaai.org u. a.
> nicht direkt abrufbar. Ich konnte nur Suchmaschinen-Auszüge (Abstracts, Tabellen-Snippets) und
> offizielle GitHub-READMEs lesen. Gegen Ende war außerdem das Suchbudget aufgebraucht. Markierung der Zahlen:
> **[P]** = aus Abstract oder Textauszug des Papers, **[R]** = aus dem offiziellen README,
> **[S]** = nur aus einer Sekundärquelle (Zusammenfassung) und damit unverifiziert. Vor einer
> Verwendung in Publikationen oder Entscheidungsvorlagen bitte gegen die PDFs prüfen. Werke, die ich
> in dieser Recherche nicht selbst gefunden habe, sind mit „(unverified)“ markiert.

---

## Executive Summary

1. **Generative Few-Shot-Defektgenerierung** (AnomalyDiffusion → DualAnoDiff → SeaS → GroundingAnomaly,
   FAST, 2024–2026) verbessert sich von Jahr zu Jahr, aber fast immer **im Vergleich mit anderen
   Generatoren** im selben MVTec-Protokoll. Ein Beispiel: SeaS erreicht 57,7 mittlere IoU, AnomalyDiffusion 46,5 [R].
   Einen Vergleich mit einer **starken Real-Daten-Baseline** habe ich nicht gefunden, also mit wenigen
   echten Defekten plus gutem Copy-Paste auf vielen Normalbildern, wie es unsere Pipeline schon macht.
   Ebenso fehlt eine Prüfung mit ≤ 30 Masken über viele verschiedene Aufgaben. **Der Nutzen für unser
   Setting ist damit unbewiesen und muss selbst gemessen werden.**
2. **Mask-Bild-Alignment ist der Schwachpunkt für die Labelqualität.** Die Maske kommt aus einer von
   drei Quellen: (a) sie ist die Bedingung, und der Generator füllt sie zu wenig oder zu weit,
   (b) sie wird nachträglich segmentiert (U-2-Net, Feature-Fusion, Grounding-DINO), oder (c) es gibt nur
   eine Box (AnoGen). Eine Standardmetrik fehlt. Die beste gefundene Zahl ist eine Pixel-AUC von 0,929 für
   die Maskenpipeline von MIRAGE [P]; für pixelgenaue Labels reicht das nicht. Wie wichtig Alignment ist,
   zeigt AnomalyDiffusion: Ohne seinen Alignment-Mechanismus fällt die Pixel-AP von 81,4 auf 64,9 [P].
3. **Cross-Fitting ist Pflicht.** Jede Synthesekomponente, die aus den gelabelten Bildern L lernt, muss
   foldspezifisch sein. Sonst werden die OOF-Vorhersagen verzerrt, und mit ihnen Kalibrierung,
   konforme Recall-Schwelle und PPI-Selbsteinschätzung. Die Defektbank ist im Code bereits
   foldspezifisch (`train_L`). Ein Diffusionsmodell bräuchte dagegen K Fine-Tunings pro Aufgabe und
   Klasse; DualAnoDiff braucht ≈ 4,5 V100-Stunden pro Anomalietyp [P].
4. **Nicht-generative Synthese bleibt der beste Aufwand-Nutzen-Punkt.** Gradient-Domain-Blending schlägt
   hartes Einfügen: NSA erreicht 97,2 Image-AUROC, 2,0 Punkte mehr als CutPaste [P]. Feature-Space-Synthese
   (SimpleNet, GLASS, SuperSimpleNet) ist billig und hat exakte Masken, löst aber dünne 1–3-px-Defekte
   schlecht auf. SuperSimpleNet kann echte Labels und synthetische Anomalien mischen (KSDD2: AP 97,4 [S]).
5. **Große und kontextabhängige Defekte sind die größte Lücke unserer Pipeline.** Große Defekte werden
   derzeit gar nicht synthetisiert. Günstiger Weg: Residuen in mehreren Frequenzbändern bzw. im
   Gradientenbereich übertragen, mit Wahl zwischen additivem und multiplikativem Modell, und den
   Einfügeort über die Features des Normalitätsmodells passend zum Kontext wählen. Mask-geführtes
   Diffusions-Inpainting (MAGIC/DefectFill-Typ) bleibt eine GPU-Option.
6. **Physikbasiertes Rendering** hilft bei geometrieabhängigen Defekten, wenn es CAD- und Materialmodelle
   gibt (Fulir 2023, SYNOSIS, ScratchSim). Der Aufwand pro Aufgabe widerspricht aber der
   Aufgaben-Agnostik. Auch die Industriepaper zeigen: rein synthetisches Training liegt unter Real-Daten;
   Mischen oder Vortrainieren hilft vor allem bei knappen Real-Daten (SynSur auf **BSData/MSD**, ScratchSim) [P].
7. **Qualitätskontrolle** läuft in der Literatur über generische Maße (CLIP, DreamSim, LPIPS). Die kennen
   unsere Domäne nicht. Besser sind domäneninterne Signale: Übereinstimmung mit dem Normalitätsmodell,
   erhaltener Hintergrund, tatsächlich sichtbarer Kontrast und ein Real-vs-Synthetik-Diskriminator (C2ST)
   als Shortcut-Monitor. **Nie danach filtern, ob das aktuelle Modell den Defekt findet:** Das entfernt
   genau die schweren Fälle und kostet Recall.
8. **Mehr Synthetik hilft nicht unbegrenzt.** ASBench: Ab einer Schwelle ändert ein höherer Anteil kaum
   noch etwas; nahe 100 % lernt das Modell synthetische Muster [S/P]. Echte Daten müssen als Anker
   bleiben, und `p_synth` sollte durchgemessen werden.
9. **Kalibrierung:** Synthetische Defekte dürfen echte OOF-Defekte in der konformen Recall-Garantie
   **nicht ersetzen**. Erlaubt sind sie nur über Verfahren mit eigener Garantie (SPI, NeurIPS 2025) oder
   als Modell, dessen Verzerrung mit echten Daten korrigiert wird (PPI). Für **Rangfolgen**, also
   Modell- und Hyperparameterwahl, ist synthetische Validierung brauchbar (SWSA) [P].
10. **Top-Empfehlungen:** (1) ehrlicher Evaluations-Harness, dazu unsere Baseline im Standard-MVTec-Protokoll
    gegen die Literaturzahlen stellen; (2) Labels aus dem tatsächlich erzeugten Signal, Null-Synthese-
    Negative und C2ST; (3) Residual-Transfer v2 für große Defekte mit kontextpassender Platzierung;
    erst danach optional Diffusion.

---

## 1. Rolle der Synthese in unserer Pipeline

Die Synthese hat bei uns drei Aufgaben, die man getrennt bewerten muss:

* **(A) Positiv-Diversität:** Was ein Defekt ist, in mehr Formen, Größen, Orientierungen und Kontrasten,
  als L enthält.
* **(B) Negativ-Diversität:** Was kein Defekt ist. Die Defekte erscheinen auf tausenden echten
  Hintergründen mit Reflexen, Textur und Staub. Das ist die eigentliche Stärke des Residual-Copy-Paste.
* **(C) Mess- und Kalibrierdaten:** Recall, Schwellen und Selbsteinschätzung (Abschnitt 7).

**Was der Code heute macht** (`autolabel/augment.py`, `autolabel/train.py`):

* Die Bank wird pro Member nur aus `train_L` gebaut (Cross-Fitting ✔).
* Residuen entstehen als Pixel minus Telea-Inpainting.
* Das Einfügen ist additiv: Gain 0,6–1,3, Skalierung 0,75–1,3, Rotation und Flip, weiche Maske σ = 1.
* Der Einfügeort ist gleichverteilt.
* Das Label ist die transformierte Quellmaske.
* Defekte mit mehr als 6000 px oder mehr als 80 % Bildausdehnung werden nicht eingefügt
  (`paste_large=False`).
* Prozedurale Bezier-Kratzer und -Blobs übernehmen den Kontrast der gelabelten Defekte.

**Vier Lücken, die aus dem Code folgen:**

1. *Unsichtbare Positive:* Das additive Residuum wird auf [0, 255] geclippt. Auf hellen oder gesättigten
   Hintergründen und bei kleinem Gain kann der erzeugte Kontrast unter dem Rauschen liegen, das Label
   bleibt trotzdem positiv. Das Modell lernt dann Halluzinationen, also mehr FP.
2. *Additiv statt multiplikativ:* Verfärbungen, Reflexions- und Albedo-Defekte skalieren mit der lokalen
   Helligkeit. Ein additiver Transfer von dunklem auf hellen Hintergrund ist dafür physikalisch falsch.
3. *Kein Kontext:* Kantenausbrüche, Defekte an Bohrungen oder an Materialübergängen landen an
   unmöglichen Orten. Das schafft das Risiko von Shortcuts und verfehlt die echte Verteilung.
4. *Große und Mehrklassen-Defekte* fehlen bzw. werden nicht klassenbewusst behandelt.

---

## 2. Nicht-generative Synthese

### 2.1 CutPaste / Crop&Paste / Copy-Paste

* **Idee.** CutPaste (Li et al., CVPR 2021, unverified) schneidet Patches aus einem *normalen* Bild und
  fügt sie hart an anderer Stelle wieder ein. Crop&Paste im Few-Shot-Generierungsprotokoll nimmt dafür
  *echte* Defektregionen und fügt sie auf Normalbilder ein. Copy-Paste für Instance Segmentation
  (Ghiasi et al., CVPR 2021, unverified) skaliert das mit Large-Scale-Jittering.
* **Ergebnisse.** NSA berichtet 97,2 Image-AUROC auf MVTec AD, 2,0 Punkte über CutPaste (3-way) [P].
  Im generativen Protokoll (U-Net nur auf synthetischen Paaren trainiert) erreicht Crop&Paste auf der
  Kategorie *bottle* Pixel-AUROC/AP/F1 von 94,5/67,4/63,5, DFMGAN dagegen 98,9/90,2/83,9 (Tabelle in
  AnomalyHybrid) [P]. Das ist eine einzelne Kategorie; ich verallgemeinere es nicht.
* **Anwendbarkeit.** Unser Residual-Copy-Paste ist eine verfeinerte Crop&Paste-Variante: Die Beleuchtung
  des Zielbilds bleibt erhalten, abgeschwächte Varianten kommen hinzu, und es gibt viele echte
  Hintergründe. *Offen ist, ob diese Verfeinerung die in der Literatur berichtete Lücke zu generativen
  Verfahren schließt.* Das lässt sich günstig prüfen (Empfehlung E1).
* **Kosten.** Millisekunden pro Sample auf der CPU.

### 2.2 DRAEM (Perlin + Textur) und NSA (Poisson)

* **DRAEM** (Zavrtanik et al., ICCV 2021). Binarisiertes Perlin-Rauschen bildet die Maske, gefüllt mit
  Texturen aus DTD. Ein rekonstruktives und ein diskriminatives Teilnetz lernen gemeinsam.
  MVTec AD: Image-AUROC 98,1, Pixel-AUROC 97,5, Pixel-AP 68,9 (vortrainierte Modelle) [R]. Die Maske ist
  per Konstruktion exakt. Die Anomalien sind absichtlich nicht realistisch, sie lehren „alles
  Ungewöhnliche“. Für unbekannte Defekttypen ist das gut. Auf reflektierenden oder texturierten
  Oberflächen ist es riskant, weil das Modell lernt, auf jede Abweichung von der Textur anzuspringen.
* **NSA** (Schlüter et al., ECCV 2022, arXiv 2109.15222). Patches anderer Normalbilder werden in
  verschiedenen Skalierungen mit **Poisson Image Editing** eingefügt (Pérez et al., SIGGRAPH 2003,
  unverified). Die Begründung: Poisson-Beispiele haben keine künstlichen Diskontinuitäten und
  generalisieren daher besser auf echte Defekte [P]. Image-AUROC 97,2 [P]. Die README nennt einen
  wichtigen Punkt: Für MVTec-Objektklassen werden „source gradients“ statt „mixed gradients“ genutzt,
  „due to artefacts of the cloning method“ [R]. Eine spätere Arbeit merkt an, Poisson-Blending führe oft
  Interpolationsartefakte ein (arXiv 2512.23227) [P].
* **FLASH** (arXiv 2609.37314, nur Snippet, Inhalt unverifiziert). Kombiniert Feathered-Alpha für kleine
  und Poisson-Blending für große Defekte, mit Rückfall auf Alpha, wenn Poisson scheitert. Das ist genau
  unsere Größenproblematik.
* **MemSeg** (unverified) ist in ASBench als Syntheseverfahren enthalten [R]; Zahlen habe ich nicht geprüft.
* **Anwendbarkeit.** Gradient-Domain-Transfer ist der naheliegende Weg zu **großen Defekten**
  (Abschnitt 6). Es gilt dabei die Warnung, dass Blending-Artefakte selbst zum Shortcut werden können
  (Abschnitt 5).
* **Kosten.** Ein Poisson-Solve dauert auf der CPU im Bereich ms bis 100 ms pro Patch (eigene Schätzung).

### 2.3 Feature-Space-Synthese: SimpleNet, GLASS, SuperSimpleNet, DSR, PBAS

* **SimpleNet** (Liu et al., CVPR 2023): Feature-Encoder, Feature-Generator und Defekt-Diskriminator
  [R]. Anomalien entstehen im Feature-Raum; dass es sich um Gaußrauschen handelt, ist unverified.
* **GLASS** (Chen et al., ECCV 2024): globale Synthese im Feature-Raum per Gradient Ascent plus lokale
  Synthese im Bildraum. Ziel ist ausdrücklich die Abdeckung **schwacher Defekte, die normal aussehen**.
  I-/P-AUROC: MVTec AD 99,9/99,3, VisA 98,8/98,8, MPDD 99,6/99,4, WFDD 100/98,9 [R]. Getestet auf einer
  A800 mit 80 GB [R].
* **SuperSimpleNet** (Rolih et al., ICPR 2024, arXiv 2408.03143) und die Erweiterung „No Label Left
  Behind“ (Rolih, Fučka, Skočaj, J. Intell. Manuf. 2025, arXiv 2508.19060): Synthese im latenten Raum,
  dual ausgelegter Klassifikationskopf, alle Supervision-Regime von unsupervised bis voll supervised,
  Inferenz < 10 ms [P]. Leaderboard KSDD2: AP 97,4 gegenüber DSR 95,2 [S].
* **DSR** (Zavrtanik et al., „Dual subspace re-projection“, ECCV 2022, Venue unverified): Anomalien im
  quantisierten latenten Raum, auf „near-in-distribution“-Defekte ausgerichtet.
* **PBAS** (TCSVT 2024, nur Titel): progressive, grenzgeführte Synthese.
* **Anwendbarkeit.** (+) Die Maske ist exakt, und es gibt kein Problem mit Realismus im Pixelraum.
  (+) Billig und M4-tauglich. (−) Die Features haben Stride 4–8; 1–3 px breite Kratzer werden
  verwischt. (−) Die Literatur betrachtet überwiegend den unsupervised Fall. Für uns ist das eher ein
  **Zusatzregularisierer** (Hilfskopf auf Encoder-Features) oder ein **zweiter, andersartiger Detektor**
  für die Review-Queue. Als Ersatz für die Pixel-Synthese taugt es nicht.

### 2.4 RealNet / SDAS

**RealNet** (Zhang et al., CVPR 2024, arXiv 2403.05897): „Strength-controllable Diffusion Anomaly
Synthesis“. Ein Diffusionsmodell, das **nur auf Normalbildern** trainiert wird, erzeugt Anomalien mit
einstellbarer Stärke.

* I-/P-AUROC: MVTec AD 99,6/99,0, MPDD 96,3/98,2, BTAD 96,1/97,9, VisA 97,8/98,8 [R].
* Etwa 360 000 synthetische Bilder sind veröffentlicht [R].
* Kosten: Diffusionstraining auf 4×A40 in ≈ 48 h [R], auf dem M4 nicht realistisch.
* **Konzeptuell wichtig für uns:** Ein Generator, der nur auf U trainiert wird, kann keine Information
  aus L in die OOF-Vorhersagen tragen (siehe Empfehlung E6).

### 2.5 Was überträgt sich auf echte Defekte? Evidenzlage

* Eine Arbeit, die *systematisch* misst, welche synthetischen Anomaliearten auf *echte* Defektsegmentierung
  übertragen, habe ich nicht gefunden.
* Indizien gibt es: NSA (keine Diskontinuitäten → besser), GLASS und DSR (nahe an der Verteilung
  liegende, schwache Anomalien → besser bei subtilen Defekten).
* **ASBench** (arXiv 2510.07927, IEEE TAI 2026) untersucht vier Dimensionen: Generalisierung über
  Datensätze und Pipelines, Anteil synthetischer zu echten Daten, Korrelation von Bildmetriken
  (SSIM, PSNR, LPIPS, IS, FID, KID) mit AUROC/AUPR und hybride Strategien. Abgedeckt sind CutOut,
  CutPaste, CutPaste-Scar, DeSTSeg, DFMGAN, DRAEM, FPI, Fractal, MemSeg, NSA und RealNet [R].
  Berichtet wird [S]: Oberhalb einer Schwelle hat der Synthetik-Anteil kaum noch Einfluss, und bei
  Anteilen nahe 1,0 droht Overfitting auf synthetische Muster, besonders bei DFMGAN und
  AnomalyDiffusion. Die Korrelationskoeffizienten konnte ich nicht abrufen.
* **Folgerung:** Wir messen den Nutzen nur auf echten Testdefekten und behandeln Bildqualitätsmetriken
  höchstens als Hinweis.

---

## 3. Generative Few-Shot-Defektgenerierung (GAN / Diffusion)

### 3.1 GAN-Generation

* **Defect-GAN** (Zhang et al., WACV 2021, unverified) und **SDGAN** erzeugen Defektbilder mit binären
  Masken (laut Related Work in arXiv 2408.00372) [P].
* **DFMGAN** (Duan et al., AAAI 2023, arXiv 2303.02389): Zuerst wird StyleGAN2 auf Normalbildern
  trainiert. Danach lernen „defect-aware residual blocks“ aus wenigen Defektbildern *eines Typs*, Masken
  zu erzeugen und Features in der Maske zu verändern [R/P].
* **Anwendbarkeit.** Ein Modell pro Typ, StyleGAN-Training (README: 2 GPUs), Probleme mit dem
  Speicher [R]. Für viele Aufgaben auf dem M4 unpraktisch. Die Maske stammt aus dem Netz, eine
  Garantie für die Ausrichtung gibt es nicht.

### 3.2 Diffusion mit Masken-Konditionierung

* **AnomalyDiffusion** (Hu et al., AAAI 2024, arXiv 2312.05767).
  *Idee:* LDM mit einer „Spatial Anomaly Embedding“ aus lernbarem Anomalie-Embedding und einem aus der
  Maske kodierten Ortsembedding, dazu „Adaptive Attention Re-weighting“ (AAR) für die Ausrichtung von
  Anomalie und Maske.
  *Protokoll:* MVTec, pro Anomalietyp ist das Drittel mit den kleinsten IDs Training, der Rest Test;
  1000 generierte Paare pro Typ; ein U-Net wird **nur auf generierten Daten** trainiert [P].
  *Ergebnis:* Pixel-AUROC 99,1, AP 81,4. Ohne AAR fällt das auf 95,0/64,9 [P]. **Die Ausrichtung allein
  macht also rund 16 AP-Punkte aus.**
  *README:* Nach Neugenerierung der hazelnut-Daten stieg die AP von unter 85 auf ≈ 96, und schlecht
  generierte Klassen wurden gefiltert [R]. Das zeigt große Varianz und Einfluss der Kuratierung.
* **AnoGen** (Gui et al., ECCV 2024, arXiv 2505.09263). Gelerntes Anomalie-Embedding plus
  **Bounding-Box-Führung** mit schwach überwachtem Loss. Segmentierungs-AU-PR: DRAEM +5,8, DeSTSeg +1,5 [P].
  Weil nur Boxen vorliegen, eignet es sich nicht als Pixel-Label. Übertragbar ist die Idee, unsichere
  Bereiche *schwach* bzw. als ignore zu behandeln.
* **DefectFill** (Song et al., CVPR 2025, arXiv 2503.13985). SD-Inpainting wird mit Defekt-, Objekt- und
  Attention-Loss feingetunt. „Low-Fidelity Selection“ wählt von 8 Samples das mit der schlechtesten
  Rekonstruktion (LPIPS in der Maske), also das deutlichste [P].
  KID auf bottle: 30,99 gegenüber DFMGAN 70,90 und AnomalyDiffusion 131,52 [P].
  *Vorsicht:* MAGIC schließt DefectFill aus dem Hauptvergleich aus, weil es auf **Testset-Masken**
  angewiesen ist [P]. Das ist ein möglicher Leak oder zumindest eine Hürde für die Vergleichbarkeit.
* **MAGIC** (Choi et al., CVPR 2026 Findings, arXiv 2507.02314). SD-2-Inpainting mit DreamBooth, Prompt-
  Perturbation, räumlich adaptive Guidance und **kontextsensitive Masken-Ausrichtung**, die Masken so
  verschiebt, dass die Anomalie plausibel im Objekt liegt [P]. Protokoll wie AnomalyDiffusion; für DAGM
  5 Anomaliebilder pro Klasse [P]. Zahlen nicht abgerufen.
* **Segmentierungsorientierte Verfahren 2025:**
  * **SARD** (arXiv 2508.03143): Regions-beschränkte Diffusion; nur die Anomaliefläche wird entrauscht,
    der Hintergrund bleibt erhalten. Dazu ein maskensensitiver Diskriminator [P].
  * **FAST** (NeurIPS 2025, arXiv 2509.20295): MVTec mIoU 76,72, Acc 83,97, mit nur 10 statt 1000
    Schritten, über 100-mal schneller [P].
  * **STAGE** (arXiv 2509.06693): explizite Masken-Ausrichtung; SegFormer mIoU 75,45 [S].
  * Bei diesen Verfahren bleibt der **Hintergrund exakt erhalten**. Damit lassen sich die Masken aus der
    Differenz zwischen erzeugtem Bild und Hintergrund ableiten (Idee N2).
* **GroundingAnomaly** (arXiv 2604.08301, ECCV 2026 laut Awesome-Liste). SD 1.4 mit einem Spatial
  Conditioning Module (semantische Karten pro Pixel) über Gated Self-Attention bei eingefrorenem U-Net.
  MVTec Pixel-AUROC/AP 99,3/85,9, VisA 98,2/67,2. Instanz-mAP 49,40 (MVTec) bzw. 40,54 (VisA), besser als
  DFMGAN, DualAnoDiff und SeaS [P].
* **Springer 2026** (DOI s12626-026-00216-x): promptfreies Img2Img-Inpainting mit geometrischen
  Mustermasken und ControlNet, getestet mit 5–25 Bildern pro Klasse auf MVTec/VisA [P]. Zahlen nicht
  abgerufen.

### 3.3 Gemeinsame Bild-Masken-Generierung

* **DualAnoDiff** (Jin et al., CVPR 2025, arXiv 2408.13509).
  *Idee:* Zwei Zweige (ganzes Bild und Anomalieteil) mit je einer LoRA auf SD 1.5, LoRA-Rang 32,
  **ein Modell pro Anomalietyp**, ≈ 4,5 h auf einer V100 für 5000 Epochen [P].
  *Maske:* entsteht durch **U-2-Net-Segmentierung des erzeugten Vordergrunds** [R], ist also eine
  Vorhersage und keine Konstruktion.
  *Ergebnis:* 500 Paare pro Typ, Pixel-AUROC 99,1, AP 84,5 [P].
* **SeaS** (Dai et al., ICCV 2025, arXiv 2410.14987).
  *Idee:* Ein gemeinsames U-Net, ein „Unbalanced Abnormal Prompt“ (ein Normal-Token, mehrere
  Anomalie-Tokens) und ein Decoupled-Anomaly-Alignment-Loss, das neue Kombinationen von
  Anomalie-Attributen möglich macht. Masken werden aus U-Net- und hochaufgelösten VAE-Features
  verfeinert. SD 1.4, A100 mit 22 GB [R].
  *Supervised Segmentierung (mittlere IoU über mehrere Segmentierer):*
  MVTec AD 57,66 gegenüber AnomalyDiffusion 46,49 und DFMGAN 45,99; VisA 29,40 gegenüber 17,69;
  MVTec 3D 35,37 gegenüber 19,88; F1-max MVTec 69,86 gegenüber 65,40 [R].
  *Syntheseverfahren DRAEM/GLASS (Pixel-AP):* MVTec 70,99 → 76,69, VisA 31,37 → 48,56 [R].
  *Abstract:* +12,79 % IoU [P]. Mehrere Typen in einem Modell sind für Mehrklassen-Aufgaben attraktiv.
* **GAA** (Lu et al., arXiv 2507.09619): lokale Konzeptzerlegung, mehrstufiges Clustering,
  regionsgeführte Maskengenerierung und ARS-basiertes Filtern. *Eingeräumte Grenze:* globale
  Layout-Anomalien wie verschobene Bauteile [P].
* **Defect Spectrum / Defect-Gen** (Yang et al., ECCV 2024, arXiv 2310.17316): zweistufiger
  Diffusionsgenerator mit bis zu +9,85 mIoU [P]. Besonders relevant ist die **Neuannotation** von
  MVTec, VISION, DAGM und Cotton: fehlende Defekte wurden ergänzt, Konturen präzisiert. Das brachte
  +10,74 % Recall und −33,10 % FPR [P]. *Auch die Benchmark-Ground-Truth selbst ist also verrauscht.*
* **CHIMERA** (AAAI 2026, „controllable image-mask extraction“): nur der Titel; das Repository ist leer [R].

### 3.4 Training-free, Zero-Shot und Foundation-Modelle (2025–2026)

* **AnomalyAny** (Sun et al., CVPR 2025, arXiv 2406.01078): SD 1.5, ein Normalbild zur Testzeit,
  attention-geführte Optimierung, 100 Schritte auf einer A100 [P].
* **Training-frei:** TF-IDG (Xu et al., ICCV 2025, One-Shot) [P] und O2MAG/„One-to-More“
  (CVPR 2026, arXiv 2603.18093), das die Self-Attention *eines* Referenzdefekts nutzt [P].
* **Zero-Shot:** DeltaDeno (ECCV 2026, arXiv 2511.16920; ganz ohne echte Anomalien) [P];
  AnomalyPainter, Anomagic, AnoStyler, CADiff, QARAD (alle AAAI 2026), DeCo und UniScale
  („arbitrary-scale“, ECCV 2026, arXiv 2608.07864). Von diesen habe ich überwiegend nur Titel bzw.
  Abstracts gesehen.
* **UniDG/UDG** (arXiv 2604.08915): 300 000 Quadrupel aus Normalbild, Defektbild, Maske und Caption;
  MM-DiT-Attention mit Diptych-Eingabe; Generierung aus Referenzen und Bearbeitung per Text **ohne
  Fine-Tuning pro Kategorie** [P]. Diese Richtung löst das Skalierungsproblem über viele Aufgaben am
  ehesten. Verfügbarkeit der Gewichte und Lizenz sind unbekannt, und es braucht eine GPU.
* **MIRAGE** (Hu et al., CVPRW VAND 2026, arXiv 2603.13507): Gemini 2.5 Flash Image als Black-Box-API;
  VLM-Prompts; CLIP-Filter mit drei Bedingungen; Masken aus Grounding-DINO plus YOLOv26-Seg mit
  Pixel-AUC 0,9292 auf 1000 GT-Masken; Nutzerstudie mit 31 Personen und 1550 Paarvergleichen [P].
  Für vertrauliche Industriedaten ist ein externes API problematisch, zudem ändert sich das API mit
  der Zeit (Reproduzierbarkeit).
* **DefectGen** (EAAI 2025, ScienceDirect S0952197625021827): DiT mit Langtext-Steuerung und
  „DoKr“-Fine-Tuning [P].
* **APO** („Anomaly-Preference Image Generation“, ICML 2026, arXiv 2605.02439): Preference Optimization
  mit echten Anomalien als Positivbeispielen; IS +5,9 %, IC-LPIPS +7,9 % [P]. Das sind nur Bildmetriken.
* **AnomalyAgent** (arXiv 2604.07900): agentische Synthese mit Werkzeugen und RL.
* **Bewertung.** Training-freie und Zero-Shot-Verfahren sind pro Aufgabe billig. Sie hängen aber an den
  Priors von SD bzw. Text. Für Domänen, die das Modell nicht kennt (unsere Annahme), entstehen
  eher „generische“ Defekte. Für **Exploration** unbekannter Defekttypen kann das helfen; als Labelquelle
  ist es fragwürdig.

### 3.5 Kritische Bewertung: Verbessert das Training auf generierten Defekten die Segmentierung echter Defekte messbar?

1. **Das Protokoll ist eng.** MVTec, ein Drittel zu zwei Dritteln *pro Anomalietyp*, ein Generator
   (bzw. eine LoRA) pro Typ, ein U-Net trainiert *nur* auf 500–1000 generierten Paaren, Test auf den
   übrigen Bildern **desselben Typs**. Gemessen wird Interpolation innerhalb eines homogenen Typs. Unser
   Fall sieht anders aus: 5–30 Bilder über mehrere Typen, unbekannte Typen in U, viele Aufgaben.
2. **Die Baselines sind schwach.** Verglichen wird ein Generator mit dem anderen, dazu naives
   Crop&Paste. Eine **Real-only-Kontrolle** (dieselben echten Defekte, Standard-Augmentation) oder ein
   starkes Copy-Paste auf vielen Normalbildern habe ich in keinem der gesichteten Paper gefunden.
   Ausnahmen mit gemischtem Training sind „Background-Aware Defect Generation“ (arXiv 2411.16767; U-Net
   auf echten plus synthetischen Daten, Zahlen nicht abgerufen) [P] und SynSur.
3. **Die Metriken sättigen.** Die Pixel-AUROC liegt bei allen neueren Verfahren bei 99,1–99,3. Allein AP,
   IoU und F1 unterscheiden. Recall auf Defektebene, Toleranzmetriken und der Einfluss auf die
   Kalibrierung werden praktisch nie berichtet.
4. **Varianz und Kuratierung:** hazelnut-AP von unter 85 auf ≈ 96 allein durch Neugenerierung, manuelles
   oder automatisches Wegfiltern schlechter Klassen [R]. Meist wird nur ein Lauf berichtet.
5. **Leaks:** Testset-Masken (DefectFill laut MAGIC). Maskengeneratoren werden auf Masken desselben Typs
   trainiert.
6. **Bildmetriken als Ersatzgröße:** KID, IS und IC-LPIPS dienen oft als Hauptbeleg (DefectFill, APO).
   Ihre Korrelation mit dem Nutzen für die Segmentierung ist offen (ASBench untersucht das).
7. **Evidenz aus der Industrie, auch auf unseren Benchmarks:**
   * **SynSur** (Fraunhofer IGD, arXiv 2604.26633), BSData-Pitting und MSD-Kratzer, mit YOLOX, YOLOv26 und
     LW-DETR: Rein synthetisches Training erreicht Real-Daten nicht. Synthetik *zusätzlich* bringt in
     einigen Konfigurationen AP-Gewinne und verkleinert die Lücke bei knappen Real-Daten [P].
   * **ScratchSim** (arXiv 2607.27065, BlenderProc): Fine-Tuning von synthetisch vortrainierten Gewichten
     schlägt reines Real-Training; Mischen hilft bei knappen Real-Daten [P].
   * Das NPI-Paper zu Kratzern (arXiv 2604.22850) nutzt **mAP bei IoU 0,01**, weil die IoU bei dünnen
     Kratzern zu streng ist [P]. Das bestätigt unsere Toleranzmetriken.
8. **Die Kosten skalieren mit Aufgaben × Klassen × Folds.** Ein Beispiel: 5 Aufgaben × 3 Klassen ×
   4 Folds ergeben 60 Fine-Tunings, bei 4,5 V100-Stunden je Lauf (DualAnoDiff) ≈ 270 GPU-Stunden
   (eigene Hochrechnung).

**Fazit.** Plausibel ist ein Gewinn dort, wo Copy-Paste versagt: bei großen Defekten, Formänderung,
fehlendem Material und kontextabhängiger Erscheinung. Für kleine und dünne Defekte, wo Residual-Paste
schon echte Pixel nutzt, ist der Zusatznutzen eher klein. Beides muss **gegen unsere Baseline**
gemessen werden.

### 3.6 Mask-Bild-Alignment

| Masken-Paradigma | Beispiele | Fehlerbild | Für Labels? |
|---|---|---|---|
| Maske als Bedingung | AnomalyDiffusion, DefectFill, MAGIC, SARD, FAST | Generator füllt zu wenig oder überläuft, schwacher Rand | nur mit nachgelagerter Prüfung |
| Maske nachträglich segmentiert | DualAnoDiff (U-2-Net), SeaS (Feature-Fusion), MIRAGE (Grounding-DINO+YOLO, Pixel-AUC 0,929) | Fehler des Segmentierers gehen ins Label ein | nur mit Unsicherheit bzw. ignore |
| nur Box | AnoGen | keine Pixelinformation | nein (nur schwach) |
| per Konstruktion | DRAEM, NSA, Residual-Paste, Rendering | exakt, aber das Label kann *unsichtbare* Pixel enthalten (Clipping, Gain) | ja, mit Prüfung der Sichtbarkeit |

Eine etablierte Metrik für das Alignment fehlt. GAA filtert über die Überlappung von Saliency und Maske
(ARS, Details unklar), DualAnoDiff bewertet visuell [S]. **Für uns** soll das Label nicht die Absicht
abbilden (Konditionierungsmaske), sondern das **tatsächlich erzeugte Signal** (Idee N2).

---

## 4. Physikbasiertes und prozedurales Rendering, Domain Randomization

* **Fulir et al.** (CVPRW 2023): gepaartes reales und synthetisches Defektsegmentierungs-Dataset für ein
  metallisches Kupplungsteil, mit mehreren Ansichten. Zwei Techniken für Effizienz und Abdeckung dunkler
  Bildbereiche, drei Segmentierungsarchitekturen bestätigen den Nutzen synthetischer Daten [P].
* **SYNOSIS** (arXiv 2410.14844, Sensors 2025): vollständige Kette aus Aufnahme, Textur- und
  Defektmodellierung, Generierung, Vergleich mit Realdaten und Segmentierungstraining, für gefrästes und
  sandgestrahltes Aluminium; Dual-Dataset aus realen und synthetischen Bildern [P]. Zahlen nicht abgerufen.
* **Mao, Fulir et al. 2026** (TU Wien, „Rendering Synthetic Defects for Learning-Based Industrial
  Inspection“, Venue unverified): **Defektform, Material, Beleuchtung und Blickwinkel** beeinflussen die
  Segmentierung viel stärker als die Samples pro Pixel des Renderers [P]. *Realismus der Physik schlägt
  Rauschfreiheit.*
* **Wagenstetter et al.** (ETFA 2024, „Synthetically Generated Images for Industrial Anomaly Detection“);
  **„Fully-Synthetic Training for Visual Quality Inspection in Automotive Production“** (arXiv 2503.09354,
  CIRP 2025): beide nur bibliografisch geprüft.
* **Eyecandies** (Bonfiglioli et al., ACCV 2022, arXiv 2210.04570; Erstautor unverified): Defekte am
  **3D-Modell** wirken realistischer als 2D-Defekte, die auf Renderings aufgebracht werden [P].
* **ISP-AD** (Krassnig & Gruber, J. Intell. Manuf. 2026): siebgedruckte Muster mit 245 664
  synthetischen und 711 echten Defektpatches; gemischt supervised Training [R]. Bisher ist das das
  größte gefundene Industriedataset, das Synthetik und Real-Daten paart. Für die Frage „Synthetik → Real“
  ist es ein potenzieller externer Benchmark.
* **Ein Springer-Paper von 2026** (Titel unverified, DOI s00371-026-04569-w): geometriebasierte,
  physikinspirierte Synthese von Beulen, Dellen und Kratzern auf Metall mit Fokus auf Dateneffizienz [P].
* **Domain Randomization** (Tremblay et al., CVPRW 2018, unverified) ist das Grundprinzip von ScratchSim
  (Materialien, Kameramodi).
* **Anwendbarkeit.** Rendering ist die einzige saubere Quelle für **geometrieabhängige** Defekte (Dellen,
  Ausbrüche, Verformungen) unter realistischer BRDF auf reflektierendem Metall. Pro Aufgabe braucht es
  aber CAD, Materialkalibrierung und Beleuchtungsmodell, also Tage bis Wochen Expertenaufwand. Das
  widerspricht „wenig Aufwand pro Aufgabe“. **Empfehlung:** kein Standardmodul. Sinnvoll höchstens als
  Spezialpfad für Aufgaben mit CAD und wiederkehrender Geometrie, und dort vor allem als
  **Vortraining**, wie es ScratchSim nahelegt.

---

## 5. Qualitätskontrolle synthetischer Samples und Vermeidung von Shortcuts

**Filter in der Literatur:**

| Filter | Arbeit | Signal | Kritik für uns |
|---|---|---|---|
| Low-Fidelity Selection | DefectFill | 1 aus 8, schlechteste LPIPS-Rekonstruktion in der Maske | bevorzugt deutliche Defekte → **gegen** schwache Defekte (Recall!) |
| CLIP mit drei Bedingungen | MIRAGE | Prompt-Fit, Defekt- gegen Normal-Prompt, gegen Originalbild | CLIP kennt die Domäne nicht |
| DreamSim + CLIPScore + Memorierungstest | SynSur | Nähe zu echten Patches, Prompt-Treue, Abstand zum nächsten Trainingsnachbarn | Memorierungstest ist sinnvoll und übertragbar |
| ARS | GAA | Überlappung von Saliency und Maske | Prinzip passt (Alignment), Definition unklar |
| DINOv3-Änderungskarten | arXiv 2609.37360 | Bearbeitungen außerhalb des Zielbereichs verwerfen | übertragbar, um den Hintergrund zu prüfen |
| manuell bzw. Klassen streichen | AnomalyDiffusion-README | – | nicht reproduzierbar |

**Shortcut-Risiken.** Das Segmentierungsmodell kann lernen, *Spuren der Synthese* zu erkennen statt
Defekten: Nahtstellen und Diskontinuitäten (CutPaste), Klon-Artefakte (NSA-README), Unschärfe oder
falsches Rauschniveau aus Inpainting oder Resampling, VAE-Spuren der Diffusion im Maskenbereich (das
Latent-Inpainting verändert typischerweise Textur und Rauschen), Clipping-Kanten. ASBench meldet
Overfitting bei sehr hohem Synthetik-Anteil [S].

**Curriculum und Schwierigkeit.** GLASS (Gradient Ascent nahe an der Verteilung) und RealNet (Stärke
einstellbar) steuern die Schwierigkeit explizit. Bei uns übernimmt das bisher nur der feste Gain-Bereich
0,6–1,3.

**Vorgeschlagene QC-Batterie für unsere Pipeline** (jede Synthesequelle, ob Residual, Poisson oder
Diffusion):

1. *Hintergrund-Erhalt:* |I_syn − I_bg| außerhalb von dilate(M, 3 px) unter dem lokalen
   Rauschniveau, sonst verwerfen. Das zielt auf Diffusion, bei Residual-Paste ist es trivial erfüllt.
2. *Realisierte Sichtbarkeit:* Pixel in M mit |ΔI| < κ·σ_lokal (z. B. κ = 1) bekommen das Label
   *ignore*, nicht *positiv*. Damit verschwinden die unsichtbaren Positiven (Lücke 1). Für schwache
   Defekte setzt man κ bewusst niedrig, damit die harten Beispiele erhalten bleiben.
3. *Übereinstimmung mit dem Normalitätsmodell:* Der PatchCore-Score muss in M steigen und darf außerhalb
   nicht steigen. Bei sehr schwachen Defekten nur zur Gewichtung, nicht zum Verwerfen.
4. *Shortcut-Monitor (C2ST):* Ein kleines CNN unterscheidet Patches echter Defekte (L) von
   synthetischen. Eine AUC ≫ 0,5 heißt, die Synthetik ist unterscheidbar. Pro Sample kann man das als
   Dichteverhältnis w ≈ p_real/p_syn zur Gewichtung nutzen, wegen der wenigen echten Defekte aber nur
   stark regularisiert.
5. *Memorierung bzw. Duplikate:* Bei Diffusion den Abstand zur nächsten Quelle in L prüfen (wie SynSur).
6. **Verboten:** Filtern danach, ob das aktuelle Ensemble den Defekt findet. Das bevorzugt leichte Fälle
   und senkt den Recall genau dort, wo er am wichtigsten ist. Detektierbarkeit darf nur ins
   *Curriculum* oder die *Stratifizierung* eingehen.

---

## 6. Große, kontextabhängige und Mehrklassen-Defekte

**In der Literatur:**

* Kontextsensitive Masken-Ausrichtung (MAGIC) [P] und kontextsensitive Diffusion (CADiff, Titel).
* Komponentenbewusste logische Anomalien (arXiv 2502.11712, Titel).
* Beliebige Skalen (UniScale, Titel).
* Eingeräumte Grenze bei globalen Layout-Anomalien (GAA) [P].
* Größenabhängiges Blending (FLASH, Snippet).
* Mehrklassen: SeaS (mehrere Anomalie-Tokens in einem Modell), DualAnoDiff (ein Modell pro Typ),
  UniDG (kein Fine-Tuning pro Kategorie), Defect Spectrum (mehrere Typen pro Bild annotiert).

**Konkrete Bausteine für uns, nach Defektart:**

| Defektart | Problem mit additivem Residuum | Vorschlag |
|---|---|---|
| Verfärbung, Fleck, Albedo-Defekt (groß) | wirkt multiplikativ; der Tiefpassanteil des Residuums passt nicht zur Zielbeleuchtung | **multiplikatives Modell** I = B·(1 + r) mit r = I_def/B_inp − 1; Hochfrequenz und Tiefpass getrennt übertragen (Laplace-Pyramide), Tiefpass relativ zur lokalen Helligkeit des Ziels |
| Großer Kratzer- oder Rissverbund | Größe, Skalierungsartefakte | Gradient-Domain-Transfer (Poisson auf ∇r), Skalierung durch Resampling des Gradientenfelds |
| Fehlendes Material, Ausbruch, Loch | das Residuum hängt von Geometrie und Hintergrund dahinter ab | Platzierung nur an Orten mit passendem Kontext (Kanten bzw. Rand); Diffusions-Inpainting (MAGIC/DefectFill-Typ) als GPU-Option; Rendering, wenn CAD vorhanden ist |
| Delle, Verformung | Schattierung hängt von der Orientierung zum Licht ab | Rotation nur, wenn die Beleuchtung isotrop ist; sonst Rotation auf Winkel beschränken, die in L belegt sind |
| Kontextgebunden (an Bohrung, Kante, Naht) | eine gleichverteilte Platzierung erzeugt unmögliche Bilder | **kontextpassende Platzierung**: Feature-Deskriptor des inpainteten Quellhintergrunds aus dem Normalitätsmodell (WRN50 L2+L3) per kNN-Suche über die Positionen in U |

**Wahl des Modells pro Defekt (datengetrieben, ohne Zusatzlabels):** Für jeden Bankdefekt passen wir
additives (I = B + r) und multiplikatives Modell (I = B·(1 + r)) auf ein *zweites* Inpainting mit leicht
veränderter Maske an. Wir wählen das Modell, dessen Residuum unter dieser Variation des Hintergrunds
stabiler ist. Alternativ oder zusätzlich: Wir transferieren den Defekt auf einen anderen Ort *desselben*
L-Bilds mit anderer Helligkeit und messen per C2ST, welches Modell realistischer wirkt.

**Mehrklassen:** Pro Klasse eine Bank, Sampling klassenbalanciert. Gemischte Pastes mehrerer Klassen
pro Crop erzeugen Konfusions-Negative zwischen den Klassen. Labels im Mehrklassen-Kanal. Für seltene
Klassen bei der Konformitätsanalyse klassenweise stratifizieren und offenlegen, dass für Klassen mit
n < 9 keine Garantie möglich ist.

---

## 7. Synthetische Daten für Kalibrierung und Validierung

* **SWSA** (Fung et al., arXiv 2310.10461): Synthetische Anomalien (DiffStyle, CutPaste) aus einem
  kleinen Support-Set (≥ 10 Bilder) bilden ein Validierungsset für die **Modellwahl**. Die gewählten
  Modelle bzw. Prompts stimmen *oft* mit der Wahl auf echter Validierung überein [P]. In v1 war die
  Aussage noch stärker, in v3 ist sie vorsichtiger formuliert.
* **„Visual Anomaly Synthesis for Model Selection under Data Scarcity“** (arXiv 2609.37360):
  Instruktionsbasierte Bearbeitungen, mit DINOv3-Änderungskarten gefiltert, als Validierungsdaten für
  die Modellwahl [P].
* **AUCp** (arXiv 2606.08742): Pseudo-AUC auf ungelabelten Validierungsdaten (nur Titel bzw. Snippet).
* **SPI – Synthetic-Powered Predictive Inference** (Bashari, Lotan, Lee, Dobriban, Romano, NeurIPS 2025,
  arXiv 2505.13432): Ein „Score Transporter“ ordnet reale und synthetische Nonconformity-Scores per
  Quantil-Abbildung einander zu. Die Coverage gilt in endlichen Stichproben **ohne Annahme über die
  Beziehung** zwischen realer und synthetischer Verteilung. Effizienter, also schärfer, wird es nur bei
  guter Übereinstimmung der Verteilungen [P].
* **Klassisch:** Weighted Conformal unter Covariate Shift (Tibshirani et al., NeurIPS 2019, unverified);
  Conformal Prediction ohne Austauschbarkeit mit TV-abhängiger Coverage-Lücke (Barber et al., Ann.
  Statist. 2023, unverified).

**Risiken für uns:**

1. Unsere konforme Recall-Garantie setzt Austauschbarkeit der *echten* Defekte in L und U voraus.
   Synthetische Defekte sind nicht austauschbar mit echten. Mischt man sie naiv in die
   Kalibrierungsmenge, ist die Garantie falsch, typischerweise zu optimistisch, wenn die Synthetik
   leichter ist.
2. Ein synthetisch geschätzter Recall misst die Detektierbarkeit der *Synthese-Verteilung* (Gain,
   Größe, Ort), nicht die der echten Defekte.
3. Rangfolgen sind robuster als absolute Niveaus: SWSA zielt auf Modellwahl, nicht auf ein absolutes
   AUROC.

**Zulässige Verwendung:**

* (a) Relative Entscheidungen: Hyperparameter, Wahl der Synthesevariante, Vergleich von Detektoren.
* (b) Stratifizierte Detektierbarkeitskurven nach Größe × Kontrast × Kontext, die anschließend mit
  echten OOF-Defekten *rekalibriert* werden.
* (c) Bildspezifische Sensitivitätskarten per Injection-Recovery als Merkmal für die Selbsteinschätzung.
  Die Gültigkeit kommt dann aus dem PPI-Audit (Idee N1).
* (d) SPI als formal sauberer Weg, synthetische Scores in eine konforme Schwelle einzubeziehen. Prüfen,
  ob das bei n ≈ 10–30 echten Defekten die Schwelle sinnvoll schärft.

---

## 8. Vergleichstabelle

| Methode (Jahr) | Typ | Echte Defekte nötig | Masken-Genauigkeit | Groß/Kontext | Berichtete Evidenz (Protokoll) | Kosten | Eignung für uns |
|---|---|---|---|---|---|---|---|
| **Residual-Copy-Paste** (unsere Baseline) | Pixel, additiv | ja (L) | exakt, aber unsichtbare Positive möglich | nein (nur klein und mittel) | noch keine Zahlen auf U/T | CPU, ms | Basis, ausbauen |
| Crop&Paste / CutPaste | Pixel, hart | ja / nein | exakt | nein | bottle 94,5/67,4/63,5 (Crop&Paste, nur synth.) [P] | CPU | Vergleichspunkt |
| DRAEM (2021) | Perlin + DTD | nein | exakt | Form zufällig | MVTec 98,1/97,5/68,9 [R] | CPU / kleine GPU | Hilfsdiversität, FP-Risiko |
| NSA (2022) | Poisson | nein | exakt (Intensität) | ja (Skalen) | Image-AUROC 97,2 (+2,0 vs. CutPaste) [P] | CPU | **Blending für große Defekte** |
| SimpleNet / GLASS / SuperSimpleNet | Feature-Raum | nein / optional | exakt (grob, Stride) | – | GLASS 99,9/99,3 MVTec [R]; SSN KSDD2 AP 97,4 [S] | M4 ok | Regularisierer / 2. Detektor |
| RealNet SDAS (2024) | Diffusion nur auf Normalbildern | nein | Maske vorgegeben | mittel | 99,6/99,0 MVTec [R] | 4×A40, 48 h | Konzept (U-only, kein Leak) |
| DFMGAN (2023) | GAN pro Typ | ja | aus dem Netz | mittel | bottle 98,9/90,2/83,9 [P] | GPU | gering |
| AnomalyDiffusion (2024) | LDM + Masken-Konditionierung | ja (⅓ pro Typ) | AAR, ohne AAR −16 AP | ja | 99,1 / AP 81,4 [P] | GPU, pro Typ | Referenz |
| AnoGen (2024) | LDM + Box | ja | nur Box | ja | AU-PR DRAEM +5,8 [P] | GPU | nein (Label) |
| DefectFill (2025) | SD-Inpainting | ja | Bedingung; Testmasken-Problem | ja | KID bottle 30,99 [P] | GPU | Inpainting-Idee |
| MAGIC (2025/26) | SD2-Inpainting + DreamBooth | ja (z. B. 5/Klasse) | Bedingung + Kontext-Ausrichtung | **ja, Kontext** | Zahlen nicht abgerufen | GPU | **Kandidat für große/Kontext-Defekte** |
| FAST / SARD / STAGE (2025) | segmentierungsorientiert | ja | Hintergrund erhalten → Differenzmaske möglich | ja | FAST mIoU 76,72, 10 Schritte [P] | GPU (FAST schnell) | Kandidat |
| DualAnoDiff (2025) | 2× LoRA, gemeinsam | ja, pro Typ | U-2-Net im Nachgang | ja | 99,1 / AP 84,5 [P] | V100 4,5 h/Typ | teuer bei vielen Aufgaben |
| SeaS (2025) | gemeinsames U-Net, Mehr-Token | ja | Feature-Fusion | ja | IoU 57,66 vs. 46,49 [R] | A100, 22 GB | Mehrklassen-Kandidat |
| GroundingAnomaly (2026) | SD 1.4 + räumliche Konditionierung | ja | Bedingung | ja | 99,3 / AP 85,9 [P] | GPU | beobachten |
| AnomalyAny / TF-IDG / O2MAG | training-frei | 0–1 | Attention-basiert | ja | qualitativ / AD | GPU, pro Aufgabe billig | Exploration |
| UniDG (2026) | MM-DiT, Foundation | Referenz | Quadrupel-Training | ja | Tabellen-Mittel 77,36 (Metrik unklar) [P] | große GPU | langfristig für viele Aufgaben |
| MIRAGE (2026) | API (Gemini) | nein | Grounding-DINO+YOLO, AUC 0,929 [P] | ja | U-Net MVTec/VisA | API-Kosten | nein (Vertraulichkeit) |
| Rendering (Fulir, SYNOSIS, ScratchSim) | physikbasiert | nein | exakt | **ja, Geometrie** | synthetisch allein < real; Vortraining hilft [P] | CAD + Tage | Spezialfall |
| SynSur (2026) | LoRA-Diffusion + Filter | ja | Inpainting + abgeleitete Labels | ja | BSData/MSD: hilft bei Knappheit [P] | GPU | Protokollvorbild |

---

## 9. Empfehlungen für unsere Pipeline (nach Rang)

| Rang | Maßnahme | Erwarteter Gewinn | Aufwand | Risiko |
|---|---|---|---|---|
| **E1** | **Evaluations-Harness für Synthese** und **Anker-Experiment zur Literatur**: unsere Residual-Paste im AnomalyDiffusion-Protokoll (MVTec, ⅓ / ⅔, U-Net nur synthetisch, 1000 Paare pro Typ) laufen lassen und neben Crop&Paste, AnomalyDiffusion und SeaS einordnen | Klarheit, ob Diffusion für uns überhaupt etwas bringen kann; vermeidet einen Fehlinvest | 1–2 Tage | gering |
| **E2** | **Labels aus dem erzeugten Signal** (Sichtbarkeitsschwelle → ignore), **Null-Synthese-Negative**, **C2ST-Monitor** | weniger FP durch unsichtbare Positive und Artefakt-Shortcuts; messbare Synthesequalität | 1–2 Tage | gering |
| **E3** | **Residual-Transfer v2 für große Defekte**: Laplace-Band- bzw. Poisson-Transfer, additiv vs. multiplikativ, **kontextpassende Platzierung** über die Features des Normalitätsmodells; `paste_large` dann standardmäßig an | große Defekte (heute gar nicht synthetisiert) und kontextgebundene Defekte → Recall bei großen Defekten, weniger Shortcuts | 3–5 Tage | mittel (Realismus), abgefangen durch E2 |
| **E4** | **Detektierbarkeitsabgestimmte Gain- und Skalenverteilung** (Idee N3) statt fest 0,6–1,3 | Recall auf schwache Defekte | 1–2 Tage | gering bis mittel |
| **E5** | **Mehrklassen-Bänke** (klassenbalanciert, Konfusions-Negative) | Mehrklassen-Aufgaben (MSD, MT) | 1–2 Tage | gering |
| **E6** | **Optionales GPU-Modul „Diffusions-Inpainting“** (MAGIC-, DefectFill- oder FAST-Typ, SD-Inpainting + LoRA) **nur für große bzw. kontextabhängige Klassen**; cross-fitted (K LoRAs) *oder* nur auf U trainiert und dann nur als *Harmonisierer* (SDEdit mit wenig Rauschen im Maskenring), stets mit Differenzmasken (N2) und QC-Batterie | potenziell hoch bei komplexen Erscheinungen; laut Literatur unsicher | 1–2 Wochen + GPU | hoch (Alignment, Leak, Rechenzeit) |
| **E7** | **Feature-Space-Hilfsaufgabe** (GLASS- bzw. SuperSimpleNet-artig) als Regularisierer oder als zweiter Detektor für die Review-Queue | Robustheit gegen Textur-FP, diverserer zweiter Detektor | 2–4 Tage | gering bis mittel (dünne Defekte) |
| **E8** | **Bank aus U erweitern** (fold-konsistent: nur Defekte mit hoher Konfidenz von Teacher k fließen in Member k) | Diversität | 1 Tag | Bestätigungs-Bias (nur leichte Defekte) |
| **E9** | **Synthetik für Kalibrierung** nur nach Abschnitt 7 (Rangfolgen, rekalibrierte Kurven, Injection-Recovery, SPI-Prototyp) | engere bzw. informativere Qualitätsaussagen | 3–5 Tage | hoch, wenn falsch angewandt → nur mit PPI-Prüfung |
| **E10** | Rendering nur bei Aufgaben mit CAD und geometrieabhängigen Defekten, dann zum Vortraining | im Einzelfall hoch | Wochen pro Aufgabe | widerspricht der Aufgaben-Agnostik |

### 9.1 Den Nutzen der Synthetik ehrlich bewerten (verbindliches Protokoll)

1. **Cross-Fitting der Synthese:** Bank, LoRA und Diffusionsmodell für Member k sehen nur L∖Fold_k.
   Alternativ wird der Generator nur auf U trainiert. Das wird per Unit-Test erzwungen (Defekt-IDs der
   Bank ∩ Fold_k = ∅).
2. **Gepaarte Varianten bei gleicher Iterationszahl und gleichem L:** (i) nur echt (`p_synth = 0`),
   (ii) Baseline-Residual-Paste, (iii) Baseline plus Kandidat, (iv) nur Kandidat, ohne Residual-Paste.
3. **Bewertet wird nur auf echten Daten:** Testset T (gruppiert getrennt) und U-Ground-Truth. Nie auf
   synthetischen Daten.
4. **Metriken:**
   * Defekt-Recall bei festen FP/Bild;
   * Toleranz-F1;
   * Pixel-AP;
   * Boundary-F1;
   * **nach Größe stratifiziert** (Terzile oder klein < 64 px, mittel, groß > 6000 px) und **nach
     Kontrast stratifiziert** (Terzile des Normalitäts-Scores);
   * pro Klasse;
   * dazu Kalibrierung: OOF-ECE, empirische Recall-Coverage der konformen Schwelle auf T und die Breite
     der PPI-Intervalle.
5. **Varianz:** mindestens 5 L-Ziehungen × 2 Seeds pro Datensatz. Gepaarter Bootstrap über die
   Ziehungen; die Effektgröße vorher festlegen, z. B. ≥ +2 Prozentpunkte Defekt-Recall bei gleichen FP.
6. **Anteil synthetischer Daten:** `p_synth` ∈ {0, 0,2, 0,35, 0,5, 0,8} durchmessen (ASBench-Befund).
7. **Shortcut-Tests:** FP-Rate auf Normalbildern, die mit Null-Synthese bearbeitet wurden, gegenüber
   unbearbeiteten Normalbildern; C2ST-AUC berichten.
8. **Rechenzeit pro Aufgabe** berichten (M4 und GPU getrennt).
9. **Datensätze:**
   * große, flächige Defekte: MT (inkl. *uneven*), KSDD2, BSData (Pitting unterschiedlicher Größe);
   * dünne Defekte: DeepCrack, CrackForest;
   * Mehrklassen: MSD, MT;
   * Kontext bzw. Objekte: MVTec AD/AD 2, VisA.

---

## 10. Neue Forschungsideen über die Literatur hinaus

### N1 – Injection-Recovery-Karten für Vollständigkeit als Selbsteinschätzung und für stratifizierte Audits

**Idee.** In der Astronomie ist es gängige Praxis (hier nicht zitiert), die Vollständigkeit einer
Detektionspipeline zu schätzen, indem man künstliche Quellen einbringt und misst, wie viele davon
wiedergefunden werden. Wir übertragen das: In *jedes* U-Bild werden an zufälligen Orten viele schwache
synthetische Defekte eingefügt (Residuen aus der Bank des passenden Folds, Gain im unteren Bereich).
Gemessen wird pro Bild und Region, welcher Anteil gefunden wird. Das ergibt eine **lokale
Sensitivitätskarte**: In Bereichen mit Reflexen, starker Textur oder Sättigung findet das Modell
synthetische Defekte schlecht, und dort wird es *auch echte* übersehen. Die Karte wird ein neues Merkmal
für P(übersehener Defekt) und definiert **Strata für ein stratifiziertes PPI-Audit**: mehr Audit-Bilder
dort, wo die Sensitivität niedrig ist, gewichtet nach Horvitz-Thompson. Die Gültigkeit kommt weiter aus
dem Zufalls-Audit; die Synthetik senkt nur die Varianz und lenkt die Aufmerksamkeit. **[Korr. 07 F5/F7]**
Ein stratifiziertes PPI-Audit liefert nur approximative Intervalle. Gültig ist es erst mit einer
designbasierten Schranke und unter den Bedingungen aus 07 §7 Nr. 7 (u. a. Floor q ≥ γ/N′, eingefrorene
Strata und Schranken, Messung nach Protokoll P).

**Warum neu.** Die gefundene Literatur nutzt Synthetik für Training, Modellwahl (SWSA) oder
Conformal-Effizienz (SPI). Eine *bildspezifische* Recall-Abschätzung per Injection, die in eine PPI- bzw.
Stratifizierungslogik eingeht, habe ich nicht gefunden. **[Korr. 07 §4, C1]** Injection-Recovery selbst
ist bekannt (Kepler DR25, Balrog, Fehler-Seeding); neu ist höchstens die Übertragung auf Pixel-Auto-Labels
industrieller Defekte.

**Falsifizierbares Experiment.** MT, KSDD2 und BSData, L = 15, 10 L-Ziehungen.
(a) AUROC der Injection-Sensitivität, gemessen daran, ob ein Bild in U-GT einen übersehenen echten
Defekt enthält, gegenüber den bestehenden Signalen (Uneinigkeit, unerklärte Anomalie) und der
Kombination. (b) Breite des PPI-Intervalls für den Defekt-Recall bei gleichem Audit-Budget von 30
Bildern: zufällig gegenüber stratifiziert. **Hypothese verworfen**, wenn die AUROC-Steigerung < 0,03
ist oder die Intervallbreite um < 10 % sinkt oder die empirische Coverage unter nominal − 5 pp fällt.

### N2 – Labels aus dem tatsächlich erzeugten Signal (Differenzmasken) für jeden Generator

**Idee.** Das Label entsteht nicht aus der Konditionierungs- bzw. Quellmaske, sondern aus dem Signal,
das die Synthese tatsächlich erzeugt hat:

* Y⁺ = Zusammenhangskomponenten von {|I_syn − I_bg| > κ·σ_lok} ∩ dilate(M_cond) mit Kontakt zum Kern von M_cond;
* Y^ignore = dilate(M_cond) ∖ Y⁺ ∖ {klar unverändert};
* Sample verwerfen, wenn die Energie außerhalb von dilate(M_cond) größer als ε ist.

Für Residual-Paste ist I_bg das Canvas vor dem Einfügen. Für hintergrundtreue Diffusion (SARD, FAST,
RePaint-artig) ist es das Eingabebild. Das löst zugleich die unsichtbaren Positiven (Clipping, kleiner
Gain), das Unter- bzw. Überfüllen generierter Masken und die Unsicherheit am Rand, die über ignore
behandelt wird.

**Was daran neu ist (korrigiert nach Red-Team, [07](07_red_team.md) §4, C6).** Die erste Fassung sagte, eine
radiometrisch definierte Differenzmaske als Labelvertrag sei „nicht gesehen“. Das ist falsch.
Signalbasierte Labels sind bekannt:

* NSA (Schlüter, Tan, Hou, Kainz, ECCV 2022) setzt das Label als geschwellte Intensitätsdifferenz zwischen
  Ergebnis und Hintergrund, auch im Modus „binary“. Das hat 07 im Code (`self_sup_tasks.py`) geprüft.
* FPI/PII (Tan et al., arXiv:2011.04197; MICCAI 2021) leiten das Label aus dem Syntheseschritt selbst ab
  (Interpolationsfaktor bzw. erzeugte Veränderung), nicht aus einer Quellmaske. Laut 07 nächste Vorarbeit; Details
  nicht am Volltext geprüft.

Die übrigen gefundenen Verfahren nehmen die Bedingung, segmentieren nachträglich mit einem *semantischen*
Segmentierer (U-2-Net, Grounding-DINO) oder lernen die Maske mit. Unser Beitrag ist enger: die
**Verallgemeinerung** dieses Prinzips auf alle Generatoren (Residual-Paste, hintergrundtreue Diffusion), ergänzt um
das **Ignore-Band** und die **Verwerfungsregel** für Energie außerhalb von dilate(M_cond).

**Falsifizierbares Experiment.** Gleiche Synthese (Residual-Paste, später Diffusion), vier
Labelvarianten:

* Quellmaske;
* **NSA-Arm**: geschwellte Intensitätsdifferenz wie in NSA, ohne Ignore-Band und ohne Verwerfungsregel
  (nach 07 C6 ergänzt, damit der Beitrag gegen die bekannte Vorarbeit gemessen wird);
* Differenzmaske mit lokaler Normierung (κ·σ_lok) und Verwerfungsregel;
* Differenzmaske + ignore.

Bewertet werden Boundary-F1, Toleranz-F1, Pixel-AP und FP/Bild auf T für KSDD2, MT und DeepCrack, außerdem die
OOF-ECE. **Verworfen**, wenn Boundary-F1 um < 1 pp steigt und FP/Bild nicht sinkt, über 5 Ziehungen gepaart. Der
Vergleich, der über den eigenen Beitrag entscheidet, ist „Differenzmaske + ignore“ gegen den NSA-Arm. Der Vergleich
mit der Quellmaske zeigt nur, ob signalbasierte Labels überhaupt helfen.

### N3 – Detektierbarkeitsabgestimmte Synthese

**Idee.** Der Gain-Bereich 0,6–1,3 ist willkürlich. Stattdessen werden Gain und Skalierung so
gesteuert, dass die Verteilung des **Normalitäts-Scores** der synthetischen Defekte (Score in M minus
lokaler Hintergrund-Score) der Verteilung der *echten* L-Defekte entspricht. L ist eine Zufallsstichprobe
und damit für U repräsentativ. Dazu kommt eine gezielte Übergewichtung des schwachen Endes, etwa die
unteren 30 % der realen Score-Verteilung doppelt. Umsetzung: 1-D-Optimal-Transport oder
Quantil-Matching auf den Gain pro Bankdefekt. Begründung: Übersehen werden die schwachen Defekte, und
das Modell soll deren Schwierigkeitsverteilung sehen, ohne auf unrealistisch schwache Signale (die
Unsichtbaren aus N2) trainiert zu werden.

**Falsifizierbares Experiment.** Fester Gain, abgestimmter Gain und abgestimmter Gain mit Übergewicht
des schwachen Endes. Primärmetrik: Defekt-Recall im **schwächsten Kontrast-Terzil** echter
Testdefekte bei festen FP/Bild (MT, KSDD2, BSData, CrackForest). **Verworfen**, wenn der Gewinn
< 3 pp ist oder die FP/Bild um mehr als 10 % steigen.

### N4 – Kontextpassende Platzierung und Transfer in mehreren Bändern für große Defekte

**Idee.** Zu jedem Bankdefekt wird der Kontext-Deskriptor des inpainteten Quellhintergrunds gespeichert
(WRN50-Features des Normalitätsmodells, bereits vorhanden, plus Kantenorientierung). Einfügeorte in U
werden über eine kNN-Suche mit Temperatur gewählt, damit die Diversität erhalten bleibt. Die Rotation
wird auf Winkel mit ähnlicher Kontextorientierung beschränkt. Für große Defekte wird in mehreren Bändern
übertragen (Hochfrequenz additiv, Tiefpass multiplikativ und relativ), mit Modellwahl wie in
Abschnitt 6.

**Falsifizierbares Experiment.** Zufällige gegenüber kontextpassender Platzierung, alter gegenüber neuem
Transfer.
(a) C2ST-AUC (real gegen synthetisch) muss sinken.
(b) Defekt-Recall bzw. IoU in den Strata **groß** (MT *uneven*/*fray*, KSDD2 groß, BSData groß) und bei
kontextgebundenen MVTec-Objektklassen (z. B. screw, metal_nut), VisA.
**Verworfen**, wenn die C2ST-AUC nicht um ≥ 0,05 sinkt oder der Recall bei großen Defekten nicht um
≥ 3 pp steigt.

### N5 – Null-Synthese-Negative gegen Artefakt-Shortcuts

**Idee.** Für jede Syntheseoperation gibt es eine „Null“-Variante mit identischen Artefakten, aber ohne
Defekt: Residuum mit Gain 0, aber gleichem Resampling, gleicher Weichmaske und gleichem Clipping;
Poisson-Klon eines *normalen* Patches; Diffusions-Inpainting mit Normal-Prompt bzw. Normal-LoRA. Diese
Bilder werden als **negativ** gelabelt. Das Modell kann sich so nicht auf Spuren der Synthese stützen.

**Vorarbeit (nach Red-Team, [07](07_red_team.md) C6).** Negative mit denselben Einfügeartefakten sind aus der
Copy-Paste-Synthese für Detektion bekannt (Dwibedi et al., ICCV 2017; Dvornik et al., arXiv:1809.02492). Neu ist
höchstens die systematische Null-Variante für jede Syntheseoperation.

**Falsifizierbares Experiment.** Mit bzw. ohne Null-Negative (Anteil 10–20 % der synthetischen
Samples). Gemessen werden FP/Bild auf null-bearbeiteten Normalbildern (sollte → 0), FP/Bild auf
echten Normalbildern, Pixel-AP auf T. **Verworfen**, wenn die FP auf echten Normalbildern nicht sinken
oder die Pixel-AP um mehr als 1 pp fällt.

### N6 (ergänzend) – Wegerklärungstest für die Vollständigkeit von Pseudo-Labels

Die vorhergesagte Defektmaske eines U-Bilds wird mit einem Inpainting aus normalem Kontext entfernt
(Telea bzw. später ein Diffusionsmodell nur auf U), dann wird der Normalitäts-Score neu berechnet.
Bleibt Anomalie stehen, ist das Label unvollständig (FN-Pixel bzw. ein weiterer Defekt). Fällt der Score
schon durch das Entfernen eines *Teils* der Maske, deutet das auf FP-Teile. Experiment: AUROC für
FN-Pixel bzw. FN-Defekte auf U-GT gegenüber dem bestehenden Signal „unerklärte Anomalie“; verworfen bei
einem Gewinn < 0,03.

---

## 11. Referenzen

Markierung: (unverified) = in dieser Recherche nicht selbst gefunden bzw. nur aus dem Gedächtnis.

**Nicht-generativ / Feature-Raum**
- Zavrtanik, Kristan, Skočaj: *DRAEM – A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection*, ICCV 2021. https://github.com/VitjanZ/DRAEM
- Schlüter et al.: *Natural Synthetic Anomalies for Self-Supervised Anomaly Detection and Localization* (NSA), ECCV 2022, arXiv 2109.15222. https://github.com/hmsch/natural-synthetic-anomalies
- Li et al.: *CutPaste: Self-Supervised Learning for Anomaly Detection and Localization*, CVPR 2021 (unverified).
- Ghiasi et al.: *Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation*, CVPR 2021 (unverified).
- Pérez, Gangnet, Blake: *Poisson Image Editing*, SIGGRAPH 2003 (unverified).
- Liu et al.: *SimpleNet: A Simple Network for Image Anomaly Detection and Localization*, CVPR 2023. https://github.com/DonaldRR/SimpleNet
- Chen et al.: *A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial Anomaly Detection and Localization* (GLASS), ECCV 2024. https://github.com/cqylunlun/GLASS
- Zhang et al.: *RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection*, CVPR 2024, arXiv 2403.05897. https://github.com/cnulab/RealNet
- Zavrtanik et al.: *DSR – A Dual Subspace Re-Projection Network for Surface Anomaly Detection*, ECCV 2022 (Venue unverified). https://github.com/VitjanZ/DSR_anomaly_detection
- Rolih et al.: *SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection*, ICPR 2024, arXiv 2408.03143. https://github.com/blaz-r/SuperSimpleNet
- Rolih, Fučka, Skočaj: *No Label Left Behind: A Unified Surface Defect Detection Model for all Supervision Regimes*, J. Intell. Manuf. 2025, arXiv 2508.19060.
- Božič, Tabernik, Skočaj: *Mixed supervision for surface-defect detection: from weakly to fully supervised learning*, Computers in Industry 2021. https://github.com/vicoslab/mixed-segdec-net-comind2021
- *Progressive Boundary Guided Anomaly Synthesis for Industrial Anomaly Detection* (PBAS), TCSVT 2024. https://github.com/cqylunlun/PBAS (nur Titel)
- *FLASH: A "Generate Once, Synthesize Many" Framework for Synthetic Anomaly Generation in Industrial Anomaly Detection*, arXiv 2609.37314 (nur Snippet)
- *Anomaly Detection by Effectively Leveraging Synthetic Images*, arXiv 2512.23227 (nur Snippet)

**Generativ (GAN / Diffusion)**
- Duan, Hong, Niu, Zhang: *Few-Shot Defect Image Generation via Defect-Aware Feature Manipulation* (DFMGAN), AAAI 2023, arXiv 2303.02389. https://github.com/Ldhlwh/DFMGAN
- Zhang et al.: *Defect-GAN: High-Fidelity Defect Synthesis for Automated Defect Inspection*, WACV 2021 (unverified).
- *Few-shot Defect Image Generation based on Consistency Modeling* (DefectDiffu), ECCV 2024, arXiv 2408.00372.
- Hu et al.: *AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model*, AAAI 2024, arXiv 2312.05767. https://github.com/sjtuplayer/anomalydiffusion
- Gui et al.: *Few-Shot Anomaly-Driven Generation for Anomaly Classification and Segmentation* (AnoGen), ECCV 2024, arXiv 2505.09263. https://github.com/gaobb/AnoGen
- Jin et al.: *Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation* (DualAnoDiff), CVPR 2025, arXiv 2408.13509. https://github.com/yinyjin/DualAnoDiff
- Dai et al.: *SeaS: Few-shot Industrial Anomaly Image Generation with Separation and Sharing Fine-tuning*, ICCV 2025, arXiv 2410.14987. https://github.com/HUST-SLOW/SeaS
- Song et al.: *DefectFill: Realistic Defect Generation with Inpainting Diffusion Model for Visual Inspection*, CVPR 2025, arXiv 2503.13985.
- Sun, Cao, Dong, Fink: *Unseen Visual Anomaly Generation* (AnomalyAny), CVPR 2025, arXiv 2406.01078.
- Yang et al.: *Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics*, ECCV 2024, arXiv 2310.17316. https://github.com/EnVision-Research/Defect_Spectrum
- Choi, Kim, Hong: *MAGIC: Few-Shot Mask-Guided Anomaly Inpainting with Prompt Perturbation, Spatially Adaptive Guidance, and Context Awareness*, CVPR 2026 Findings, arXiv 2507.02314.
- Lu et al.: *Generate Aligned Anomaly: Region-Guided Few-Shot Anomaly Image-Mask Pair Synthesis for Industrial Inspection* (GAA), arXiv 2507.09619.
- *SARD: Segmentation-Aware Anomaly Synthesis via Region-Constrained Diffusion with Discriminative Mask Guidance*, arXiv 2508.03143 (Autoren nicht erfasst).
- *FAST: Foreground-aware Diffusion with Accelerated Sampling Trajectory for Segmentation-oriented Anomaly Synthesis*, NeurIPS 2025, arXiv 2509.20295.
- *STAGE: Segmentation-oriented Industrial Anomaly Synthesis via Graded Diffusion with Explicit Mask Alignment*, arXiv 2509.06693.
- *GroundingAnomaly: Spatially-Grounded Diffusion for Few-Shot Anomaly Synthesis*, arXiv 2604.08301 (ECCV 2026 laut Awesome-Liste).
- *Anomaly-Preference Image Generation* (APO), ICML 2026, arXiv 2605.02439.
- *Large-Scale Universal Defect Generation: Foundation Models and Datasets* (UDG/UniDG), arXiv 2604.08915.
- Hu et al.: *MIRAGE: Model-agnostic Industrial Realistic Anomaly Generation and Evaluation for Visual Anomaly Detection*, CVPRW (VAND) 2026, arXiv 2603.13507.
- *AnomalyAgent: Agentic Industrial Anomaly Synthesis via Tool-Augmented Reinforcement Learning*, arXiv 2604.07900.
- *AnomalyHybrid: A Domain-agnostic Generative Framework for General Anomaly Detection*, CVPRW (SyntaGen) 2025, arXiv 2504.04340.
- *Background-Aware Defect Generation for Robust Industrial Anomaly Detection*, arXiv 2411.16767.
- Xu et al.: *Training-Free Industrial Defect Generation with Diffusion Models* (TF-IDG), ICCV 2025.
- *One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control* (O2MAG), CVPR 2026, arXiv 2603.18093.
- *DeltaDeno: Zero-Shot Anomaly Generation via Delta-Denoising Attribution*, ECCV 2026, arXiv 2511.16920.
- *AnomalyPainter*, AAAI 2026, arXiv 2503.07253; *Anomagic*, AAAI 2026, arXiv 2511.10020; *AnoStyler*, AAAI 2026; *CHIMERA*, AAAI 2026; *CADiff*, AAAI 2026; *UniScale: Arbitrary-Scale Anomaly Generation*, ECCV 2026, arXiv 2608.07864; *AnomalyControl*, arXiv 2412.06510 / ACM MM 2025; *Component-aware Unsupervised Logical Anomaly Generation*, arXiv 2502.11712 (alle nur Titel bzw. Abstract gesehen).
- *Enhancing anomaly detection with few-shot fine-tuned long text-to-image models* (DefectGen-DiT), Eng. Appl. Artif. Intell. 2025, ScienceDirect S0952197625021827.
- *Few-Shot Industrial Defect Detection via Diffusion-Based Data Augmentation with Geometric Pattern Masks*, Springer 2026, doi:10.1007/s12626-026-00216-x.
- Simoni, Pelosin: *Bounding Box-Guided Diffusion for Synthesizing Industrial Images and Segmentation Maps*, CVPRW 2025, arXiv 2505.03623. https://github.com/covisionlab/diffusion_labeling
- *ASBench: Image Anomalies Synthesis Benchmark for Anomaly Detection*, IEEE TAI 2026, arXiv 2510.07927. https://github.com/M-3LAB/ASBench
- *A Survey on Industrial Anomalies Synthesis*, arXiv 2502.16412.

**Rendering / Industrie-Pipelines**
- Fulir et al.: *Synthetic Data for Defect Segmentation on Complex Metal Surfaces*, CVPRW 2023.
- *SYNOSIS: Image Synthesis Pipeline for Machine Vision in Metal Surface Inspection*, Sensors 2025, arXiv 2410.14844.
- Mao, Fulir et al.: *Rendering Synthetic Defects for Learning-Based Industrial Inspection*, 2026 (TU Wien, Venue unverified).
- Wagenstetter et al.: *Synthetically Generated Images for Industrial Anomaly Detection*, ETFA 2024.
- *Fully-Synthetic Training for Visual Quality Inspection in Automotive Production*, CIRP 2025, arXiv 2503.09354.
- Bonfiglioli et al. (Erstautor unverified): *The Eyecandies Dataset for Unsupervised Multimodal Anomaly Detection and Localization*, ACCV 2022, arXiv 2210.04570.
- Krassnig, Gruber: *ISP-AD: A large-scale real-world dataset for advancing industrial anomaly detection with synthetic and real defects*, J. Intell. Manuf. 2026. https://github.com/p4ulk/isp-ad
- *SynSur: An end-to-end generative pipeline for synthetic industrial surface defect generation and detection*, arXiv 2604.26633.
- *ScratchSim: A Procedural Synthetic Data Pipeline for Surface Scratch Detection*, arXiv 2607.27065.
- *Accelerating New Product Introduction for Visual Quality Inspection via Few-Shot Diffusion-Based Defect Synthesis*, arXiv 2604.22850.
- Tremblay et al.: *Training Deep Networks with Synthetic Data: Bridging the Reality Gap by Domain Randomization*, CVPRW 2018 (unverified).

**Synthetik für Validierung / Kalibrierung**
- Fung, Qiu, Li, Rudolph: *Model Selection of Anomaly Detectors in the Absence of Labeled Validation Data* (SWSA), arXiv 2310.10461.
- *Visual Anomaly Synthesis for Model Selection under Data Scarcity*, arXiv 2609.37360 (nur Snippet).
- *AUCp: Pseudo-AUC for Inference Model Selection with Unlabeled Validation Data in Abnormality Detection*, arXiv 2606.08742 (nur Titel).
- Bashari, Lotan, Lee, Dobriban, Romano: *Synthetic-Powered Predictive Inference*, NeurIPS 2025, arXiv 2505.13432.
- Tibshirani, Barber, Candès, Ramdas: *Conformal Prediction Under Covariate Shift*, NeurIPS 2019 (unverified).
- Barber, Candès, Ramdas, Tibshirani: *Conformal Prediction Beyond Exchangeability*, Ann. Statist. 2023 (unverified).
- Angelopoulos et al.: *Prediction-Powered Inference*, Science 2023 (unverified in dieser Recherche; bereits in METHOD.md).

**Übersicht**
- M-3LAB: *awesome-industrial-anomaly-detection* (Abschnitt „Anomaly Synthesis“, Stand Oktober 2026). https://github.com/M-3LAB/awesome-industrial-anomaly-detection
