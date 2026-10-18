# Methode: Auto-Labeling feiner Defekte mit messbarer Labelqualität

Dieses Dokument beschreibt *warum* die Pipeline so aufgebaut ist, wie sie ist. Ergebnisse und Zahlen
stehen in [`RESULTS.md`](RESULTS.md).

## 1. Problemanalyse → Designprinzipien

| Eigenschaft der Aufgabe | Konsequenz für das Design |
|---|---|
| Sehr wenige Masken, viele ungelabelte Bilder | Ungelabelte Bilder müssen *aktiv* genutzt werden: als Normalitätsmodell, als Hintergründe für Defektsynthese, für Self-Training |
| Defekte klein, dünn (1–3 px), kontrastarm | Vorhersage in voller Auflösung (Stride 1), Hysterese-Schwellen, Metriken mit Randtoleranz |
| Starkes Klassenungleichgewicht | defektzentriertes Cropping, Dice+BCE, Schwellen auf Validierungsdaten statt 0.5 |
| Reflexionen, Texturen, Staub als Fehlalarm-Quellen | Defekte auf *echte* ungelabelte Oberflächen synthetisieren → das Modell sieht tausende Varianten normaler Oberfläche |
| Defektgrenze subjektiv | Toleranzmetriken, Ignore-Ringe um Pseudo-Labels |
| Übersehene Defekte besonders kritisch | Schwellenwahl mit **verteilungsfreier Recall-Garantie** (Conformal Prediction), unabhängiger zweiter Detektor (Normalitätsmodell) für die Review |
| Qualität muss messbar sein, kaum Ground Truth | (1) ehrliche Kreuzvalidierung auf L, (2) Selbsteinschätzung jedes Labels, (3) **PPI-Audit**: kleine zufällige Stichprobe + Selbsteinschätzung → gültige Konfidenzintervalle |
| Generische Vision-Modelle kennen die Domäne nicht | ImageNet-Features nur als Initialisierung bzw. als *Distanzmaß* (Normalitätsmodell), Entscheidungen werden aus L und U gelernt |

## 2. Überblick

```
          ┌──────────────── ungelabelte Bilder U ─────────────────┐
          │                                                        │
          ▼                                                        ▼
  (A) Normalitätsmodell                                  (B) Defektbank aus L
  PatchCore-artig, eigene Bilder                           (Residuen: Defekt − inpainteter
  ausgeschlossen, k-NN (k=3)                                Hintergrund)
          │  saubere Hintergründe, Anomaliekarten                │
          ▼                                                        ▼
  (C) Kreuzvalidiertes Ensemble (K=4 Folds über L)  ◄── Residual-Copy-Paste auf U-Hintergründe
      U-Net (ImageNet-Encoder) mit Full-Res-Pfad
          │ Out-of-fold-Vorhersagen auf L           │ Ensemble-Vorhersagen auf U
          ▼                                          ▼
  (D) Kalibrierung, Schwellen (Hysterese),     (E) Fold-konsistentes Self-Training:
      konforme Recall-Schwelle                     Member k lernt nur von Teacher k
          │                                          │  (Pseudo-Labels + Ignore-Regionen)
          └───────────────► Runde 2 ◄────────────────┘
                              │
                              ▼
  (F) Pseudo-Labels + "recall-sichere" Labels + Ignore-Masken
  (G) Selbsteinschätzung je Bild (erwarteter Dice, P(übersehener Defekt), Ensemble-Uneinigkeit,
      unerklärte Anomalie) → Review-Queue
  (H) Zufalls-Audit (n≈20–40 Bilder) + PPI → Konfidenzintervalle für Recall/Precision/Dice
```

## 3. Komponenten

### 3.1 Kreuzvalidiertes Ensemble (Cross-Fitting)
L wird in K Folds geteilt; Member k trainiert ohne Fold k. Dadurch bekommt **jedes gelabelte Bild eine
ehrliche Out-of-fold-Vorhersage**, obwohl es kein separates Validierungsset gibt. Diese OOF-Vorhersagen
sind die Grundlage für Kalibrierung, Schwellenwahl, Recall-Garantie und das Training der
Qualitätsschätzer. Die K Member bilden gleichzeitig das Ensemble für U (Mittelwert der
Wahrscheinlichkeiten + horizontales Flip-TTA); ihre Uneinigkeit ist ein Unsicherheitssignal.

### 3.2 Netzarchitektur
U-Net mit ImageNet-vortrainiertem Encoder (Standard `resnet18d`, Gewichte von den timm-GitHub-Releases)
und einem zusätzlichen **Full-Resolution-Stem**, dessen Features in der letzten Decoder-Stufe konkateniert
werden. Logits entstehen in voller Auflösung – 1–2 px breite Kratzer werden nicht wegskaliert.
Training auf 192×192-Crops, 60 % der Crops defektzentriert, BCE+Dice-Loss, AdamW mit Cosine-Schedule.

### 3.3 Ungelabelte Daten (A, B, E)
**Normalitätsmodell (A).** Defekte sind selten, also beschreibt U die *normale* Oberfläche gut –
inklusive Reflexionen, Texturen und Bearbeitungsspuren. Patch-Features (WideResNet50, Layer 2+3,
Zufallsprojektion auf 256 Dim.) von 64 zufälligen Patches pro Bild bilden eine Memory-Bank. Anomaliescore
= Distanz zum **k-ten** nächsten Bank-Patch **aus einem anderen Bild**. Das macht das Modell robust gegen
die (unbekannten) Defekte in U: ein Defekt gilt nur dann als normal, wenn es ≥k ähnliche Patches in
anderen Bildern gibt. Verwendung: (i) die normalsten 70 % von U dienen als Hintergründe für die
Synthese, (ii) „unerklärte Anomalie“ (hoher Score außerhalb des Labels) ist ein unabhängiges Signal für
übersehene Defekte.

**Residual-Copy-Paste (B).** Jeder gelabelte Defekt wird als *Abweichung vom lokalen Hintergrund*
gespeichert (Pixel − Telea-Inpainting der Defektregion) und additiv mit zufälligem Gain (0.6–1.3),
Rotation, Spiegelung und Skalierung auf Crops ungelabelter Bilder übertragen. Anders als klassisches
Copy-Paste bleibt die Beleuchtung/Reflexion des Zielbildes erhalten, und abgeschwächte Varianten
(Gain < 1) erzeugen gezielt schwer sichtbare Beispiele. So lernt das Modell aus 10 Defekten +
~1000 Hintergründen, *was kein Defekt ist*. (Großflächige Defekte werden nicht synthetisiert, weil
Alpha-Blending dort unrealistisch ist.)

**Fold-konsistentes Self-Training (E).** In Runde 2 lernt Member k zusätzlich von Pseudo-Labels auf U –
aber **nur von Teacher k** (Member k aus Runde 1), der Fold k ebenfalls nie gesehen hat. Dadurch bleiben
die OOF-Vorhersagen ehrlich (kein Informationsleck über den Teacher) und die Member bleiben divers
(sonst kollabiert das Ensemble auf einen gemeinsamen Teacher und die Uneinigkeit verliert ihren Wert).
Pseudo-Labels: Positiv = Hysterese-Maske; Ignore = unsichere Pixel (p ≥ 0.5·t_ext) und ein 2-px-Ring
um jede Maske; Rest negativ. Auch die Synthese-Hintergründe tragen in Runde 2 diese Pseudo-Labels,
damit echte, ungelabelte Defekte im Hintergrund nicht als „normal“ gelernt werden.

### 3.4 Von Wahrscheinlichkeiten zu Labels
**Hysterese-Schwellen**: eine Region `p ≥ t_ext` wird übernommen, wenn sie mindestens ein Pixel
`p ≥ t_seed` enthält (und ≥ min_area). Schwache Teile eines Kratzers, die mit einem deutlichen Teil
verbunden sind, bleiben erhalten; isolierte schwache Antworten (Textur) fallen weg. `(t_seed, t_ext,
min_area)` werden per Grid-Search auf den OOF-Vorhersagen gewählt (Ziel: Pixel-F1 mit 2-px-Toleranz).

**Kalibrierung**: Histogramm-basierte isotone Regression (Logit-Bins) auf allen OOF-Pixeln.

### 3.5 Recall-Garantie (Conformal Prediction)
Für jeden gelabelten Defekt ist `s = max p` in seiner (toleranz-erweiterten) Fläche bekannt (OOF). Ein
Defekt wird gefunden ⇔ `s ≥ t_seed`. Mit n gelabelten Defekten und `t = s_(⌊α(n+1)⌋)` gilt unter
Austauschbarkeit von L und U: **P(neuer Defekt wird gefunden) ≥ 1−α** (Split-Conformal). Zusätzlich
wird eine PAC-Schwelle (Clopper-Pearson) angeboten: mit Wahrscheinlichkeit ≥ 1−δ ist die Missrate auf
U ≤ α. Ist n zu klein (z. B. n<9 bei α=0.1), ist die einzige zertifizierbare Schwelle 0 – das Tool sagt
das ehrlich, statt eine Garantie vorzutäuschen. Die Ausgabe `labels_recall_safe/` nutzt
`min(t_seed, t_conformal)`.

### 3.6 Selbsteinschätzung pro Bild
Aus kalibrierten Wahrscheinlichkeiten q und Label ŷ: erwartete TP/FP/FN (`Σqŷ`, `Σ(1−q)ŷ`, `Σq(1−ŷ)`),
erwarteter Dice. Auf Defektebene: Kandidatenregionen (`p ≥ t_low`) erhalten über eine auf OOF gefittete
isotone Abbildung `P(echter Defekt | max p)`; daraus erwartete Treffer, erwartete übersehene Defekte,
`P(Bild enthält übersehenen Defekt)`, erwartete Fehlalarme. Dazu Ensemble-Uneinigkeit (paarweiser Dice
der Member) und unerklärte Anomalie. Die Review-Priorität ist der Rang-Mittelwert dieser Signale.

### 3.7 Messbarkeit: PPI-Audit
Die Selbsteinschätzung allein ist nicht beweiskräftig (sie kann systematisch falsch sein). Deshalb:
eine **zufällige** Stichprobe von n Bildern wird manuell geprüft, und die Qualität des gesamten Pools
wird mit **Prediction-Powered Inference** (Angelopoulos et al., Science 2023) in der Form des
Differenzschätzers für endliche Populationen geschätzt:

    Σ_pool y  ≈  Σ_pool f_i  +  M · mean_audit(y_i − f_i)

`f_i` = Selbsteinschätzung (z. B. erwartete TP), `y_i` = wahre Größe im Audit. Der Audit-Term entfernt
jede Verzerrung der Selbsteinschätzung, die Vorhersagen auf allen Bildern senken die Varianz. Für
Verhältnis-Metriken (Recall, Precision) wird die Delta-Methode verwendet. Ergebnis: **gültige
Konfidenzintervalle** für die Labelqualität des ganzen Pools – deutlich schmaler als ein Audit allein,
wenn die Selbsteinschätzung gut ist, und nie ungültig, wenn sie schlecht ist.

## 4. Praktischer Workflow

1. `autolabel run --images … --masks … --out run1/` → Labels, Ignore-Masken, Wahrscheinlichkeiten,
   Overlays, `review_queue.csv`, `audit_sample.txt`, `quality_report.json`.
2. Menschliche Review der riskantesten Bilder aus `review_queue.csv` (korrigierte Masken fließen als
   neue Labels zurück → erneuter Lauf).
3. **Getrennt davon** die zufällige Audit-Stichprobe labeln → `autolabel estimate` liefert
   Konfidenzintervalle für Recall, Precision, Dice. (Review-Bilder dürfen *nicht* als Audit dienen –
   sie sind gezielt ausgewählt und würden die Schätzung verzerren.)
4. Training des Produktivmodells auf L ∪ Pseudo-Labels, Ignore-Masken als „ignore index“.

## 5. Annahmen und Grenzen
* Garantien gelten unter **Austauschbarkeit**: die gelabelten Bilder müssen eine Zufallsstichprobe aus
  demselben Pool sein (nicht „die schönsten Beispiele“). Driftet die Produktion, gilt die Garantie nicht.
* Die konforme Garantie ist marginal über Defekte; Defekte desselben Bildes sind abhängig.
* Defekttypen, die in L gar nicht vorkommen, kann das überwachte Modell kaum finden – hier hilft nur
  das Normalitätsmodell als Review-Signal.
* Die Qualität der Selbsteinschätzung beeinflusst nur die Breite der PPI-Intervalle, nicht ihre
  Gültigkeit.
