# Ergebnisse Phase 1 – Baseline-Benchmarks

Stand: 19.10.2026. Die Abschnitte 1 und 2 wurden *vor* dem ersten vollen Lauf geschrieben und
committet (Commit f8b5a97), damit die Entscheidungsregeln für G1 und G2 vorab feststehen. Alle Zahlen
ab Abschnitt 3 stammen aus 24 echten Läufen auf dem Mac mini M4: 4 Datensätze × 2 Varianten × 3 Seeds,
19.10.2026, 13:29–17:05. Angegeben sind Mittelwert ± Standardabweichung über die Seeds, in Prozent,
sofern nicht anders vermerkt. Die vollständigen Tabellen stehen in [RESULTS_raw.md](RESULTS_raw.md).

**Noch offen:** KolektorSDD2 (Loader fertig und getestet, Daten fehlen noch), MVTec AD 2 (Lizenzfrage
offen) und die Oracle-Läufe mit voller GT. Für diese Datensätze bzw. Läufe gibt es keine Zahlen.

## Kurzfassung

1. **Die volle Pipeline ist nicht besser als die Ablation ohne ungelabelte Daten.** In Runde 0 liegt
   `full` beim tol-F1 auf U bei BSData (84,2 ± 8,4 gegen 92,1 ± 1,8), CFD (82,0 ± 4,1 gegen 88,6 ± 2,0)
   und DeepCrack (73,5 ± 7,8 gegen 86,5 ± 2,8) deutlich unter `base`. Bei MT liegen beide gleichauf
   (43,3 ± 11,8 gegen 43,0 ± 12,2). Auch das schwellenfreie Pixel-AP ist bei `full` niedriger, ein
   Schwellenartefakt ist es also nicht. Verloren geht vor allem Defekt-Precision; gewonnen wird
   Recall kleiner Defekte (§4).
2. **Die Schwellenwahl auf 10–15 OOF-Bildern ist der instabilste Baustein.** Bei MT schwankt t_seed
   über die Seeds zwischen 0,2 und 0,8. G2 ist für MT positiv (Lücke 7,0 pp); in Runde 1 auf DeepCrack
   liegt die Lücke bei 20 pp (§7).
3. **G1 ist nur für DeepCrack positiv** (Δρ_low = −8,0 pp). Die Post-hoc-Kontrolle zeigt aber: Mit den
   Schwellen aus Runde 0 hat Runde 1 auf DeepCrack einen *höheren* Recall. Der Verlust stammt aus der
   Schwellenwahl in Runde 1, nicht aus dem Modell (§6).
4. **Die Risikoreihenfolge ist nur wenig besser als Zufall.** Die normierte Fläche unter K′(k)/K liegt
   bei 0,23–0,47 (Zufall 0,50, Orakel 0,02–0,15). Bei c_review/c_audit = 0,25 spart Review in
   Risikoreihenfolge 38–60 % der Kosten bis zum ζ-Zertifikat. Bei `full` wäre dann aber optimal,
   ≈ 90 % des Pools zu reviewen; gegenüber zufälliger Reihenfolge spart das nur −1 bis 16 %. Mit perfekter Reihenfolge
   sänken die Kosten um weitere 30–60 %. Bei gleichen Stückkosten (Verhältnis 1) lohnt Review fast
   nie (§5).
5. **Ein einseitiges ζ-Ziel wird in der Praxis manipuliert.** Niedrige Schwellen drücken ζ, während φ
   steigt. Ein Beispiel ist MT Seed 0, Runde 1: ζ 0,4 % bei φ 3,8 und 2 % Bildspezifität. Die
   Kostenrechnung in §5 kennt nur ζ. Ein φ-Zertifikat (00 §4.7, offen) ist deshalb Voraussetzung für
   jede ehrliche Aussage.


## 1. Vorab festgelegte Entscheidungsregeln

Die Regeln stehen auch im Kopf von `scripts/phase1_gates.py`, das sie mechanisch anwendet.

**Gate G1: Senkt Self-Training (Runde 1 gegen Runde 0) den Recall auf U in den Niedrigrisiko-Strata?**

* Strata sind die Quintile Q1–Q5 des Review-Risikos **aus Runde 0**. Die Referenz ist fest, damit
  beide Runden auf denselben Bildmengen verglichen werden. Niedrigrisiko heißt Q1–Q3, also die
  unteren 60 %, die ein Review in Risikoreihenfolge zuletzt erreicht.
* Kenngröße ist ρ_low(r) = 1 − (Fehlbilder in Q1–Q3) / (defekte Bilder in Q1–Q3) in Runde r, also
  der Bild-Recall (any-hit, Toleranz 2 px).
* **G1 ist positiv** für einen Datensatz, wenn Δ = ρ_low(1) − ρ_low(0) in mindestens 2 von 3 Seeds
  negativ ist **und** das mittlere Δ ≤ −2 Prozentpunkte beträgt.
* Folge laut 00 §4.6: Bei positivem G1 wird doppeltes Cross-Fitting (C4) relevant. Bei negativem G1
  zählt allein Go-Kriterium 5: K′(k\*) muss in Runde 1 gegenüber Runde 0 um ≥ 10 % sinken, sonst
  bleibt Runde 0 der Labeler (00 §4.2 L1).

**Gate G2: Wie weit liegt die OOF-Schwelle von der U-Oracle-Schwelle entfernt?**

* Die OOF-Schwellen (t_seed, t_ext, min_area) werden per Grid-Search auf den Out-of-fold-Vorhersagen
  der gelabelten Bilder L gewählt und auf den Ensemble-Mittelwert über U angewandt. Die
  Oracle-Schwellen liefert dieselbe Grid-Search direkt auf U mit GT, eine nicht erreichbare
  Obergrenze.
* Kenngröße ist die Lücke im tol-F1 auf U (Oracle-Schwelle minus OOF-Schwelle), Runde 0. Berichtet
  werden außerdem beide Schwellen und die Lücke im Defekt-Recall.
* **G2 ist positiv** (Schwellenübertrag verbesserungsbedürftig, etwa durch Kalibrierung auf
  Ensemble-Pseudo-OOF oder Quantilabgleich, HANDOFF §5.1), wenn die mittlere Lücke über die Seeds
  ≥ 5 Prozentpunkte beträgt.

**Kosten(k).** Kosten(k) = k·c_review + n₉₀(K′(k))·c_audit nach 00 §4.5, mit `certify.plan_n`:
exakt hypergeometrisch, fixes n, Korrekturen angerechnet, δ = 0,05, Power 0,9, Rahmen = ganzes U.
Annahmen: perfekter Reviewer (s = 1), der ein Bild beim Review vollständig repariert; K′(k) wird aus
der GT abgelesen. Ziele sind ζ\* ∈ {0,02; 0,05}, Kostenverhältnisse c_review/c_audit ∈ {1; 0,25}.
Das Verhältnis 0,25 entspricht grob einer Verify-Ansicht (3–6 s) gegen eine Vollinspektion unter P
(10–120 s). Beides ist eine Annahme ohne Messung (00 §5.1).

## 2. Setup

* **Hardware:** Mac mini M4 (10 Kerne, 16 GB), PyTorch 2.14.1 auf MPS. Training mit contiguous
  Speicherformat, weil `channels_last` auf MPS ≈ 4× langsamer ist.
* **Protokoll** (`data.make_split`): T = 20 % der Gruppen, L = 10 defekte + 5 defektfreie Bilder
  (Defektarten stratifiziert), U = Rest des Pools. Seeds 0, 1, 2 ändern Split, Initialisierung und
  Synthese.
* **Varianten:**
  * `full`: Default-Konfiguration, also K = 4 Folds, 600 Iterationen, 2 Runden (Runde 1 =
    Self-Training mit Fold-Teacher), Synthese p_synth = 0,35 mit der korrigierten Labelregel aus
    Schritt 2, Normalitätsmodell WRN50.
  * `base`: Ablation ohne ungelabelte Daten (`--rounds 1 --p-synth 0 --no-anomaly`), also keine
    Synthese, kein Normalitätsmodell, kein Self-Training.
* **Metriken:** Toleranz 2 px, merge_radius 2. ζ ist der Anteil der U-Bilder mit mindestens einer
  nicht getroffenen GT-Komponente (any-hit), φ die Zahl der Falschkomponenten pro Bild. SegF1 ist
  das gepoolte Pixel-F1 bei der gewählten Schwelle. Größenklassen beziehen sich auf die Fläche der
  GT-Komponente: S < 100 px ≤ M < 1000 px ≤ L.

## 3. Laufzeit auf dem M4

Gemessen am 19.10.2026, PyTorch 2.14.1, MPS:

| Größe | Wert | Bemerkung |
|---|---|---|
| Training | ≈ 10 s / 100 Iterationen (Batch 8, Crop 192, resnet18d) | allein auf der GPU; Remote-CPU vorher 46 s. Mit 2 parallelen Läufen ≈ 28 s pro Lauf |
| Inferenz | 24–38 ms/Bild, nach 64-px-Padding 26 ms (MT) | inklusive hflip-TTA |
| Datengenerierung | 8–13 ms/Batch | kein Engpass |
| Normalitätsmodell (WRN50, kNN) | 318 s für 1060 MT-Bilder | wächst quadratisch mit |U| |
| Auswertung (Re-Evaluation) | 16 s für 120 Bilder | nach Thread-Pool und schnellerer Hysterese, vorher ≈ 120 s |
| **voller Lauf `full`** (2 Runden, K = 4, 600 it) | MT 34,3 min · BSData 29,5 · DeepCrack 22,2 · CFD 16,5 | jeweils mit einem zweiten Lauf parallel |
| **voller Lauf `base`** | MT 12,4 min · BSData 10,6 · DeepCrack 10,2 · CFD 8,1 | ebenso |
| 24 Läufe insgesamt | 3 h 36 min | 2 Worker parallel |

Zwei Befunde zu MPS: `channels_last` ist dort ≈ 4× langsamer (Commit d7c23dd). Die kNN-Distanzblöcke
des Normalitätsmodells trieben einen Lauf auf 7–11 GB Unified Memory und damit in den Swap (Commit
9f9f299). Ein Lauf allein, ohne zweiten parallel, wurde nicht gemessen.

## 4. Labelqualität auf U

Runde 0 = überwacht (+ Synthese bei `full`), Runde 1 = Self-Training (nur `full`). tol-F1 und Defekt-
Metriken mit 2 px Toleranz; T = gruppengetrenntes Testset.

| Datensatz | Variante | tol-F1 U | Pixel-AP U | Defekt-Recall U | Defekt-Precision U | tol-F1 T (letzte Runde) |
|---|---|---|---|---|---|---|
| BSData | base | **92,1 ± 1,8** | **89,2 ± 2,7** | 79,7 ± 3,8 | **83,9 ± 3,0** | **92,6 ± 3,1** |
| | full R0 | 84,2 ± 8,4 | 80,9 ± 8,0 | 75,9 ± 2,4 | 60,7 ± 6,4 | |
| | full R1 | 87,6 ± 2,7 | 85,6 ± 3,1 | 79,6 ± 3,3 | 50,0 ± 3,2 | 80,4 ± 7,0 |
| CFD | base | **88,6 ± 2,0** | **72,1 ± 1,5** | 94,8 ± 3,2 | 76,0 ± 21,3 | **87,7 ± 4,3** |
| | full R0 | 82,0 ± 4,1 | 58,8 ± 3,3 | 95,3 ± 5,1 | 64,5 ± 24,0 | |
| | full R1 | 77,5 ± 5,6 | 60,2 ± 3,9 | 96,0 ± 2,0 | 75,1 ± 23,1 | 81,9 ± 2,6 |
| DeepCrack | base | **86,5 ± 2,8** | **88,9 ± 2,0** | 80,7 ± 3,6 | **86,3 ± 7,4** | **82,3 ± 5,3** |
| | full R0 | 73,5 ± 7,8 | 71,5 ± 4,3 | 84,0 ± 4,6 | 50,2 ± 17,0 | |
| | full R1 | 63,9 ± 16,0 | 77,1 ± 3,3 | 75,8 ± 9,0 | 78,1 ± 10,0 | 72,2 ± 6,5 |
| MT | base | 43,0 ± 12,2 | 48,8 ± 12,1 | 78,1 ± 23,7 | 15,8 ± 8,3 | 44,5 ± 9,5 |
| | full R0 | 43,3 ± 11,8 | 39,7 ± 9,6 | 77,7 ± 22,3 | 15,0 ± 6,4 | |
| | full R1 | 37,0 ± 3,6 | 36,6 ± 9,7 | 83,6 ± 16,9 | 13,8 ± 4,9 | 40,1 ± 2,1 |

Beobachtungen:

* **`full` gegen `base` (Runde 0).** Die Trainingsdaten unterscheiden sich nur durch die Synthese:
  p_synth = 0,35, Hintergründe aus U, gewählt vom Normalitätsmodell. Das Normalitätsmodell selbst
  geht nicht ins Training ein. Die Synthese senkt die Defekt-Precision stark (BSData 83,9 → 60,7,
  DeepCrack 86,3 → 50,2) und hebt den Recall kleiner Defekte (Tabelle §4.1). Alle `full`-Läufe nutzen
  die korrigierte Labelregel aus Schritt 2. Eine Ablation „alte gegen neue Labelregel" gibt es
  nicht; ob die Korrektur den Abstand vergrößert oder verkleinert, ist **ungemessen**.
* **Ensemble gegen Einzelmodell (Runde 0, `full`):** tol-F1 BSData 84,2 gegen 73,8, CFD 82,0 gegen
  65,6, DeepCrack 73,5 gegen 64,6, MT 43,3 gegen 43,1. Das Ensemble hilft, außer bei MT.
* **MT ist der schwierigste Datensatz.** Die Defekt-Precision liegt unter 16 %, die Streuung über
  Seeds ist groß (K reicht von 17 bis 149 Fehlbildern). Großflächige Klassen (uneven, fray) werden
  nicht synthetisiert.
* **Normalitätsmodell allein** (Schwelle auf L gewählt): Pixel-AP 2,8 (BSData), 9,5 (CFD), 21,2
  (DeepCrack), 8,9 (MT). Als Labeler ist es unbrauchbar, was B2/B5 aus 00 §2 bestätigt.
* **„Recall-sichere" Masken** (Saatschwelle min(t_seed, t_konform)) sind praktisch Vollmasken:
  tol-F1 BSData 43,6 ± 37,8, DeepCrack 14,5 ± 8,1. Die konforme Schwelle aus 10–27 gelabelten
  Defekten liegt oft bei 0,0–0,3 (Tabelle „Recall-Garantie" in RESULTS_raw.md). Das bestätigt B9.

### 4.1 Pool-Qualität: ζ, φ und Recall nach Größenklassen

Größenklassen nach Fläche der GT-Komponente: S < 100 px ≤ M < 1000 px ≤ L. DeepCrack und CFD enthalten
nur defekte Bilder (Prävalenz 100 %, φ_sauber undefiniert); MT hat 28 % Prävalenz, BSData 37–38 %. N =
|U|: MT 1060, BSData 868/868/802, DeepCrack 416/387/422, CFD 84.

| Datensatz | Variante | ζ | K | φ | Recall S | Recall M | Recall L |
|---|---|---|---|---|---|---|---|
| BSData | base | 7,6 ± 0,7 | 65 ± 8 | 0,07 ± 0,02 | 39,9 ± 10,2 | 97,6 ± 1,6 | 93,3 ± 4,7 |
| | full R0 | 9,2 ± 1,2 | 77 ± 7 | 0,22 ± 0,07 | 35,9 ± 15,2 | 95,4 ± 4,1 | 78,3 ± 31,3 |
| | full R1 | 7,7 ± 0,9 | 65 ± 11 | 0,36 ± 0,08 | 35,0 ± 11,1 | 98,7 ± 0,8 | 100,0 ± 0,0 |
| CFD | base | 6,7 ± 4,5 | 6 ± 4 | 0,82 ± 0,60 | 56,2 ± 23,4 | 93,8 ± 4,6 | 100,0 ± 0,0 |
| | full R0 | 6,0 ± 6,3 | 5 ± 5 | 3,12 ± 4,21 | 72,0 ± 20,6 | 93,8 ± 7,3 | 98,7 ± 2,2 |
| | full R1 | 5,2 ± 2,5 | 4 ± 2 | 2,08 ± 2,45 | 68,1 ± 11,9 | 94,9 ± 6,5 | 99,6 ± 0,7 |
| DeepCrack | base | 24,3 ± 2,7 | 99 ± 6 | 0,86 ± 0,47 | 34,8 ± 6,8 | 80,9 ± 6,6 | 99,1 ± 1,6 |
| | full R0 | 19,7 ± 3,4 | 81 ± 16 | 3,38 ± 1,14 | 59,1 ± 5,9 | 83,9 ± 5,6 | 94,1 ± 4,1 |
| | full R1 | 29,7 ± 9,1 | 122 ± 40 | 1,21 ± 0,49 | 46,8 ± 9,6 | 75,9 ± 10,6 | 87,5 ± 8,5 |
| MT | base | 6,7 ± 7,4 | 71 ± 78 | 1,93 ± 1,30 | 56,8 ± 37,1 | 77,9 ± 32,4 | 86,0 ± 11,7 |
| | full R0 | 6,6 ± 6,6 | 70 ± 70 | 2,48 ± 1,91 | 66,7 ± 15,3 | 84,2 ± 21,7 | 75,2 ± 26,7 |
| | full R1 | 4,8 ± 5,0 | 51 ± 53 | 2,42 ± 1,39 | 76,7 ± 11,7 | 88,5 ± 16,2 | 80,9 ± 20,2 |

* **Kleine Defekte (S) sind der Engpass** auf BSData (Recall 35–40 %) und DeepCrack (35–59 %).
  Die Synthese hebt den S-Recall auf DeepCrack (34,8 → 59,1), CFD (56,2 → 72,0) und MT (56,8 → 66,7).
  Erkauft wird das mit 3–4× mehr Falschkomponenten (φ).
* ζ und φ bewegen sich gegenläufig. Ein Vergleich nur über ζ würde auf DeepCrack `full` R0 (ζ 19,7)
  vor `base` (24,3) setzen, obwohl `base` beim tol-F1 13 pp besser ist und viermal weniger Fehlalarme hat.

## 5. K′(k) und Kosten(k)

Normierte Fläche unter K′(k)/K über k/N; 0,5 entspricht Zufall, kleiner ist besser:

| Datensatz | Variante | Risiko | Zufall | Orakel |
|---|---|---|---|---|
| BSData | base | 0,234 ± 0,036 | 0,497 | 0,038 |
| | full R0 | 0,321 ± 0,038 | 0,496 | 0,045 |
| CFD | base | 0,473 ± 0,057 | 0,513 | 0,028 |
| | full R0 | 0,345 ± 0,043 | 0,513 | 0,024 |
| DeepCrack | base | 0,325 ± 0,032 | 0,501 | 0,120 |
| | full R0 | 0,350 ± 0,059 | 0,496 | 0,097 |
| MT | base | 0,465 ± 0,060 | 0,503 | 0,033 |
| | full R0 | 0,357 ± 0,034 | 0,496 | 0,033 |

Minimale Kosten bis zum ζ-Zertifikat, ζ\* = 0,02, c_review/c_audit = 0,25, in Einheiten einer
Audit-Inspektion (perfekter Reviewer):

| Datensatz | Variante | n₉₀(K) ohne Review | Kosten Risiko | k\* Risiko | Kosten Zufall | Kosten Orakel |
|---|---|---|---|---|---|---|
| BSData | base | 711 ± 38 | 285 ± 19 | 449 ± 195 | 339 ± 12 | 147 ± 3 |
| | full R0 | 730 ± 23 | 323 ± 6 | 769 ± 15 | 340 ± 11 | 150 ± 1 |
| CFD | base | 80 ± 2 | 80 ± 2 | 0 | 79 ± 3 | 66 ± 1 |
| | full R0 | 78 ± 5 | 77 ± 5 | 4 ± 7 | 77 ± 5 | 66 ± 1 |
| DeepCrack | base | 389 ± 15 | 213 ± 4 | 379 ± 18 | 220 ± 4 | 143 ± 3 |
| | full R0 | 386 ± 19 | 212 ± 12 | 374 ± 50 | 220 ± 4 | 139 ± 2 |
| MT | base | 693 ± 322 | 368 ± 48 | 718 ± 568 | 364 ± 56 | 152 ± 20 |
| | full R0 | 746 ± 220 | 369 ± 6 | 940 ± 25 | 391 ± 7 | 152 ± 17 |

* **Mit Verhältnis 0,25** spart Review in Risikoreihenfolge gegenüber „kein Review" 38–60 %
  (BSData, DeepCrack, MT). Optimal ist dann aber, fast den ganzen Pool zu reviewen (k\* ≈ 90 % von
  N bei BSData `full`, MT `full` und DeepCrack). Der Abstand zu zufälliger Reihenfolge beträgt nur
  −1 bis 16 %. Mit perfekter Reihenfolge (Orakel) lägen die Kosten 30–60 % niedriger als mit der
  heutigen Risikoreihenfolge.
* **Mit Verhältnis 1** lohnt Review fast nie. Die Ersparnis liegt bei 0–4 %, nur BSData `base` erreicht
  19 ± 15 % (RESULTS_raw.md).
* **CFD** ist mit N = 84 zu klein: n₉₀ ≈ N, Review bringt nichts.
* Das Kostenverhältnis entscheidet, ob Review sich lohnt, und ist **nicht gemessen**. Die einstündige
  Selbst-Zeitmessung aus 00 §5.1 ist damit die billigste Information mit dem größten Einfluss auf
  die Planung.
* Die Kostenrechnung kennt nur ζ. Ein Labeler mit Vollmasken hätte K ≈ 0 und minimale Kosten; ohne
  φ-Ziel ist diese Tabelle deshalb nicht als Rangliste von Labelern zu lesen.

## 6. Gate G1: Self-Training und Recall in den Niedrigrisiko-Strata

Vorab festgelegte Regel (§1): Für jeden Seed wird ρ_low verglichen, der Bild-Recall in Q1–Q3 des
Runde-0-Risikos.

| Datensatz | Δρ_low je Seed (pp) | mittleres Δ | **G1** |
|---|---|---|---|
| BSData | 0,0 · 0,0 · −0,9 | −0,3 | negativ |
| CFD | +3,9 · −3,9 · −2,0 | −0,7 | negativ |
| DeepCrack | −13,2 · −8,2 · −2,8 | −8,0 | **positiv** |
| MT | +7,5 · +8,9 · −4,1 | +4,1 | negativ |

**Post-hoc-Kontrolle (nicht vorab registriert):** Runde 1 mit den Schwellen aus Runde 0 (R1@T0),
`scripts/g1_fixed_threshold.py`:

| Lauf | ρ_low R0 | ρ_low R1 | ρ_low R1@T0 | ζ R0 | ζ R1 | ζ R1@T0 | φ R0 | φ R1 | φ R1@T0 |
|---|---|---|---|---|---|---|---|---|---|
| deepcrack_full_s0 | 83,6 | 70,4 | 87,6 | 23,6 | 40,1 | 16,8 | 3,00 | 0,70 | 4,38 |
| deepcrack_full_s1 | 92,7 | 84,5 | 98,7 | 17,3 | 23,8 | 2,3 | 2,48 | 1,24 | 7,22 |
| deepcrack_full_s2 | 86,2 | 83,5 | 96,1 | 18,2 | 25,1 | 2,8 | 4,67 | 1,68 | 7,64 |
| mt_full_s0 | 91,5 | 99,0 | 93,5 | 4,2 | 0,4 | 2,5 | 2,33 | 3,76 | 2,62 |
| mt_full_s1 | 64,9 | 73,8 | 73,8 | 14,1 | 10,3 | 10,3 | 0,65 | 0,99 | 1,00 |
| mt_full_s2 | 96,8 | 92,6 | 99,5 | 1,6 | 3,9 | 0,2 | 4,46 | 2,50 | 8,66 |

**Entscheidung und Deutung.** Nach der registrierten Regel ist **G1 für DeepCrack positiv, für BSData,
CFD und MT negativ.** Die Kontrolle zeigt allerdings, dass der Recall-Verlust auf DeepCrack nicht aus
dem Modell kommt. Bei festen Schwellen hat Runde 1 *mehr* Recall und *mehr* Fehlalarme: Die
Wahrscheinlichkeiten sind insgesamt höher. Die OOF-Grid-Search wählt in Runde 1 dann t_seed = 0,80
(alle 3 Seeds; U-Oracle 0,43 ± 0,15). Das spricht gegen den in 00 §4.6 vermuteten Mechanismus
(übersehene Defekte werden in-sample als normal gelernt) und für die Schwellenwahl als Ursache, also
G2. **Doppeltes Cross-Fitting (C4) ist durch diese Daten nicht begründet.** Go-Kriterium 5 (K′(k\*)
sinkt in Runde 1 um ≥ 10 %) ist nicht zuverlässig erfüllt. K sinkt bei BSData (77 → 65) und MT
(70 → 51, bei MT jedoch großteils durch Schwellenwechsel) und steigt bei DeepCrack (81 → 122).
Die minimalen Kosten (ζ\* = 0,02, Verhältnis 0,25) ändern sich zwischen den Runden um höchstens 9 %
(MT 369 → 336; sonst ≤ 1 %, RESULTS_raw.md).
Runde 0 bleibt der Labeler (00 §4.2 L1).

## 7. Gate G2: Schwellenübertrag OOF → U

| Datensatz | Lücke tol-F1 je Seed (pp), Runde 0 | mittlere Lücke | **G2** | t_seed OOF | t_seed Oracle |
|---|---|---|---|---|---|
| BSData | 0,7 · 0,0 · 7,0 | 2,6 | negativ | 0,80 ± 0,00 | 0,73 ± 0,06 |
| CFD | 4,3 · 0,5 · 0,0 | 1,6 | negativ | 0,40 ± 0,26 | 0,20 ± 0,00 |
| DeepCrack | 10,1 · 0,8 · 1,0 | 4,0 | negativ (knapp) | 0,33 ± 0,06 | 0,27 ± 0,12 |
| MT | 0,6 · 13,3 · 7,0 | 7,0 | **positiv** | 0,50 ± 0,30 | 0,47 ± 0,12 |

In Runde 1 ist die Lücke größer: DeepCrack 20,1 ± 14,8 pp, MT 13,5 ± 10,5, CFD 8,9 ± 8,1. Ohne
Vorab-Regel für Runde 1 ist das nur beschreibend. Die Schwelle der Ablation `base` überträgt sich auf
MT schlecht (14,2 ± 17,1 pp), auf BSData gut (0,8 ± 0,7).

**Entscheidung:** **G2 ist für MT positiv**, für die anderen drei Datensätze negativ, DeepCrack nur
knapp (4,0 pp). Das Problem ist weniger die Kalibrierung Einzelmodell → Ensemble, die HANDOFF §5.1
vermutete, als die **Varianz der Grid-Search auf 10–15 OOF-Bildern**. Auf MT liegt t_seed über die
Seeds zwischen 0,2 und 0,8, in Runde 1 auf DeepCrack immer am Gitterrand (0,8).

## 8. Weitere Befunde

* **Fehlersignale (AUROC für „Bild mit übersehenem Defekt", `full`):** Rangmittel `risk` 0,68
  (BSData), 0,69 (CFD), 0,74 (DeepCrack), 0,60 ± 0,13 (MT). Das Normalitätssignal `anomaly_out` liegt
  bei 0,45–0,61 und trägt praktisch nichts bei. Die Member-Uneinigkeit `inv_member_dice` ist das
  beste Einzelsignal (0,59–0,76).
* **Fehler in der Selbsteinschätzung (gefunden, nicht behoben):** `quality.image_quality` setzt
  exp_dice = 0 für jedes Bild ohne vorhergesagte Komponente. Wahr ist dort Dice 1, wenn das Bild
  sauber ist. Daher liegt der geschätzte mittlere Dice bei 25 statt 89 % (BSData `base`). Weil
  `inv_exp_dice` in `review_risk` eingeht, rücken alle leeren Labels nach vorn in die Review-Queue.
  Das schadet vermutlich der Risikoreihenfolge bei hoher Sauberquote. Alle Läufe dieser Phase
  enthalten den Fehler.
* **Defekttyp-Zuordnung** (`scripts/eval_types.py`, `mt_full_s0`, 361 Komponenten, 5 Klassen):
  Genauigkeit 0,53, Macro-F1 0,52. Die konformen Mengen decken 0,81 statt der angestrebten 0,90 ab,
  bei mittlerer Mengengröße 2,7 und nur 11 Kalibrierkomponenten.
* **Synthese-Labelregel (Schritt 2):** Auf echten Hintergründen werden 0,9 % (BSData), 2,4 %
  (DeepCrack), 6,9 % (MT) bzw. 17,1 % (CFD) der eingefügten Quellmaskenpixel zu Ignore. Keine
  Einfügung war komplett unsichtbar.

## 9. Einschränkungen

* 3 Seeds; die Streuung ist teils so groß wie die Effekte, vor allem bei MT (K = 17–149).
* K′(k) und Kosten(k) setzen einen perfekten Reviewer voraus, der ein Bild beim Review vollständig
  repariert (s = 1). Gemessen wird gegen GT, nicht gegen Protokoll P.
* Das Kostenverhältnis c_review/c_audit ist angenommen (1 bzw. 0,25), nicht gemessen.
* Die Kosten berücksichtigen nur ζ; ein φ-Zertifikat fehlt (00 §4.7).
* Zwei Läufe liefen jeweils parallel. Laufzeiten gelten für diese Konfiguration; die Ergebnisse
  selbst hängen davon nicht ab.
* Die Re-Evaluation (`reevaluate.py`) arbeitet auf uint8-quantisierten Wahrscheinlichkeiten und
  weicht im 3.–4. Nachkommastellenbereich von der Erstauswertung ab. Alle Tabellen hier stammen aus
  der Erstauswertung.
* BSData Seed 2 hat nur 3 statt 5 defektfreie L-Bilder (Gruppenbeschränkung in `make_split`).
* DeepCrack und CFD haben keine defektfreien Bilder. Dort sind ζ und ρ_b dieselbe Größe, und
  φ_sauber ist undefiniert.

## 10. Vorschlag: nächste Ausbaustufe (00 §4.6)

Ziel ist, K′(k) pro Review-Minute zu senken. Gemessen wird an Kosten(k\*) bei ζ\* = 0,02 und
c_review/c_audit = 0,25. Das Go-Kriterium „K′(k\*) sinkt um ≥ 10 %" taugt bei diesem Verhältnis
nicht: Das Optimum liegt dort meist bei K′(k\*) = 0, weil bis zum letzten Fehlbild reviewt wird.
Vorschlag: Go = **Kosten(k\*) sinken um ≥ 10 %** gegenüber Phase 1, bei unveränderter
Gültigkeit.

**Empfehlung: Stufe 4 bzw. Roadmap-Phase 2, eine bessere Risikoreihenfolge durch ein unabhängiges
Signal.** Konkret: AnomalyDINO mit DINOv2-S (Apache 2.0) als Kontrolldetektor (L2) und ein
logistischer Stacker auf L-OOF (L3). Begründung aus den Daten:

* **Der größte gemessene Spielraum liegt in der Reihenfolge.** Kosten mit Orakelreihenfolge liegen
  30–60 % unter der heutigen Risikoreihenfolge (BSData 147 gegen 285–323, DeepCrack 139–143 gegen
  212–213, MT 152 gegen 368–369). Die heutige Reihenfolge ist kaum besser als Zufall (AUC 0,23–0,47
  gegen 0,50). Schon ein kleiner Teil dieses Abstands erfüllt das Go-Kriterium von 10 %.
* **Das vorhandene Normalitätssignal trägt nichts** (AUROC 0,45–0,61). Ein zweiter, stärkerer und
  vom Labeler unabhängiger Detektor ist genau die Lücke. Das beste vorhandene Signal ist die
  Member-Uneinigkeit (0,59–0,76).
* **Billiger Vortest ohne neue Komponente:** Den exp_dice-Fehler (§8) beheben und die
  Risikoreihenfolge aus den gespeicherten Läufen neu berechnen. Das kostet CPU-Minuten, zeigt, wie
  viel allein die Korrektur bringt, und liefert die Messlatte für den Kontrolldetektor.

Weshalb die anderen Stufen ihr Kriterium in Phase 1 eher nicht erreichen:

| Stufe | Einschätzung aus Phase 1 |
|---|---|
| 1 Stärkerer Encoder | B1 macht ihn plausibel, aber solange die Reihenfolge schlecht ist, wirken weniger Fehlbilder kaum auf die Kosten: k\* hängt an der Position des *letzten* Fehlbilds. Sinnvoll erst nach Stufe 4 oder zusammen mit ihr (gleicher DINOv2-Encoder) |
| 3 Synthese v2 | Die heutige Synthese senkt die Precision stark und hebt nur den S-Recall (§4.1); K sinkt nicht durchgängig (BSData `full` 77 gegen `base` 65). Vorher klären, *warum* `full` schlechter ist (Ablation p_synth, alte gegen neue Labelregel, Hintergrundwahl) |
| 5 Self-Training / C4 | G1 positiv nur auf DeepCrack, und dort über die Schwellenwahl (§6). K′ sinkt nicht konsistent, die Kosten ändern sich um ≤ 9 %. C4 ist nicht begründet |
| 2 PPR-Audit | Rein statistisch, hilft nicht bei K′. Wird interessant, weil K über Seeds stark streut (MT 17–149) und eine fixe Planung dann oft falsch liegt. Prüfbar in CPU-Minuten mit den echten K-Werten |
| 7 PPS/PPI | Für ζ laut 07 ohne Nutzen; Phase 1 liefert keinen Gegenbeleg |

**Vorher nötig, unabhängig von der Stufe** (zum CertAL-lite-Kern bzw. Phase 0 gehörig, daher keine
Ausbaustufe):

1. **φ-Zertifikat:** Ohne zweiseitiges Ziel bevorzugt die Kostenrechnung Vollmasken (§2 Kurzfassung,
   Punkt 5).
2. **Robustere Schwellenwahl im Labeler (L1),** etwa eine glattere Zielfunktion, ein gröberes Gitter
   oder Bootstrap über OOF. Der Grund ist G2 positiv auf MT und die Gitterrand-Schwelle in Runde 1
   auf DeepCrack. Das ist Labeler-Robustheit, keine neue Komponente.
3. **Einstündige Selbst-Zeitmessung** für c_review/c_audit (00 §5.1). Sie entscheidet, ob Review
   überhaupt lohnt (Verhältnis 1: nein; 0,25: ja).

Lizenzen: DINOv2 und anomalib stehen unter Apache 2.0. DINOv3 wäre vor einem Einsatz zu klären.

## 11. Reproduktion

```bash
source runs/env.sh                      # venv, AUTOLABEL_DATA/CACHE, AUTOLABEL_DEVICE=mps
python runs/make_jobs.py > runs/jobs_phase1.txt
bash runs/smokes.sh mt bsdata deepcrack cfd && bash runs/queue.sh runs/jobs_phase1.txt 2
python scripts/summarize.py --runs "runs/[mbcd]*_s[0-9]"
python scripts/phase1_gates.py
python scripts/g1_fixed_threshold.py
```

Die Hilfsskripte unter `runs/` (Queue, Smokes) sind nicht versioniert. Jeder Lauf entspricht
`python scripts/run_benchmark.py --dataset <d> --seed <s> [--rounds 1 --p-synth 0 --no-anomaly]`.
