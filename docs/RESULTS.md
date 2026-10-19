# Ergebnisse Phase 1 – Baseline-Benchmarks

Stand: 19.10.2026, **in Arbeit**. Die Abschnitte 1 und 2 wurden *vor* den ersten vollen Läufen
geschrieben und committet (Vorab-Registrierung der Entscheidungsregeln für G1 und G2). Alle Zahlen
in den folgenden Abschnitten stammen aus echten Läufen auf dem Mac mini M4. Ungetestetes ist als
solches gekennzeichnet.

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
