# 06 – Vollständigkeitsschätzung: Wie viele Defekte übersehen die Auto-Labels?

Theoriebeitrag zu `auto-label-tool` · Stand 18.10.2026.
Bezüge: [01](01_foundation_models_anomaly.md) (Idee 11.3), [02](02_defect_synthesis.md) (N1),
[03](03_semisupervised_pseudolabels.md) (N2). Alle drei Reviews kommen unabhängig voneinander auf
dieselbe Lücke: **Keine etablierte Methode schätzt die übersehenen Defekte (FN) eines Auto-Labelers
direkt.** Dieses Dokument formalisiert eine Lösung. Es ist ein Vorschlag, nichts davon ist empirisch
geprüft.

---

## 1. Warum die bisherige Selbsteinschätzung blind für übersehene Defekte ist

Die aktuelle Selbsteinschätzung (`quality.py`) schätzt erwartete FN aus den kalibrierten
Wahrscheinlichkeiten **desselben** Modells, das die Labels erzeugt hat. Ein Defekt, den das Modell gar
nicht „sieht“ (p ≈ 0), trägt fast nichts bei. Das ist strukturell: Ein Detektor kann seine eigenen
blinden Flecken nicht messen. Die Kalibrierung auf OOF-Daten hilft nur, solange die Defekte in U
denen in L ähneln. Gerade bei **unbekannten Defekttypen, schwachem Kontrast, ungewohntem Kontext**
versagt sie, also genau dort, wo übersehene Defekte entstehen.

Zwei Informationsquellen sind unabhängig von den blinden Flecken des Labelers:

1. **Andere Detektoren** mit anderem induktivem Bias: Normalitätsmodell (DINO-kNN), ein zweiter
   überwachter Member mit anderem Backbone, ein Zero-Shot-VFM. Wo sie Defekte finden, die der Labeler
   nicht findet, wird der blinde Fleck sichtbar.
2. **Künstlich eingefügte Defekte** mit bekannter Lage. Bei ihnen ist auch das Ereignis „von allen
   Detektoren übersehen“ beobachtbar, das bei echten Defekten prinzipiell unbeobachtbar bleibt.

Ökologie (Fang–Wiederfang), Epidemiologie (Multiple-Systems Estimation) und Astronomie
(Injection–Recovery) lösen genau dieses Problem seit Jahrzehnten: Wie groß ist eine Population, deren
nicht beobachteter Teil per Definition unsichtbar ist? Wir übertragen das auf Pixel-Auto-Labeling.

---

## 2. Modell

* **Population:** U enthält N echte Defekte d = 1…N (Zusammenhangskomponenten nach der
  `merge_radius`-Konvention von `metrics.py`), N unbekannt. Jeder Defekt hat Kovariaten x_d (Fläche,
  lokaler Kontrast bzw. Normalitäts-Score, Elongation, Klasse, Kontext-Cluster).
* **Detektorpanel:** J ≥ 2 Detektoren. Detektor j „fängt“ Defekt d (δ_dj = 1), wenn eine seiner
  Kandidatenregionen den Defekt innerhalb der Toleranz trifft. Detektor 1 = A ist der Labeler (die
  ausgelieferten Masken). Fanggeschichte h_d = (δ_d1, …, δ_dJ) ∈ {0,1}^J.
* **Beobachtet** wird nur, was mindestens ein Detektor fängt. Diese Regionen r ∈ R enthalten auch
  Fehlalarme. Jede Region bekommt q_r = P(echter Defekt | Merkmale), aus dem OOF-kalibrierten
  Regions-Stacker (01-R3, 03-R4b).
* **Erfassungswahrscheinlichkeit:** π(x) = P(h ≠ 0 | x) = 1 − P(h = 0 | x).

Gesucht sind
(i) N (Gesamtzahl der Defekte in U),
(ii) N_A = Zahl der echten Defekte, die A erfasst, also der **Defekt-Recall des Pools** R_A = N_A / N,
(iii) pro Bild i die erwartete Zahl übersehener Defekte, als Prädiktor f_i für das Audit (Abschnitt 5).

---

## 3. Drei Schätzer mit unterschiedlichen Annahmen

### 3.1 Horvitz–Thompson mit Injection–Recovery (Annahme T: Transportierbarkeit)

Es werden synthetische Defekte s mit bekannten Kovariaten x_s in zufällige U-Bilder eingefügt
(Residual-Paste aus der Defektbank, Gain und Größe gestreut). Alle Detektoren laufen auf den
modifizierten Bildern, also werden die **vollständigen** Fanggeschichten inklusive h = 0 beobachtet.
Daraus lässt sich P(h | x) als Regression schätzen, z. B. multinomial-logistisch über die 2^J Zellen
oder als J logistische Modelle mit Interaktionstermen. Insbesondere entsteht so π̂(x).

**Annahme T:** Für echte und synthetische Defekte mit gleichen Kovariaten sind die
Fangwahrscheinlichkeiten gleich: P_real(h | x) = P_syn(h | x).

Horvitz–Thompson-Schätzer:

    N̂_HT = Σ_{r ∈ R} q_r / π̂(x_r)
    N̂_A  = Σ_{r ∈ R, r ⊂ A} q_r            (direkt beobachtet, kein π nötig)
    R̂_A  = N̂_A / N̂_HT

*Erwartungstreue (Skizze):* Bei korrektem q und π gilt E[Σ_d 1{h_d≠0}/π(x_d)] = Σ_d π(x_d)/π(x_d) = N.
Die Varianz ist näherungsweise Σ_d (1 − π_d)/π_d plus der Schätzfehler von π̂, letzterer per Bootstrap
über die Injektionen. Die Kovariaten werden nur für **gefangene** Defekte gebraucht. Voraussetzung ist
Positivität (π(x) > 0 für alle echten Defekte). Defekte, die *kein* Detektor je finden kann, sind aus
Fanggeschichten prinzipiell nicht schätzbar. Für sie bleibt nur das Audit (Abschnitt 5).

### 3.2 Huggins–Alho: bedingte Likelihood auf echten Fanggeschichten (Annahme CI)

Ohne Synthetik gilt unter **bedingter Unabhängigkeit der Detektoren gegeben x**
P(h | x) = Π_j p_j(x)^{h_j} (1 − p_j(x))^{1 − h_j}. Die p_j(x) werden aus den **echten**
Fanggeschichten der beobachteten Defekte geschätzt, über die auf h ≠ 0 bedingte Likelihood:

    ℓ(θ) = Σ_{r ∈ R} q_r · log[ P_θ(h_r | x_r) / (1 − P_θ(0 | x_r)) ],   p_j(x) = σ(θ_jᵀ x)

Danach gilt N̂_HG = Σ_r q_r / π̂_θ(x_r). Das ist der Huggins-(1989)- bzw. Alho-(1990)-Schätzer aus der
Ökologie. Heterogenität, also dass kleine, kontrastarme Defekte für alle Detektoren schwer sind, ist über
x modelliert. Restabhängigkeit, also gemeinsame blinde Flecken, die x nicht erklärt, führt zu
**Unterschätzung** von N. Ab J = 3 lassen sich paarweise Abhängigkeiten über log-lineare Modelle
(Multiple-Systems Estimation) schätzen. Annahme ist dann nur noch, dass die Dreifach-Interaktion null
ist.

### 3.3 Hybrid: Abhängigkeitsstruktur aus der Injektion, Niveau aus echten Daten

T und CI sind beide angreifbar: Synthetik ist nie ganz real, und Detektoren sind nie ganz unabhängig.
Der Hybrid nimmt aus der Injektion nur, was dort gut messbar und vermutlich robust ist, nämlich die
**Abhängigkeitsstruktur** (Interaktionsterme, etwa „wenn A übersieht, übersieht B mit Faktor κ(x)
häufiger“). Die **Haupteffekte** p_j(x) werden über die bedingte Likelihood auf echten Daten geschätzt.
Formal ist das ein log-lineares Modell mit Offset: Die Interaktionsparameter sind aus den synthetischen
Daten fixiert, die Haupteffekte aus den echten Daten gefittet.

---

## 4. Die Transportannahme ist ohne Ground Truth testbar

Das ist der aus unserer Sicht wichtigste neue Punkt. Aus der Injektion folgt eine Vorhersage für die
**beobachtbaren** Zellen echter Defekte:

    P(h | x, h ≠ 0) = P_syn(h | x) / (1 − P_syn(0 | x))   für h ∈ {0,1}^J ∖ {0}

Für echte gefangene Defekte sind die Anteile der Zellen „nur A“, „nur B“, „A und B“ usw. direkt
beobachtbar (gewichtet mit q_r). Stimmt die Vorhersage aus der Synthetik mit diesen Anteilen
überein, pro Kovariaten-Bin geprüft per χ²- bzw. Kalibrierungstest, ist T auf dem beobachtbaren Teil
konsistent. Weicht sie ab, ist die Synthetik für diese Kovariaten nicht repräsentativ. Das System
merkt das **ohne einen einzigen gelabelten U-Defekt** und kann

* die Strata mit schlechtem Fit im Audit hochgewichten (Abschnitt 5),
* auf den Hybrid (3.3) oder auf Huggins (3.2) ausweichen,
* den Fit-Status im Qualitätsbericht ausweisen („Vollständigkeitsschätzung in Stratum *groß* nicht
  vertrauenswürdig“).

Die drei Schätzer beruhen auf verschiedenen Annahmen. **Ihre Übereinstimmung ist selbst ein
Diagnosesignal** (Triangulation): Liegen N̂_HT, N̂_HG und N̂_Hybrid weit auseinander, ist mindestens
eine Annahme verletzt, und das Audit-Budget sollte steigen.

Eine einschränkende Bemerkung: Der Test prüft T nur auf dem *gefangenen* Teil. Ein Unterschied zwischen
Synthetik und Realität, der ausschließlich die Zelle h = 0 betrifft, bleibt unsichtbar, z. B. ein
realer Defekttyp, den *kein* Detektor sieht und den die Synthetik nicht abbildet. Gegen diesen Fall
hilft nur das zufällige Audit. Deshalb ist das Audit Pflicht und nicht optional.

---

## 5. Kopplung an das Audit: Gültigkeit unabhängig von den Annahmen

Die Schätzer aus Abschnitt 3 sind **modellbasiert**. Gültige Konfidenzintervalle entstehen erst durch
die Kopplung mit einem Audit nach Design, also Zufallsstichprobe oder Poisson-Sampling mit bekannten
Inklusionswahrscheinlichkeiten π_i^audit (siehe 05, Abschnitt 4.2). Die Rollen sind:

* **Prädiktoren pro Bild** für den Differenz- bzw. IPW-PPI-Schätzer von `ppi.py`:

      f_i^{N}  = Σ_{r ∈ i} q_r  +  m̂_i                          (erwartete echte Defekte in Bild i)
      f_i^{A}  = Σ_{r ∈ i, r ⊂ A} q_r                          (davon von A erfasst)
      m̂_i      = (N̂ − Σ_r q_r) · w_i / Σ_k w_k,   w_i = 1 − s̄_i

  m̂_i verteilt die geschätzten „von allen übersehenen“ Defekte auf die Bilder. s̄_i ist die mittlere
  Injection-Recovery-Rate in Bild i: In Bildern mit Reflexen, starker Textur oder Sättigung, wo
  synthetische Defekte schlecht gefunden werden, landet mehr der unsichtbaren Masse. Das ist
  02-N1 als Prädiktor.
* **Audit-Schätzer:** R_A = Σ_i y_i^A / Σ_i y_i^N mit den wahren Zählungen y aus dem Audit, geschätzt
  mit `ppi.estimate_ratio(…, fa = f^A, fb = f^N)` bzw. mit der IPW-Variante bei nicht-uniformen π.
* **Garantie:** Die Gültigkeit des Intervalls hängt nur vom Stichprobendesign ab, nicht von T, CI oder
  der Qualität von q. Schlechte Prädiktoren machen das Intervall *breiter*, nie *falsch*. Gute
  Prädiktoren, also gute Vollständigkeitsschätzung, machen es *schmaler*. **Damit wird die
  Vollständigkeitsschätzung ein reiner Effizienzgewinn ohne Gültigkeitsrisiko.**
* **Stratifizierung:** Die Strata aus dem Transporttest (Abschnitt 4) und der Injection-Recovery-Rate
  definieren die Audit-Inklusionswahrscheinlichkeiten. Wo die Modelle unsicher oder inkonsistent sind,
  wird häufiger auditiert (Active Inference nach Zrnic & Candès 2024, siehe Bericht 04).

---

## 6. Fallstricke und wie das Design sie entschärft

1. **Synthetik-Leck:** Das Ensemble wurde mit Residual-Paste derselben Defektbank trainiert. Injiziert
   man dieselben Defekte, findet es sie leichter als echte (π überschätzt → N unterschätzt).
   Gegenmittel:
   (a) *cross-fitted Injection*: Für Member k nur Defekte aus Fold k injizieren, die Member k nie
   gesehen hat. Die Ensemble-Entscheidung wird dann über die Member-Scores mit
   OOF-Schwellen approximiert.
   (b) Eine reservierte Injektionsbank aus L_cal, also Defekte, die in keine Synthese eingehen (vgl. die
   L_cal/L_train-Trennung in 05).
   (c) Defekte aus *anderen* Aufgaben bzw. öffentlichen Datensätzen als „fremde“ Injektionen.
   Der Transporttest (Abschnitt 4) deckt ein verbleibendes Leck als Fehlanpassung auf.
2. **Abhängigkeit durch Design:** Wird die Anomaliekarte als Eingangskanal in das U-Net gespeist
   (01-R4), sind A und B nicht mehr annähernd bedingt unabhängig. **Konsequenz für die Architektur:**
   Das Detektorpanel wird auf *Unabhängigkeit* optimiert, nicht nur auf Genauigkeit. Mindestens ein
   Detektor bleibt vom Labeler informationell getrennt: anderer Backbone, anderes Lernprinzip, keine
   gemeinsamen Eingaben außer dem Bild.
3. **Fehlalarme:** Ohne q_r würde jede Normalitäts-Reflexion als „von A übersehener Defekt“ zählen.
   q_r stammt aus dem OOF-kalibrierten Regions-Stacker. Seine Fehlkalibrierung wird über das Audit
   korrigiert (die Gültigkeit bleibt erhalten, siehe Abschnitt 5).
4. **Regionsabgleich:** Kandidaten verschiedener Detektoren müssen demselben Defekt zugeordnet werden.
   Dafür Überlapp mit Toleranz und `merge_radius` wie in `metrics.py`. Mehrdeutige Zuordnungen
   (ein großer B-Kandidat überdeckt zwei A-Komponenten) werden als ein Defekt gezählt; das ist eine
   Konvention, die in Schätzer und Metrik identisch sein muss.
5. **Abhängigkeit innerhalb eines Bildes:** Alle inferenzrelevanten Größen werden pro **Bild**
   aggregiert. Das Bild ist die Stichprobeneinheit des Audits, sodass die Abhängigkeit zwischen
   Komponenten desselben Bildes die Gültigkeit nicht berührt.

---

## 7. Designprinzip, das daraus folgt

> **Ein Auto-Labeler soll nicht nur gute Labels erzeugen, sondern seine eigene Unvollständigkeit
> messbar machen.** Dafür braucht er ein Panel von Detektoren mit *möglichst unkorrelierten Fehlern*,
> einen Mechanismus, der die unbeobachtbare Zelle „von allen übersehen“ beobachtbar macht (Injection),
> und ein Audit-Design, das die Gültigkeit garantiert, wenn Modellannahmen versagen.

Für die Ensemble-Konstruktion verschiebt das die Zielfunktion: Neben OOF-Genauigkeit zählt die
**Fehlerkorrelation** zwischen Labeler und Kontrolldetektoren. Ein schwächerer, aber unabhängiger
Kontrolldetektor kann für die Messung wertvoller sein als ein stärkerer, korrelierter.

---

## 8. Validierungsplan (falsifizierbar)

Datensätze mit vollständiger U-Ground-Truth: MT, BSData, DeepCrack, CFD (verfügbar); lokal KSDD2,
MVTec AD 2 (`test_public`), VisA. Dazu L = 15, 3 Seeds, 3 Labelbudgets.

| Hypothese | Messung | Verworfen, wenn |
|---|---|---|
| H1: Die Vollständigkeitsschätzung trifft die wahre Zahl übersehener Defekte besser als die bisherige Selbsteinschätzung | relativer Fehler von N̂ − N̂_A gegen die wahre Zahl auf U, für N̂_HT, N̂_HG, N̂_Hybrid vs. Σ exp_missed (`quality.py`) | keiner der drei Schätzer senkt den mittleren absoluten Fehler um ≥ 25 % |
| H2: Der Transporttest erkennt schlechte Synthetik | absichtlich verschlechterte Injektion (falsche Gain-Verteilung, nur kleine Defekte) | χ²-Test schlägt in < 80 % der verschlechterten Fälle an oder in > 20 % der korrekten |
| H3: Vollständigkeitsprädiktoren verengen PPI-Intervalle | Breite des 90-%-CI für R_A bei n_audit = 10/20/40, mit f aus Abschnitt 5 vs. ohne | Breite sinkt um < 15 % oder Abdeckung < 88 % |
| H4: Unabhängigkeit schlägt Stärke | Panel A+B mit B = DINO-kNN (unabhängig) vs. B = zweiter U-Net-Member (stärker, korreliert) | der korrelierte Partner liefert die kleineren Fehler in H1 |
| H5: Positivitätslücke ist klein | Anteil echter U-Defekte, die *kein* Detektor findet, pro Größenklasse | > 10 % in einer Größenklasse → das Panel braucht dort einen weiteren Detektor |

---

## 9. Einordnung

* Capture–recapture mit Kovariaten: Huggins (1989, *Biometrika*), Alho (1990, *Biometrics*);
  log-lineare Multiple-Systems-Schätzung: Fienberg (1972, *Biometrika*), Bishop/Fienberg/Holland
  (1975). In der Software-Inspektion ist die Schätzung verbleibender Fehler aus mehreren Inspektoren
  etabliert (Eick et al. 1992, ICSE; Briand et al. 2000, TSE). Diese Verweise sind aus Vorwissen,
  in dieser Session nicht neu geprüft **(unverified)**.
* Injection–Recovery zur Vollständigkeitsschätzung ist Standardpraxis in Astronomie-Surveys
  (Quellenkataloge, Exoplaneten-Transits).
* Neu ist nach den Recherchen 01–03 die Kombination für **Pixel-Auto-Labeling**: (a) Injection beobachtet
  die Zelle „von allen übersehen“, (b) echte Fanggeschichten testen die Transportannahme ohne GT,
  (c) PPI/IPW-Audit macht das Ganze annahmefrei gültig, (d) das Panel wird nach Unabhängigkeit
  konstruiert. Eine Vorarbeit mit genau dieser Kombination haben die Recherchen nicht gefunden; das
  ist keine erschöpfende Suche.
