# 00 – Synthese v2 (nach Red-Team): Auto-Labeling mit gültigem Pool-Zertifikat

Stand 19.10.2026. Zusammenführung der Recherchen [01](01_foundation_models_anomaly.md)–
[05](05_human_in_the_loop_task_adaptation.md), des Theoriebeitrags [06](06_completeness_theory.md) und
des Red-Team-Berichts [07](07_red_team.md). Version 1 (18.10.2026) schlug unter dem Namen „CertAL“
eine fünfschichtige Architektur vor. Diese Fassung setzt die Änderungen aus 07 §7 um, die 00 betreffen
(Nr. 1–5, 7, 8, 10–17, 23, 24).

Der Rahmen bleibt **aufgabenunabhängig**: verschiedene Produkte, Materialien und Kameras, Defekte klein
bis groß, eine oder mehrere Klassen, 5–30 Masken pro Aufgabe, Tausende ungelabelte Bilder. Übersehene
Defekte sind der teuerste Fehler, und die Labelqualität muss messbar sein.

> **Was sich gegenüber v1 geändert hat**
>
> * **Empfohlener Kern ist CertAL-lite** (§4): Labeler, Review in Risikoreihenfolge, danach ein
>   uniformes Audit ohne Zurücklegen aus dem *ganzen* Pool mit exakter Schranke bei fixem n. Das ist
>   ein klassischer Abnahmeprüfplan mit Korrektur (LTPD, Rectifying Inspection), keine neue Methode
>   ([07 §6](07_red_team.md)).
> * **Gültig nur unter registrierten Annahmen:** Messprotokoll P statt „Mensch = Gold“, eingefrorene
>   und gehashte Größen, harte Schranken wo nötig, δ-Aufteilung über Runden ([07 §1, §3.1–3.3](07_red_team.md)).
> * **Das Review-Stratum gehört in den Stichprobenrahmen.** In v1 lag es außerhalb; im Beispiel von
>   07 war dann jedes ausgestellte Zertifikat falsch ([07 §3.2](07_red_team.md)).
> * **Verify-first dient nur der Reparatur**, nie der Messung ([07 §3.1](07_red_team.md)).
> * **Defekt-Recall ist bei seltenen Defekten nicht zertifizierbar.** Primärziel ist die Fehlbildrate
>   ζ, der Bild-Recall ρ_b wird daraus abgeleitet, eine Regime-Weiche entscheidet vorab
>   ([07 §3.4](07_red_team.md)).
> * **Effizienz entsteht im Labeler und in der Risikoreihenfolge**, nicht in raffinierteren Schätzern (§4.5).
> * **Neuheit:** Alle sieben Beiträge C1–C7 aus v1 sind teilweise bekannt (§5.3, [07 §4](07_red_team.md)).
> * **Nordstern** ist ein Paar aus Menschminuten und Falschzertifikatsrate unter fehlbarem Reviewer
>   (§5.1, [07 §5.3](07_red_team.md)).
> * **Gestrichen:** „nie falsch“ ohne Bedingungen, „Mensch = Goldstandard“ als Gegebenheit, „um
>   Größenordnungen schärfer“, „löst das Unterdeckungsproblem“, Lemma 0 als Beitrag, Kacheln als
>   Stichprobeneinheit, die exakte hypergeometrische Schranke in einem jederzeit stoppbaren Audit.
> * PPS/PPI, Satz 2, Kacheln, Vollständigkeitsschätzer (06), doppeltes Cross-Fitting und die
>   Aufgabenanpassung S5 sind **Ausbaustufen hinter gemessenen Go-Kriterien** (§4.6).

> **Quellenlage.** Literaturzahlen tragen die Marker der Einzelberichte ([R] README/Ergebnisdatei,
> [A] Abstract/Snippet, [P] Textauszug des Papers, „unverified“ = nicht geprüft). Zahlen aus 07
> stammen aus kleinen Monte-Carlo-Läufen oder exakten Rechnungen auf **synthetischen Pools**. Sie
> tragen den Prüfstatus aus 07: B = vom Gegenprüfer bestätigt, U = umstritten, N = nicht
> gegenverifiziert. Auf echten Daten ist davon nichts gemessen. Alle Gewinnerwartungen sind Hypothesen.

---

## 1. Kernaussage

Kein Modell wird aus 5–30 Masken auf einer fremden Industrieoberfläche *beweisbar* gute Labels
erzeugen. Selbst die besten unüberwachten DINOv3-Verfahren erreichen auf MVTec AD 2 nur ≈ 57 % SegF1
[A], Few-Shot-Foundation-Modelle liegen auf Defekten bei 28–48 mIoU [A]. Die **Arbeitsteilung** aus v1
bleibt deshalb bestehen: *Modelle* erzeugen Labels und einen Risikoscore pro Bild. Ein **registriertes,
randomisiertes Audit** liefert das Zertifikat und korrigiert dabei, was es prüft. Zertifiziert wird der
*ausgelieferte*, korrigierte Pool.

Neu ist die genaue Bedingung, unter der das Zertifikat gilt. Es ist gültig, **sofern die registrierten
Annahmen gelten**:

1. **Messprotokoll P.** Jede Audit-Einheit wird zuerst blind vollständig abgesucht. Erst danach wird
   das Overlay der Auto-Labels eingeblendet, und als Befund gilt die Vereinigung beider Durchgänge.
   Das Zertifikat gilt relativ zu P, nicht relativ zu einer unerreichbaren Wahrheit. Verify-first
   („Label OK?“) dient nur der Reparatur und zählt nie als Messung.
2. **Eingefrorene Größen.** Vor der ersten Auditziehung werden Labels, Risikoscore r, Konventionen
   (Toleranz, merge_radius), Ziele, die δ-Aufteilung, R_max, n und der Zufallsseed festgelegt und
   gehasht.
3. **Rahmen = ganzer Pool.** Das Audit zieht aus allen ausgelieferten Bildern, das Review-Stratum R
   eingeschlossen, es sei denn, R wurde vollständig unter P inspiziert.
4. **Harte Schranken, wo die Zielgröße sie braucht.** Für ζ und φ sind sie bekannt (b_i ∈ {0, 1},
   F_i ≤ P_i). Zähl-Recall bräuchte eine vorab begründete Schranke D_max, die das Design nicht prüfen
   kann. Er gehört deshalb nicht zum Kern.
5. **δ-Spending über Runden.** Folgt auf ein gescheitertes Audit ein Retraining und ein neues Audit,
   gilt Σ_r δ_r ≤ δ.

Unter diesen Annahmen gilt weiter: **Genauigkeit kauft Effizienz, das Stichprobendesign kauft
Gültigkeit.** Ein schlechteres Modell macht das Zertifikat teurer, aber nicht häufiger falsch: Die
Falschzertifikatsrate bleibt ≤ δ, solange die registrierten Annahmen gelten. Ist eine verletzt, etwa
weil die Inspektion unter P Defekte übersieht, kann sie gegenüber der Wahrheit steigen. Der v1-Satz
„Schlechte Modelle machen das Zertifikat teurer, aber nie falsch“ war ohne diese Bedingungen falsch:
In v1 lag R außerhalb des Rahmens, und ein schlechteres Modell schob mehr Fehlbilder dorthin
(07 §3.2, B).

Das Ergebnis ist ein **statistisches Pool-Zertifikat (designbasiert)**. Es ist keine
Prozesszertifizierung nach ISO/IEC 5259 und sagt nichts über künftige Produktionsbilder.

> **Beispielziel (zweiseitig, vorab registriert):** „Höchstens 2 % der Bilder im ausgelieferten Pool
> enthalten einen unter P auffindbaren Defekt, den das Label nicht trifft (ζ_final ≤ 0,02), und die
> Zahl falscher Komponenten pro Bild ist höchstens φ\*, gemeinsam mit Konsumentenrisiko δ = 0,05
> (aufgeteilt auf beide Aussagen).“
> Eine Schranke für den Recall defekter Bilder ρ_b folgt daraus ohne zusätzliches δ (§4.3). Ein
> direktes Recall-Zertifikat ist bei seltenen Defekten meist nicht machbar (§4.4).

Effizienz entsteht fast nur dadurch, dass vor dem Audit weniger Fehlbilder übrig sind (§4.5). Der
Forschungsaufwand gehört deshalb in einen besseren Labeler, eine bessere Risikoreihenfolge und
unabhängige Fehlersignale, nicht in raffiniertere Schätzer. Gemessen wird jede Komponente am
Nordstern-Paar aus §5.1.

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
| B9 | **Modellbasierte Garantien aus L sind mit ≈ 10 Bildern praktisch nicht zu haben.** Die marginale Split-Conformal-Abdeckung ist bedingt Beta(11,1)-verteilt: P(Recall < 0,9) = 0,31. Bildweise CRC ist unerfüllbar. **Designbasierte Garantien** (randomisiertes Audit mit bekannten Auswahlwahrscheinlichkeiten) brauchen keine Austauschbarkeit. Sie gelten aber nur relativ zum Messprotokoll, für eingefrorene Größen und bei Zählmetriken wie Recall nur unter einer harten Schranke pro Einheit (07 F1, B) | 04, 07 |
| B10 | **Für Pixel-Auto-Labels fehlt eine etablierte Methode, übersehene Defekte direkt zu schätzen.** In Software-Test (Fehler-Seeding nach Mills), zerstörungsfreier Prüfung (POD), Astronomie (Injection-Recovery), Daten-Cleaning (DQM) und TAR (Chao-Stopp) gibt es direkte Verfahren. Ihre tragende Annahme, dass sich die Entdeckbarkeit von injizierten auf echte Defekte überträgt, ist ohne GT nicht prüfbar (07 F3, B). Für das Zertifikat werden sie nicht gebraucht, höchstens als Planungs- und Risikosignal | 01, 02, 03, 06, 07 |

---

## 3. Schwächen im Code und im Entwurf v1

### 3.1 Im bestehenden Code

Sortiert nach Schwere. Details stehen in 04 §4.8, 02 §9 und 07 §2.

1. **Die Recall-Garantie gilt für U nicht** (04):
   * Schwellen aus Einzel-Member-OOF werden auf den Ensemble-Mittelwert angewandt. Wegen
     max-of-mean ≤ mean-of-max ist das antikonservativ: Im Toy-Modell liegt die Missrate bei
     0,12–0,32 statt 0,1.
   * U-Vorhersagen sind *in-sample*. Runde 0 nutzt 70 % von U als Negativ-Hintergründe, Runde 1
     trainiert auf allen U-Pseudo-Labels. Genau die übersehenen Defekte werden so als „normal“ gelernt.
   * Komponenten werden gepoolt, obwohl sie innerhalb eines Bildes abhängig sind.
   * `certified_recall` wird bei einem per F1-Grid-Search gewählten `t_seed` ausgegeben. Das ist
     nicht zertifiziert.
2. **Die Audit-Intervalle sind nur approximativ** (04 S1/S2, 07 F5). Bei seltenen Defekten liegt die
   Abdeckung von audit-only teils bei 0,23–0,69 statt 0,95, von PPI mit λ = 1 bei 0,83–0,88; getuntes λ
   unterdeckt. `ppi.estimate_ratio` (Delta-Methode, t-Quantil, SRS-FPC) erreicht mit schlechtem
   Prädiktor 0,884 bei n = 50, ohne Prädiktor 0,594. Bei tb ≤ 0 gab v1 NaN aus statt „kein
   Intervall“; seit v2 liefert es das triviale Intervall [0, 1]. Für Zertifikate ist das Modul
   ungeeignet.
3. **Unsichtbare Positive in der Synthese** (02): Wird ein additives Residuum geclippt oder mit kleinem
   Gain eingefügt, ist der Defekt unsichtbar, bleibt aber als positiv gelabelt. Das trainiert
   Fehlalarme und ist ein Korrektheitsfehler. Außerdem fehlen großflächige Defekte, multiplikative
   Defekte (Verfärbungen) und eine kontextpassende Platzierung.
4. **Review und Audit sind getrennt**, und die Qualität des *korrigierten* Pools wird nicht berichtet (04).
5. **Normalitätsmodell:** WRN50 mit 8-px-Raster ist der schwächste Baustein. Es gibt keinen Schutz gegen
   konsistente Anomalien und keinen Gruppenausschluss (01).
6. **Die Simulationen, auf die sich v1 stützte, enthielten Fehler** (07 F4, F16; in v2 der Sims
   behoben, siehe 04 K1):
   * `sims/sim_recall_cert.py` v1 berechnete die Wettobergrenze aus der *aktuellen* Ziehung. Sie
     war nicht prädiktabel. Auf einem adversarialen Pool lag der Fehler 1. Art bei 0,63–0,99, mit
     korrekter Obergrenze bei 0,000 (B). Die Abdeckung 1,00 der alten Spalte „Betting-Fieller“ in
     04 S1 belegte nur Konservativität auf einem gutartigen Pool. v2 berechnet die Obergrenze
     F₀-messbar über den ganzen Pool, prüft Z_t ≤ z^max zur Laufzeit (Verletzung: kein Zertifikat);
     S1 ist neu erzeugt. Ein verkleinerter adversarialer Test liegt in `tests/test_sims_adversarial.py`.
   * `sims/sim3.py` v1 gab als UCB den größten nicht abgelehnten Gitterpunkt aus. Das war um bis
     zu einen Gitterschritt antikonservativ (N). `sims/sim2.py` v1 ließ in der Obergrenze einen Term
     weg, praktisch ohne Wirkung. Beides ist in v2 korrigiert, S2 ist neu erzeugt.

### 3.2 Im Entwurf v1

1. **Review-Stratum außerhalb des Rahmens.** Lemma 0 setzte M_i = 0 für i ∈ R, und das Audit zog nur
   aus U′ = U∖R. Restmisses in R fehlten im Zertifikat, und das wirkt antikonservativ. Beispiel mit
   N = 2000, 600 defekten Bildern, Top-700-Review und Reviewer-Sensitivität s = 0,7: Die
   Falschzertifikatsrate beträgt 0,55 und ist damit gleich der Zertifizierungsrate. Bei s = 0,9 sind
   es 0,123 (07 §3.2, B).
2. **Verify-first im Messpfad.** v1 S4 maß jede geprüfte Einheit verify-first. Gemessen wird dann s·ζ
   statt ζ. Bei wahrem ζ = 0,03, Ziel 0,025 und s = 0,7 (angenommen, nicht gemessen) ergeben sich
   0,135 Falschzertifikate bei n = 400 und 0,330 bei n = 1000, nominal 0,05. Der Fehler wächst mit n. Korrekturen über Seed-Defekte oder
   einen kleinen Doppelannotationsanteil stellen die Gültigkeit nicht her (07 §3.1, B).
3. **Hypergeometrische Schranke unter Überwachung.** v1 S4 kombinierte die exakte Schranke mit einem
   jederzeit stoppbaren Audit. Wird nach jeder von 20 Ziehungsstufen geprüft, entstehen 0,168
   Falschzertifikate (07 F6, B).
4. **Aus dem Audit gelernter Risikoscore.** Satz 2 gilt nur bei eingefrorenem r. Nachgelernt sind
   100 % der Zertifikate falsch, prädiktabel nachgelernt 66 % (07 F2, B).
5. **Fehlende harte Schranke.** Zähl-Recall braucht D_max als unprüfbare Annahme. Ein nie gezogenes
   Bild mit 100 Misses führt zu 0,71 Falschzertifikaten (07 F1, B). „D_max ≈ 1–2 pro Kachel“ war ein
   typischer Wert, keine Schranke; im geclusterten Testpool hatte eine Kachel 5 Defekte.
6. **Mehrere Runden ohne δ-Spending.** Bei wahrem ζ knapp über dem Ziel und Zertifizierung beim ersten
   Erfolg ergeben sich 0,044 / 0,086 / 0,127 / 0,202 Falschzertifikate für R_max = 1 / 2 / 3 / 5. Mit
   δ/R_max pro Runde bleibt der Wert ≤ 0,044 (07 §3.3, B).
7. **Einseitige Ziele sind manipulierbar.** Eine Vollvordergrund-Maske erreicht mit
   `metrics.image_stats` ζ = 0 und Recall = 1 (07 §5.3, B).
8. **Leere Ziele bei seltenen Defekten.** Gilt ζ\* ≥ p für die Prävalenz p, besteht auch ein Labeler
   ohne jede Ausgabe (07 §3.4, B).
9. **Der Floor ist eine Gültigkeitsbedingung, kein bloßer Schutz.** Prop. 3 liefert π_i = 0 bei
   σ_i = 0 (07 F7).

---

## 4. Empfohlener Kern: CertAL-lite

CertAL-lite ist der minimale gültige Satz aus 07 §6. Die Machbarkeits- und die Alternativenlinse des
Red-Teams kamen unabhängig zu ihm, der Gegenprüfer hat ihn bestätigt. Statistisch ist es ein
**klassisches Abnahmeprüfungsdesign mit Korrektur** (Rectifying Inspection nach Dodge–Romig mit
LTPD-Plan) und trägt keine Neuheitsbehauptung. Neu ist nur die Anwendung auf pixelgenaue Auto-Labels
mit korrigiertem Pool. Die Machbarkeitslinse schätzt 3–5 Wochen Entwicklungszeit; das ist ungemessen.

### 4.1 Registrierte Annahmen

| Annahme | Inhalt | Falschzertifikate bei Verletzung (§3.2) |
|---|---|---|
| Protokoll P | blinde Vollsuche (Kacheln nur als UI-Scanhilfe), dann Overlay, Vereinigung; Defektspezifikation pro Klasse; Antwort „unsicher“ erlaubt | verify-first als Messung: 0,135–0,330 |
| Freeze | Labels, r, Konventionen, Ziele, δ-Aufteilung, R_max, n und Seed vor Ziehung 1 gehasht | nachgelerntes r: bis 1,00 |
| Rahmen | ganzer ausgelieferter Pool, R eingeschlossen | R außerhalb: 0,55 |
| Ziehung | uniform ohne Zurücklegen, n vorab registriert, ein registrierter Erweiterungsblick, keine Überwachung | Überwachung: 0,168 |
| Schranken | ζ und φ brauchen keine unprüfbare Schranke; Zähl-Recall bräuchte D_max | ohne D_max: 0,71 |
| Runden | höchstens ein Retraining, Σδ_r ≤ δ | ohne Aufteilung, R_max = 2: 0,086 |
| Keine Rückkopplung | Audit-Befunde ändern die Labels ungeprüfter Bilder nicht | – |

Ein **kleiner Doppelannotationsanteil** (zweite, unabhängige Inspektion unter P) dient nur als
Bias-Monitor. Er liefert eine Punktschätzung der Übersehrate unter P, aber keine Korrektur des
Zertifikats. Bei N = 2000 und 200 Doppelprüfungen ohne Diskrepanz begrenzt er die Reviewer-Misses nur
auf 34 Bilder. Das verbraucht 0,017 eines Ziels von 0,025 (07 §3.1, B).

**Wann Modellfehler nur Effizienz kosten.** In CertAL-lite bestimmt das Modell nur, welche Bilder vor
dem Audit repariert werden, und es liefert die K′-Prognose für die Planung. Das Audit selbst ist
uniform über den eingefrorenen Pool. Modellfehler ändern dann nur die Zahl K′ der Restfehlbilder und
damit die Kosten. Für Ausbaustufen mit modellabhängigem Design (§4.6) gilt dieselbe Aussage nur, sofern
(i) f, λ und q nur eingefroren oder prädiktabel aus dem Audit lernen, bei fixem n nur cross-fitted;
(ii) q einen vom Modell unabhängigen Floor q ≥ γ/N′ hat, im Code erzwungen; (iii) r, Ŷ, U′/R,
Konventionen und Schranken vor dem Audit eingefroren sind und die Schranken nicht aus dem Modell
stammen; (iv) die Messung nicht vom Modell beeinflusst ist (Protokoll P). Eine Kopplung über
Automation Bias am Overlay ist argumentiert, nicht simuliert. Deshalb sucht P zuerst blind.

### 4.2 Ablauf L0–L9

| Schritt | Inhalt |
|---|---|
| **L9 Regime-Weiche** (vor jeder Menschminute) | LCB(p) und UCB(p) der Prävalenz aus einer Triage; Wahl des Zertifikatstyps nach §4.4; bei UCB(p) ≤ ζ\* wird das ζ-Zertifikat als leer markiert |
| **L0 Einfrieren** | vor der ersten Auditziehung: Labels nach Review, r, tol und merge_radius, Ziele (ζ\*, φ\*), δ-Aufteilung, R_max, n, Seed; Manifest mit Hash |
| **L1 Labeler** | bestehende Pipeline (K = 4, Runde 0) plus Korrektur der unsichtbaren Positiven in der Synthese; kein doppeltes Cross-Fitting, solange das Phase-1-Gate es nicht verlangt (§6) |
| **L2 Kontrolldetektor** | ein Detektor, z. B. anomalib 2.7.0 AnomalyDINO (DINOv2-S, Apache-2.0 laut METADATA), nur als Risiko-Merkmal, nie als Labeler-Input |
| **L3 Risikoscore** | `review_risk`-Rangmittel oder logistischer Stacker auf L-OOF; keine Vollständigkeitsschätzer |
| **L4 Reparatur** | verify-first in Risikoreihenfolge, Batches zu 25; Stopp, sobald der Ertrag an Fehlbildern pro Batch unter die Schwelle aus §4.5 fällt; Korrekturen zählen nicht als gemessen |
| **L5 Zertifizierung** | uniformes Audit ohne Zurücklegen aus dem ganzen Pool, R eingeschlossen (optional als eigenes Stratum); jede Einheit unter P; n aus der exakten Powertabelle mit konservativer K′-Prognose; ein registrierter Erweiterungsblick mit δ-Aufteilung; frühes Stoppen nur zum Nicht-Zertifizieren; Zertifikat für ζ_final und φ_final (§4.3) |
| **L6 Abgeleitete Schranke** | ρ_b ≥ 1 − Ū/(A + Ū), ohne zusätzliches δ und ohne D_max (§4.3) |
| **L7 Bericht** | Zertifikat mit Annahmeliste (Design, P, Freeze-Hash); Dice und Recall nach Größenklassen als *approximativ* gekennzeichnet; Begriff „statistisches Pool-Zertifikat (designbasiert)“ |
| **L8 Runden** | höchstens ein Retraining mit Σδ_r ≤ δ; das zweite Audit zieht aus U″ und behält die geprüften Einheiten als bekanntes Stratum |

Für R gibt es drei gültige Wege (07 §3.2, geprüft): R im Auditrahmen mitziehen (kostet +20 % bei
sauberem R bis +70 % bei s = 0,7), R vollständig unter P inspizieren, oder R zufällig unter P
nachprüfen mit eigener δ-Aufteilung. Bei |R| = 150 begrenzen ≈ 43 Nachprüfungen ohne Befund den Rest
in R auf ≤ 10 Einheiten bei δ′ = 0,025.

### 4.3 Zielgrößen und Schranken

Alle Zielgrößen beziehen sich auf den **ausgelieferten** Pool U mit N Bildern. Ein Bild ist ein
Fehlbild (b_i^final = 1), wenn es einen unter P auffindbaren Defekt enthält, den das finale Label
nicht trifft (Toleranz 2 px, any-hit). F_i^final ist die Zahl falscher Komponenten im finalen Label.

* ζ_final = (1/N) Σ_{i∈U} b_i^final, die Summe läuft über den ganzen Pool einschließlich R;
* φ_final = (1/N) Σ_{i∈U} F_i^final, mit der bekannten Schranke F_i ≤ P_i aus dem eingefrorenen Ŷ;
* ρ_b = Anteil der defekten Bilder ohne Restmiss (Bild-Recall);
* ρ_final = 1 − Σ_{i∈U} M_i^final / Σ_{i∈U} D_i (Zähl-Recall), wobei M_i^final die Restmisses
  M_R^res im Review-Stratum einschließt. Nicht Teil des Kerns.

**Exakte Schranke bei fixem n.** Vor dem Audit gibt es K Fehlbilder. Bei uniformer Ziehung von n aus
N Bildern ohne Zurücklegen ist die Zahl X gefundener Fehlbilder hypergeometrisch verteilt. Mit
UCB_hyp(x) = max{K : P_K(X ≤ x) > δ} gilt, weil die gefundenen x Fehlbilder korrigiert werden:

ζ_final ≤ (UCB_hyp(x) − x)/N mit Wahrscheinlichkeit ≥ 1 − δ.

Diese Regel ist identisch mit einem **LTPD/LQ-Plan mit Konsumentenrisiko δ**: In 620 geprüften
(n, k)-Zellen weicht keine ab (07 M3, B). Die richtige Formulierung ist LTPD/LQ, nicht AQL, denn AQL
beschreibt ein Produzentenrisiko. Ohne einen einzigen Fund braucht ζ\* = 0,02 bei N = 2000 n = 140
Inspektionen, ζ\* = 0,002 schon n = 901 (hier nachgerechnet mit `certify.plan_n(2000, 0, 4, 0.05)`; 07 nennt
den Wert nicht). Die Schranke gilt nur bei vorab registriertem
fixem n und uniformer Ziehung. Ein jederzeit stoppbares Audit braucht die PPR-Martingale ohne
Zurücklegen (Waudby-Smith & Ramdas 2020) auf einem separat gelosten uniformen Strom (Go-Kriterium 2 in
§4.6), ein nicht-uniformes q braucht Satz 1′ aus 07 F1.

**Zweiseitigkeit über φ.** Eine Vollvordergrund-Maske besteht jedes ζ-Ziel, erzeugt aber auf jedem
sauberen Bild eine Falschkomponente. φ (oder alternativ die Pixel- bzw. Toleranzpräzision) wird daher
mit eigenem δ-Anteil zertifiziert, z. B. δ/2 je Zielgröße. Für φ ist die Schranke F_i ≤ P_i bekannt,
eine unprüfbare Zählschranke entfällt.

**Abgeleiteter Bild-Recall.** Sei Ū die obere Schranke der Restfehlbilder aus dem Audit und A die
Zahl der unter P geprüften und dabei als defekt bestätigten Bilder. Die A Bilder und die Menge B der
Restfehlbilder sind disjunkt, also gilt #defekt ≥ A + |B|. Weil x ↦ x/(A + x) monoton steigt, folgt
auf dem Ereignis |B| ≤ Ū

ρ_b = 1 − |B|/#defekt ≥ 1 − Ū/(A + Ū).

Beispiel: A = 150, Ū = 40 ergibt ρ_b ≥ 0,79. Die Schranke kostet kein zusätzliches δ und braucht kein
D_max. Sie gilt nur, wenn die A Bilder unter P geprüft wurden; sonst können sie zugleich in B liegen
(07 L6, Algebra N; hier nachvollzogen).

**Zähl-Recall.** Ohne harte Schranke gibt es keine nichttriviale endliche LCB: Für jede gültige LCB
gilt auf jeder Population P(LCB > ε) ≤ δ·N/(N − n) (07 F1, B). D_max ist *typisch* 1–2 pro Kachel;
pro Bild liegt es höher (die S1-Pools von 04 kappen bei D_max = 6 pro Bild; in den Testpools von
`sims/redteam2_lens_lite_recall.py` liegt das Maximum bei 4–5 pro Bild und 2–3 pro Kachel). Als harte
Schranke muss D_max vorab begründet werden, sonst ist ein Recall-Zertifikat modellbasiert. Zulässig sind ein so beschrifteter, gekappter Recall mit min(M_i, D_max) und eine designbasierte UCB auf
den Anteil der Einheiten mit D_i > D_max. Ein Flächenmaß wäre gültig, aber praktisch leer. Der
bildgewichtete Recall 1 − Σ(M_i/D_i)/#defekte Bilder braucht kein D_max, ist aber eine andere, gegen
Pitting blinde Zielgröße und darf nur unter eigenem Namen erscheinen (07 §6, U).

### 4.4 Regime-Weiche, Heuhaufen, Mehrklassen und unbekannte Typen

**Heuhaufen.** Können Misses in Bildern ohne jedes Signal liegen, muss jedes Verfahren, das Recall
≥ 1 − ε zertifiziert, das saubere Stratum effektiv uniform prüfen. Im günstigsten Fall (ein Miss pro
Einheit) braucht das n_c ≈ N_c·ln(1/δ)/(ε·D_tot) Inspektionen. Bei 2000 sauberen Bildern und ≤ 7
Fehlbildern sind das exakt **624** (v1 nannte 640; 07 F10). Im gepackten schlechtesten Fall liegen
k = εD_tot + 1 Misses in ⌈k/D_max⌉ Einheiten, und der nötige Anteil steigt auf 1 − δ^{1/⌈k/D_max⌉}.
Für k = 10 sind das 0,259 / 0,451 / 0,776 bei D_max = 1 / 2 / 6. Allgemein gilt
n_c ≈ N_c·ln(1/δ)·D_max/(ε·D_tot).

Anteil der sauberen Einheiten, der für Recall ≥ 0,9 zu inspizieren ist (07 §3.4, B):

| Prävalenz p | 0,5 % | 1 % | 3 % | 10 % | 30 % |
|---|---|---|---|---|---|
| ein Miss pro Einheit | 78 % | 53 % | 26 % | 9 % | 3 % |
| gepackt, D_max = 2 | 78 % | 78 % | 45 % | 16 % | – |
| gepackt, D_max = 6 | 95 % | 95 % | 78 % | 39 % | – |
| ρ_b, das ζ ≤ 0,02 impliziert | 0 | 0 | ≥ 0,33 | ≥ 0,80 | ≥ 0,93 |

Die Benchmarks MT (28 % defekt), BSData (≈ 36 %) und KSDD2 (≈ 11 %) liegen im machbaren Bereich. Der
eigentliche Zielfall mit seltenen Defekten liegt es nicht.

**Regime-Weiche (L9).** Vor jeder Menschminute werden LCB(p) und UCB(p) aus einer Triage geschätzt,
etwa aus einer kleinen uniformen Vorstichprobe mit Ja/Nein-Urteil, die vor dem Freeze liegt
(Vorschlag, ungeprüft). Bei hoher Prävalenz gilt ζ + φ + abgeleitetes ρ_b, bei mittlerer eine
stratifizierte Aussage („Missrate ≤ UCB im signalfreien Stratum“), bei niedriger Vollinspektion mit
modellgestützter Beschleunigung. Gilt UCB(p) ≤ ζ\*, wird das ζ-Zertifikat als leer markiert. Die
Grenzen zwischen den Regimen sind aus dem Ziel abzuleiten, etwa so, dass ζ\* ein ρ_b ≥ 0,8 impliziert.
`certify.regime_check` setzt vorläufig die Grenzen 0,8 („hoch“, wenn 1 − ζ\*/LCB(p) ≥ 0,8) und 0,5
(„niedrig“, wenn 1 − ζ\*/UCB(p) < 0,5). Das sind Konventionen dieser Implementierung, nicht abgeleitet
und nicht validiert; begründete Werte stehen noch aus.

**Mehrklassen.** Das primäre Zertifikat ist **klassenagnostisches ζ bei vollem δ**. Danach werden
Klassen in fester Reihenfolge getestet (Fixed-Sequence). Die Bonferroni-Strafe für klassenweise ζ_c
wäre moderat: Nullfund-n = 140 / 171 / 219 / 241 / 270 für J = 1 / 2 / 6 / 10 / 20 Klassen bei
ζ\* = 0,02 (07 §3.5). Klassen-Recall ist durch die Heuhaufen-Grenze pro Klasse beschränkt. Eine
Zählregel für Klassenverwechslungen fehlt.

**Unbekannte Defekttypen.** Bei 90 uniformen Ziehungen wird ein Typ mit 1 % Prävalenz mit
Wahrscheinlichkeit 0,60 getroffen, einer mit 0,3 % mit 0,24. Ein Typ mit 1,5 % besteht ζ ≤ 2 % und wird
nie repariert (07 §3.5, N). Der Bericht nennt deshalb eine Rule-of-Three-Schranke: Ein in n uniformen
Ziehungen nie gesehener Typ hat mit 95 % eine Prävalenz ≤ ≈ 3/n (bei n = 140 also ≤ 2,1 %). Die Suche
nach neuen Typen bekommt ein eigenes Entdeckungsbudget mit bekannten Inklusionswahrscheinlichkeiten.

### 4.5 Woher die Effizienz kommt

Die entscheidende Größe ist **K′(k)**, die Zahl der Restfehlbilder, nachdem die k riskantesten Bilder
repariert wurden. Für ζ gibt es eine geschlossene Kostenform (07 §5.3, N):

Kosten(k) = k·c_review + n₉₀(K′(k))·c_audit.

Dabei ist n₉₀(K′) das kleinste fixe n, das mit Wahrscheinlichkeit ≥ 0,9 zertifiziert (exakt
hypergeometrisch, Korrekturen angerechnet). K′(k) lässt sich aus einem GT-Lauf ablesen. Für ζ braucht
man dann keine sequentielle Simulation.

Exakte Planungstabelle für N = 2000 und δ = 0,05, mit dem ganzen Pool als Rahmen, wie CertAL-lite ihn
verlangt. Reproduzierbar mit `certify.n90_table` bzw. `certify.plan_n` (N = 2000, δ = 0,05, Power 0,9,
Korrekturen angerechnet); die Tabelle zeigt die exakten Werte auf 10 aufgerundet (exakt: 140 / 215 /
215 / 332 / 428 / 549 / 649 / 735 bzw. 265 / 395 / 497 / 583 / 783 / 928). Das Skript
`sims/redteam2_lens_n90_table.py` nutzt den Rahmen N′ = 1850 und liefert ≈ 7–9 % kleinere Werte
(130 / 310 / 680 bei K′ = 0 / 10 / 30, ζ\* = 0,02).

| Ziel | n₉₀ bei K′ Restfehlbildern |
|---|---|
| ζ\* = 0,02 | K′ = 0: 140 · 2: 220 · 5: 220 · 10: 340 · 15: 430 · 20: 550 · 25: 650 · 30: 740 |
| ζ\* = 0,01 | K′ = 0: 270 · 1: 400 · 3: 500 · 5: 590 · 10: 790 · 15: 930 |

Daraus folgt der **Grenzwert eines Fehlbilds**: Jedes Restfehlbild, das vor dem Audit entfernt wird,
spart ≈ 18 Audit-Inspektionen bei ζ\* = 0,02 und ≈ 45 bei ζ\* = 0,01 (07 §5.2, B; die mittlere Steigung
der Tabelle liegt bei 20 bzw. 44, die Kurve ist treppenförmig). Reviewen lohnt, solange der Ertrag y an
Fehlbildern pro geprüftem Bild

y > (c_review/c_audit) / (dn₉₀/dK′)

erfüllt. Bei gleichen Kosten pro Einheit sind das ≈ 5,5 % bei ζ\* = 0,02 und ≈ 2–2,5 % bei 0,01. Ist
eine Inspektion unter P teurer als eine Review-Ansicht, sinkt die Schwelle entsprechend.

Weitere Befunde aus 07 §5.2:

* **Uniform ist minimax-optimal bei null Funden** (B). Ein Design, das bei null Funden zertifiziert,
  braucht E[n] ≥ N′(1 − δ^{1/(K\*+1)}), mit Gleichheit nur bei uniformem π. Für N′ = 1850 und K\* = 40
  sind das 129. Eine Proposal-Verteilung kann die Nullfund-Stichprobe also nicht verkleinern; Hebel
  (i) aus 04 N5 entfällt.
* **PPS hilft für ζ nicht** (B mit Korrektur). Bei wahrem ζ = 0,005 und Ziel 0,02 brauchte uniform im
  Median 314 verschiedene Einheiten, PPS (γ = 0,3) 554–680. Nahe am Ziel liegt PPS mit AUC 0,9
  gleichauf. PPI mit λ = 1 ändert ≤ 5 Prozentpunkte.
* **Volle S4 gegen CertAL-lite** (B mit Korrektur). Median 810 (q90 1014) gegen 650 Inspektionen bei
  ζ\* = 0,02, also 12–25 % Mehrkosten für 0–3 % Gewinn.
* **Jederzeitige Gültigkeit kostet** (U). Bei K = 10 von 2000 braucht fixes n 355 für 90 % Power,
  wenn die Korrekturen nicht angerechnet werden (mit Anrechnung 332, `certify.plan_n`), PPR mit Prior
  im q90 550. Der Nutzen ist Robustheit gegen eine falsche Prognose: Ein fixes n = 332, geplant für
  K′ = 10, hat bei K′ = 10 / 20 / 30 Power 0,93 / 0,57 / 0,24 (N = 2000, ζ\* = 0,02,
  `certify.certification_power`). Die in 07 §5.2 genannten 0,93 / 0,56 / 0,23 für n = 310 gelten im
  Rahmen N′ = 1850; im ganzen Pool (N = 2000) hätte n = 310 nur 0,81 / 0,38 / 0,13.

**Konsequenz.** Forschungsaufwand gehört in die Senkung von K′(k) pro Review-Minute:

1. ein **besserer Labeler**, der weniger Fehlbilder erzeugt (Encoder nach B1, korrigierte Synthese);
2. eine **bessere Risikoreihenfolge**, die Fehlbilder früh in die Queue bringt (Kontrolldetektor,
   Stacker);
3. **unabhängige Signale** für Misses, die heute kein Signal tragen (Sequenzkonsistenz, zweite
   Ansicht, zweiter Detektor). Ohne Signal sind solche Misses nicht rankbar;
4. **billigere Review-Aktionen** (verify-first-Oberfläche, Kacheln als Scanhilfe).

Komponenten der früheren Schicht S3 werden deshalb nach Misses pro Review-Minute bewertet, nicht nach
dem Fehler eines Vollständigkeitsschätzers (07 §6).

### 4.6 Ausbaustufen nur hinter Go-Kriterien

Die Schichten S1–S5 aus v1 bleiben als Forschungsspur; ihre Inhalte stehen in 01–06 und in der
Versionsgeschichte dieser Datei. Eine Komponente kommt erst in den Produktpfad, wenn sie ihr
Go-Kriterium auf GT-Datensätzen erfüllt, gemessen über die geschlossene Kostenform oder die
Nordstern-Simulation bei gleicher Gültigkeit (Falschzertifikatsrate ≤ δ). Alle Kriterien sind
Vorschläge aus 07 §6 (N).

| # | Komponente | Go-Kriterium |
|---|---|---|
| 1 | stärkerer Encoder (Labeler und Kontrolldetektor) | K′(k\*) oder n₉₀ sinkt um ≥ 10 % |
| 2 | jederzeit stoppbares Audit via PPR ohne Zurücklegen, Prior auf die Prognose zentriert | Median-n ≤ 0,9 × fixes n₉₀ bei Falschzertifikatsrate ≤ δ am Rand |
| 3 | Synthese v2 über die Fehlerkorrektur hinaus (C6) | OOF-Recall pro Größenklasse steigt, K′ sinkt |
| 4 | unabhängige Zusatzsignale (Sequenzkonsistenz, zweite Ansicht, zweiter Detektor) | Anteil unsichtbarer Misses sinkt |
| 5 | Self-Training Runde 1, danach doppeltes Cross-Fitting (C4) | K′(k) sinkt um ≥ 10 % |
| 6 | Recall-Zertifikat auf Defektebene | nur wenn vertraglich nötig, mit registrierter harter D_max |
| 7 | PPS/PPI und Prop. 3 im Audit | n₉₀ sinkt um ≥ 15 % bei gleicher Gültigkeit, für Recall getrennt geprüft |
| 8 | Injection-Härtekarte als Allokations- bzw. Risikokovariate | Top-20-%-Härte-Bilder enthalten ≥ 50 % der von allen Detektoren übersehenen echten Misses |
| 9 | Kacheln als Stichprobeneinheit | gemessene Kachel-Prüfzeit pro Fläche ≥ 30 % niedriger **und** gepoolter Recall pro Größenklasse gefordert |
| 10 | Satz 2 (zweiphasig) und Vollständigkeitsschätzer aus 06 | Forschungsspur, nie im Zertifikat |

Hinweise zu einzelnen Stufen:

* **Prop. 3, PPS, PPI (frühere S4).** Die geschlossene Form von Prop. 3 ist korrekt, doch ein
  abgestimmtes „Top-k + Neyman-Audit“ erreicht ihre Zielfunktion bis auf ≤ 1,2 % (07 C2). Der Floor
  q ≥ γ/N′ ist Gültigkeitsbedingung, die Wettobergrenze muss prädiktabel sein (07 F4).
* **Satz 2 (designbasierte Schwellenwahl).** Gültig nur zweiphasig (r auf den Ziehungen 1..t₀ lernen,
  einfrieren, S_{t₀} ins bekannte Stratum verschieben, frisch aus U″ ziehen; 0,000 Falschzertifikate
  bei Power 0,577) oder mit eingefrorenem r und der Korrekturidentität (0,001). **Der Stacker darf für
  f und q nachkalibriert werden, r für Satz 2 bleibt eingefroren.** Erlaubt sind nur streng monotone
  Transformationen von r (07 F2, B).
* **Kacheln.** Das Bild bleibt Stichprobeneinheit; die Kachel ist eine Scanhilfe der Oberfläche unter
  P und statistisch ohne Rolle. Als Stichprobeneinheit wären Kacheln nur moderat schärfer (Faktor
  1,1–2,5 in der nötigen Fläche) und das nur mit harter D_max pro Kachel. Bei 3 s Overhead pro Kachel
  verlieren sie gegen Bilder, gegen die Heuhaufen-Grenze helfen sie nicht (07 F11, C5). Lemma 0
  versagt auf Kachelebene: Eine lokale Korrektur kann eine Falschkomponente der Nachbarkachel teilen
  (F_{T′} von 1 auf 2) oder einer GT-Komponente ihren einzigen Toleranztreffer nehmen (M_{T′} von 0
  auf 1). Eine Heimatregel müsste auf der GT definiert sein, etwa über das argmax der eingefrorenen
  Wahrscheinlichkeitskarte (07 F12, N).
* **Vollständigkeitsschätzer (06, C1).** N̂_HT, Huggins, Hybrid, Triangulation und Ausweichregel
  entfallen aus Bericht und Zertifikat: T0 ist untestbar, Übereinstimmung ist kein Beleg (0,89 bei
  −30 % Bias), die Ausweichregel kann den Bias auf +0,83 treiben (07 F3, F9). Es bleiben zwei Rollen,
  in denen Modellfehler nur Power kosten: die K′-Prognose für die Auditplanung bzw. als PPR-Prior und
  die Injection-Recovery-Karte als Risiko-Merkmal hinter Go-Kriterium 8.
* **Doppeltes Cross-Fitting (C4).** Nur wenn das Phase-1-Gate zeigt, dass Self-Training den U-Recall
  in den Niedrigrisikostrata senkt. Es ist nicht strikt leckfrei, weil Lehrer j U_k als
  Negativ-Hintergrund gesehen hat. Strikt wird es erst mit K(K−1)/2 paarweise ausschließenden Lehrern
  (≈ +83 min CPU, Schätzung).
* **Synthese.** Die Korrektur der unsichtbaren Positiven ist Pflicht (L1). Null-Negative,
  Detektierbarkeits-Matching, kontextpassende Platzierung und multiplikativer Transfer stehen hinter
  Go-Kriterium 3. Die Ablation braucht einen NSA-Arm.
* **Aufgabenanpassung (S5)** wird lokal nur verkleinert geprüft (§6, Phase 5).

### 4.7 Stand der Umsetzung

* `autolabel/certify.py` existiert (Teil des CertAL-lite-Kerns, CLI-Befehle `certify` und
  `certify-plan`). Über seine Unit-Tests (`tests/test_certify.py`) hinaus gibt es dazu keine
  Ergebnisse; auf echten Daten ist es nicht gelaufen.
* **Umgesetzt:** exakte hypergeometrische Schranke bei fixem n (`hypergeom_ucb`), ζ-Zertifikat mit
  Ablehnung bei abweichendem n (`certify_zeta`), Powerplanung (`plan_n`, `n90_table`,
  `certification_power`), PPR-Konfidenzfolge ohne Zurücklegen (`ppr_wor_ucb`), abgeleiteter Bild-Recall
  (`derived_recall_bound`), Regime-Weiche mit Leerheitsflag (`regime_check`, vorläufige Grenzen
  0,8 / 0,5, siehe §4.4), Rule of Three, Freeze-Manifest mit Hash und `AuditPlan` (Standard R_max = 2,
  δ_r = δ/2), Floor q_i ≥ γ/N (`floored_inclusion`, `check_floor`).
* Grundlage ist `scipy.stats.hypergeom`. Die PPR-Martingale (Waudby-Smith & Ramdas 2020) ist in
  `certify.py` selbst implementiert; `confseq` wird nicht genutzt. `ppi_py` ist nur CLT-basiert und
  damit **nicht** für Zertifikate geeignet.
* **Nicht umgesetzt:** ein φ-Zertifikat und damit die Zweiseitigkeit aus L5. Für φ ist noch kein
  Verfahren festgelegt. Die hypergeometrische Schranke passt nicht, weil φ ein beschränktes Mittel
  und kein Anteil ist; in Frage kommt eine Schranke für beschränkte Mittel ohne Zurücklegen (etwa
  Betting auf F_i/P_max mit vorab eingefrorenem P_max), ungeprüft. Ebenso fehlen: R als eigenes
  Stratum mit kombinierter Schranke, eine registrierte Erweiterung des Audits (`certify_zeta`
  akzeptiert genau ein registriertes n), ein nicht-uniformes Audit selbst und die adversarialen Sims
  „R außerhalb des Rahmens“, Mehrrunden und verify-first. Adversarial getestet ist nur die Überwachung
  der fixen Schranke gegen PPR (in `tests/test_certify.py`) und die Wettobergrenze der Sims
  (`tests/test_sims_adversarial.py`).
* Alle übrigen Teile von §4 (Kontrolldetektor, ertragsbasierter Review-Stopp, Protokoll-P-Oberfläche,
  Triage für LCB(p)/UCB(p)) sind weder implementiert noch getestet.

---

## 5. Nordstern, These H\* und Neuheit

### 5.1 Nordstern-Metrik

Die Nordstern-Metrik ist ein **Paar**:

1. **Menschminuten bis zu einem gültigen Zertifikat**, und
2. die **Falschzertifikatsrate gegen die GT**, Sollwert ≤ δ, auch unter einem simulierten fehlbaren
   Reviewer mit Sensitivität s ∈ {1; 0,9; 0,7}. s hängt in der Simulation von n, Größe und Kontrast
   ab und korreliert positiv mit der Modellschwierigkeit.

Bei s < 1 garantiert kein Design die zweite Komponente gegen die Wahrheit, weil das Zertifikat nur
relativ zu P gilt. Der Bericht zeigt deshalb, ob und ab welchem n die Rate δ überschreitet. Eine
Politik, die Minuten spart, indem sie verify-first als Messung nutzt, fällt so sofort auf.

Pro Aufgabeninstanz werden vorab registriert: ein **zweiseitiges** Ziel (ζ\* und φ\* oder ein Ziel für
Pixel- bzw. Toleranzpräzision), δ, die Politik und ein Budget B. Berichtet wird eine **Verteilung**,
keine Einzelzahl:

1. die Erfolgskurve P(Zertifikat nach ≤ B Minuten);
2. Median und 90-%-Quantil der Minuten, gegeben Zertifizierung;
3. die Falschzertifikatsrate wie oben;
4. die wahre GT-Qualität des ausgelieferten Pools;
5. Fixkosten für L und Rechenzeit, getrennt ausgewiesen;
6. die Futility-Rate.

**Kostenmodell.** Die Simulation speichert kostenfreie Aktionszählungen n_a: Verify-Ansichten,
Vollinspektionen pro Einheitentyp, korrigierte Komponenten, Grenzkorrekturen und Ziehungen. Minuten
ergeben sich als Σ_a c_a·n_a, ausgewertet über einen Latin-Hypercube auf diesen Spannen (07 §5.3):

| Aktion | Spanne |
|---|---|
| Verify | 2,7–6 s |
| Flag + Klick | + 3–5 s |
| Korrektur | 10–120 s pro Komponente |
| Polygon / Riss | 60–300 s |
| **Vollinspektion** | 10–120 s pro Bild (Annahme ohne Anker; die 2,72 s aus 05 §3.1 sind ImageNet-Ein-Bit-Verifikation) |
| Kachel-Overhead c₀ | 1–6 s |

Berichtet wird, in welchem Bereich des Kostenraums H\* gilt, über komponentenweise Dominanz der
Zählungen oder Break-even-Verhältnisse (c_inspect/c_verify, c₀, c_review/c_audit). Parameter von
Ausbaustufen (etwa γ, κ) werden vorab fixiert und nie auf den H\*-Instanzen getunt. **Vor Phase 4**
steht eine einstündige Selbst-Zeitmessung: je 50 MT- und KSDD2-Einheiten mit Verify, Vollinspektion
und Korrektur.

**Einordnung.** Das Paar ist eine kostengewichtete ATI (Average Total Inspection nach Dodge–Romig)
bzw. ASN (Average Sample Number sequentieller Tests), ergänzt um die Gültigkeitskomponente. Verwandt
sind „budget save“ bei PAC-Labels, TAR-Gesamtkosten bei zertifiziertem Recall und WSS@95. Eine neue
Metrik ist es nicht.

Mit GT als Orakel bleibt in der Simulation nur der Menschfehler-Pfad unsichtbar. Designfehler wie
Optional Stopping, fehlende D_max, Rückkopplung, Mehrrunden-δ und gelerntes r können und müssen dort
scheitern. Die Simulationssuite enthält sie deshalb als adversariale Szenarien.

### 5.2 These H\* und Baselines

> **H\***: Bei gleichem zweiseitigem Ziel und gleicher Gültigkeit braucht CertAL-lite weniger
> Menschminuten bis zum Zertifikat als die klassischen Verfahren (b) und (c) und höchstens so viele
> wie (a-stark) und (e). Jede Ausbaustufe aus §4.6 muss CertAL-lite um ihr Go-Kriterium schlagen. Die
> Falschzertifikatsrate bleibt bei s = 1 in allen Szenarien ≤ δ, auch wenn Modellannahmen bewusst
> verletzt werden.

| Baseline | Definition |
|---|---|
| (a-naiv) | Ablauf nach 04 §1 (Stand des Codes): bestes Modell + Top-k-Review + separates Zufallsaudit nur aus U′, Korrekturen nicht angerechnet; exakte Schranke statt PPI-Intervall |
| (a-stark) | Top-k-Review, danach SRSWOR aus dem Pool nach dem Review, exakte Schranke, angerechnete Korrekturen, Planung ohne Orakel |
| (b) | bestes Modell + Vollkontrolle aller Bilder unter P |
| (c) | Dodge–Romig-LTPD- bzw. ISO-2859-2-Plan mit Rectifying (Stichprobe, bei Ablehnung Vollprüfung), ohne Modellreihenfolge |
| (d) | CertAL-lite (§4) |
| (e) | TAR-Target-Methode (Recall-Zertifizierung über eine zufällige Zielmenge, Lewis, Yang, Frieder 2021) |

Alle Baselines bekommen dieselben exakten Schranken. (a-naiv) ist bei s < 1 nicht gültig, weil R
außerhalb des Rahmens liegt; die Simulation soll genau das sichtbar machen. Der Abstand zwischen
(a-stark) und (d) misst den Wert von Kontrolldetektor, ertragsbasiertem Stopp, R im Rahmen und
Regime-Weiche, der Abstand zwischen (c) und (d) den Wert des Modells überhaupt.

### 5.3 Neuheit: ehrliche Einordnung

Grundlage ist 07 §4. Volltextquellen (arXiv, OpenReview, CVF) waren dort nicht abrufbar; die Evidenz
stammt aus Abstracts, Snippets und gelesenem Code. Alle sieben Urteile hat der Gegenprüfer bestätigt.

| # | Beitrag (v1) | Urteil | Nächstliegende Vorarbeit (Auswahl) | Ehrliche Formulierung |
|---|---|---|---|---|
| C1 | Vollständigkeitsschätzung | teilweise bekannt | Mills-Seeding (1972), Kepler-DR25-Injection, Balrog, DQM (Chung et al. 2017), Chao-Stopp in TAR, Kansal 2026 (Capture–Recapture über Labelfehlerdetektoren), Rottmann & Reese 2023 (fehlende Labels in Segmentierung; nicht nur FP wie MetaSeg) | Übertragung etablierter Vollständigkeitsmethoden auf Pixel-Auto-Labels, gekoppelt an ein Audit; der Transporttest ist ein Anpassungstest auf beobachtbaren Zellen, blind für Defekte, die kein Detektor sieht |
| C2 | Measure-and-Repair | teilweise bekannt (Kern bekannt) | PAC-Labels (Candès, Ilyas, Zrnic, ICML 2026; korrigiert gezogene Punkte und zertifiziert den korrigierten Datensatz, nicht nur Klassifikation), Dodge–Romig, Ijiri & Kaplan 1971, TAR-Zertifizierung | geschlossene KKT-Form der Inklusionswahrscheinlichkeiten für ein bekanntes Prinzip; Lemma 0 ist kein Beitrag |
| C3 | Segmentierungs-PAC-Labels | teilweise bekannt | PAC-Labels/LTT, SHANGRLA-Assorter (Stark), IPW-Testmartingale mit Kontrollvariaten (Shekhar et al. 2023), Webber 2012 zur Unterdeckung | Anwendung von PAC-Labeling bzw. TAR-Recall-Zertifizierung auf geclusterte Segmentierungseinheiten mit bekannter Linearisierung; „löst das Unterdeckungsproblem“ entfällt |
| C4 | Doppeltes Cross-Fitting | teilweise bekannt (Ingenieursbeitrag) | CV+/Jackknife+, Cross-Prediction-PPI, INCV, Confident Learning | konsequente Out-of-Fold-Disziplin über L ∪ U; Schwellentransfer setzt L~U-Austauschbarkeit voraus |
| C5 | Kacheln als Einheit | teilweise bekannt | Stark 2010 (kleine Audit-Batches), Kartenvalidierung, Proportionator, MetaAT | Übertrag eines bekannten Prinzips; moderat schärfer (1,1–2,5), nur mit harter D_max; „um Größenordnungen“ entfällt |
| C6 | Synthese-Labelvertrag | teilweise bekannt (Kern bekannt) | NSA (Schlüter et al., ECCV 2022; Label = geschwellte Intensitätsdifferenz), FPI/PII, Dwibedi et al. 2017, Dvornik et al. 2018 | Verallgemeinerung signalbasierter Labels auf alle Generatoren, plus Ignore-Band, Verwerfungsregel und Detektierbarkeits-Matching |
| C7 | Nordstern-Metrik | teilweise bekannt | PAC-Labels „budget save“, TAR-Gesamtkosten, WSS@95, Dodge–Romig-ATI, ASN | kostengewichtete ATI/ASN bis zu einem *gültigen* Zertifikat, als Paar mit der Falschzertifikatsrate |

**Was an Neuheit übrig bleibt:**

* die Übertragung auf pixelgenaue industrielle Auto-Labels mit korrigiertem Pool;
* die geschlossene KKT-Form von Prop. 3, korrekt, aber mit geringem Zusatznutzen;
* das Ignore-Band und die Verwerfungsregel im Synthese-Labelvertrag;
* das Detektierbarkeits-Matching am realen Normalitäts-Score;
* möglicherweise der GT-freie Vergleich vorhergesagter beobachtbarer Zellanteile, als
  Anpassungstest, nicht als Vollständigkeitstest.

**Übersehene Nachbarfelder** (07 §4, alle B):

* **Kartenvalidierung** in der Fernerkundung (Olofsson et al., RSE 2014; Stehman 2009): die nächste
  Vorarbeit für designbasierte Bewertung pixelgenauer Labels. Unterschiede: Flächen statt Objekte,
  CLT-Intervalle, keine Korrektur.
* **Technology-Assisted Review** (E-Discovery, Systematic Reviews): isomorphes Problem (Dokument ↔
  Bild, relevant ↔ defekt, Review-Cutoff ↔ Risikoschwelle). Die Heuhaufen-Schranke ist dort bekannt.
* **Akzeptanzprüfung** (ISO 2859-1 und -2, Dodge–Romig; für Annotationen Klie et al., ACL 2024): Das
  ζ-Zertifikat bei fixem n ist ein LTPD-Plan. ATI-optimale Rectifying-Pläne existieren.
* **Risk-Limiting Audits** (SHANGRLA/ALPHA; Shekhar et al. 2023 für Finanzaudits): gewichtete
  Ziehung ohne Zurücklegen mit adaptiven Kontrollvariaten.
* **Dual-System-Schätzung** im Zensus (Bell, JASA 1993; Wachter & Freedman 2000; `drpop` 2024): Der
  Hybrid aus 06 ist Bells Schätzer mit fixiertem Odds Ratio.
* **Probability of Detection** (MIL-HDBK-1823A, MAPOD): parametrische Form der Entdeckbarkeit; a90/95
  aus Injektionen ist modellbasiert, kein Zertifikat.
* Ferner: Proportionator (PPS über Kacheln), label-effiziente Evaluation (nur asymptotisch gültig),
  Labelfehlerdetektion für fehlende Labels (Rottmann & Reese 2023), Gold-/Honeypot-Fragen und die
  Begriffskollision mit ISO/IEC 5259-4:2024.

---

## 6. Roadmap v2

Reihenfolge nach Nutzen pro Aufwand (07 §5.2). Phase 1 kommt vor jeder weiteren CertAL-Komponente
(07 §7 Nr. 16). Ausnahme ist nur der bereits vorhandene Statistikkern aus Phase 0
(`autolabel/certify.py`, CPU-Minuten); seine offenen Teile (§4.7) werden neben Phase 1 ergänzt, nicht
vor ihr. Nichts über CertAL-lite hinaus wird gebaut, bevor Phase 1 abgeschlossen ist. Die
Rechenzeiten sind auf dem M4 **zu messen**; angegeben sind Referenzwerte der Remote-CPU (≈ 75 min pro
MT-Lauf) bzw. Schätzungen aus 07 §5.1. Die Entwicklungstage sind eigene, unüberprüfbare Schätzungen,
abgestimmt auf 07 (gesamt 11–18 Wochen, CertAL-lite 3–5 Wochen).

| Phase | Inhalt | Rechenzeit (auf M4 zu messen) | Entwicklungstage (Schätzung) | Exit-Kriterium |
|---|---|---|---|---|
| **0 – CertAL-lite-Statistik** (teilweise umgesetzt, §4.7) | `autolabel/certify.py`: exakte Schranke bei fixem n, Powertabelle n₉₀(K′), LTPD-Formulierung, φ (Verfahren offen), ρ_b, δ-Aufteilung über Runden, Freeze-Manifest mit Hash, Leerheitsflag; adversariale Simulationen (Überwachung, R außerhalb, Mehrrunden, verify-first); Korrektur der alten Sims (F4, F16) | CPU, Minuten | 5–10 | Falschzertifikatsrate ≤ δ am Rand in allen adversarialen Szenarien; Unit-Tests grün |
| **1 – Baseline-Benchmarks** | HANDOFF-Plan (MT, BSData, DeepCrack, CFD), dazu Loader für KSDD2 und MVTec AD 2; Metriken nach Größenklassen, SegF1, ζ, φ; K′(k)-Kurven aus GT; **Gate G1:** senkt Self-Training (Runde 1 gegen Runde 0) den U-Recall in den Niedrigrisikostrata? **Gate G2:** U-Schwelle gegen U-Oracle-Schwelle | 27 Läufe ≈ 24 h Remote-CPU, dazu ≈ 7 h für KSDD2 und MVTec AD 2 | 5–10 | Referenzzahlen auf allen Datensätzen mit 3 Seeds; Kosten(k)-Kurven; G1 und G2 entschieden und dokumentiert |
| **2 – Risikosignal und Synthese-Korrektur** | unsichtbare Positive beheben; ein Kontrolldetektor (AnomalyDINO, DINOv2-S) als Risiko-Merkmal; Stacker auf L-OOF; verify-first-Flags für die Reparatur | ein kNN- bzw. Injektionsdurchgang 11–27 min (Schätzung) | 10–15 | K′(k\*) oder n₉₀ sinkt um ≥ 10 % gegenüber Phase 1 (Go 1/4), sonst Komponente verwerfen |
| **3 – Labeler-Ausbau** | stärkerer Encoder (DINOv2/v3-Member), Synthese v2, Self-Training; doppeltes Cross-Fitting nur bei positivem G1 | voller Lauf 2,0–3,6 × heutiger Lauf (Schätzung) | 15–25 | jede Komponente erfüllt ihr Go-Kriterium aus §4.6 oder wird verworfen |
| **4 – Mensch-in-der-Schleife-Simulation** | nur **Einzelschuss-Politiken** aus gespeicherten Läufen (retrainende Politiken kosten ≈ 26 CPU-Tage pro Konfiguration); Aktionszählungen, Latin-Hypercube, fehlbarer Reviewer; vorher 1 h Selbst-Zeitmessung; Baselines (a-naiv) bis (e) | CPU, Stunden | 10–15 | H\* entschieden: Nordstern-Paar als Verteilung pro Datensatz und der Bereich des Kostenraums, in dem H\* gilt |
| **5 – Aufgabenanpassung** | 2–3 Fingerprint-Regeln × 3 Datensätze lokal; LODO (≈ 630 Läufe) später auf GPU | ≈ 15 h pro Datensatz und Seed auf der CPU (Schätzung) | 10–15 | Regeln erreichen ohne Tuning ≥ 90 % der Oracle-Konfiguration auf allen 3 Datensätzen, sonst bleibt die Default-Konfiguration |

**Freeze-Ablauf.** Jeder Zertifizierungslauf schreibt vor der ersten Ziehung ein Manifest (Labels, r,
Konventionen, Ziele, δ-Aufteilung, R_max, n, Seed) und dessen Hash. Standard ist R_max = 2, also
höchstens ein Retraining, mit δ_r = δ/2 pro Version. Eine reine Review-Erweiterung, deren Korrekturen
über die Identität angerechnet werden, braucht kein neues δ (07 §3.3).

---

## 7. Risiken und offene Punkte

* **Protokoll P ist nicht Gold.** s = 0,7 ist eine Annahme aus 07, keine Messung. Die Sensitivität
  der blinden Vollsuche ist unbekannt. Billige gültige Korrekturen gibt es nicht: Seed-Defekte
  versagen, sobald sie leichter sind als echte Misses (0,43–0,50 Falschzertifikate bei 0,5/0,9), und
  ein zweiphasiges Experten-Nachprüfen ist gültig, aber bei seltenen Fehlern teurer als ein direktes
  Audit unter P (07 §3.1). Vorschlag (ungeprüft): In der Selbst-Zeitmessung auch die Sensitivität der
  Blindsuche gegen GT grob schätzen.
* **Automation Bias** am Overlay ist argumentiert, nicht simuliert.
* **Subjektive Grenzen und subjektive Existenz.** Verify-first-Flags liefern keine TP/FP/FN-Pixel;
  Dice-Zertifikate brauchen volle Grenzkorrektur. Nötig sind eine Defektspezifikation pro Klasse, die
  Antwort „unsicher“ (zwei zertifizierte Versionen) und Orakelrauschen in der Simulation.
* **Elongierte Defekte.** Mit any-hit-Toleranz gilt ein zu 10 % getroffener Riss als erkannt. ζ und
  Recall sind für Risse also nachsichtig.
* **L–U-Verschiebung im Benchmark.** `data.make_split` zieht L stratifiziert über alle Defektarten,
  echte Nutzer wählen „schöne“ Beispiele. Alle Hypothesen sind auch unter verzerrtem L zu prüfen
  (Top-k nach Fläche oder Kontrast, 1–2 fehlende Arten, eine einzige Charge).
* **Lizenzen:** DINOv3, SAM 3, AnomalyVFM/RADIO und die Datensätze (MVTec CC BY-NC-SA) müssen vor
  produktiver Nutzung geprüft werden. DINOv2 (Apache 2.0) ist der Fallback.
* **Rechen- und Entwicklungsaufwand** sind auf dem M4 nicht gemessen; alle Angaben in §6 sind
  Schätzungen.
* **Quellenlage:** Viele Literaturzahlen stammen aus Abstracts und READMEs, und auch die
  Neuheitsprüfung in 07 konnte die wichtigsten Volltextquellen nicht abrufen. Vor einer
  Veröffentlichung sind sie gegen die PDFs zu prüfen.
* **Red-Team-Zahlen sind synthetisch.** Mit N markierte Befunde sind nicht gegenverifiziert.
* **Zertifikat ≠ Produktivqualität:** Eine zertifizierte Labelfehlerrate begrenzt nur die Verzerrung
  der Evaluation auf Pseudo-Labels. Das Produktivmodell wird auf einem reservierten Audit-Anteil
  designbasiert gemessen (04 §2.5).
