# 06 – Vollständigkeitsschätzung: Wie viele Defekte übersehen die Auto-Labels?

Theoriebeitrag zu `auto-label-tool` · **Fassung 2** · Stand 19.10.2026 (Fassung 1: 18.10.2026).
Bezüge: [01](01_foundation_models_anomaly.md) (Idee 11.3), [02](02_defect_synthesis.md) (N1),
[03](03_semisupervised_pseudolabels.md) (N2), [04](04_guarantees_measurement.md) (Satz 1, Prop. 4),
[07](07_red_team.md) (Red-Team-Bericht, maßgeblich für diese Fassung).

> **Fassung 2 nach Red-Team**
>
> * **Status: Forschungsspur.** Die Vollständigkeitsschätzer dieses Dokuments (N̂_HT, N̂_HG und der
>   Bell-Schätzer, in Fassung 1 „Hybrid“) sind modellbasiert. Sie sind **nie Teil eines Zertifikats**
>   und erscheinen nicht als Qualitätszahl im Bericht (07 §6, Go-Tabelle Nr. 10). Aus jeder Berichts-
>   und Zertifikatsrolle entfernt sind: N̂_HT, N̂_HG, der Hybrid, die Triangulation und die automatische
>   Ausweichregel.
> * **Zwei zulässige Produktrollen**, in denen Modellfehler nur Power kosten (§5.1):
>   (a) die K′-Prognose für die Auditplanung bzw. als Prior der PPR-Martingale;
>   (b) die Injection-Recovery-Härtekarte als Review-Risiko-Merkmal, nur wenn Go-Kriterium Nr. 8 aus
>   07 §6 erfüllt ist.
> * **Identifikation (§2.2):** Pro Kovariatenzelle x identifizieren die echten Daten nur n_obs(x) und
>   P_real(h | x, h ≠ 0). N(x) und P_real(0 | x) sind nicht identifiziert. Die Annahme T0, auf der
>   N̂_HT beruht, ist untestbar.
> * **Der „Transporttest“ heißt jetzt „Anpassungstest der beobachtbaren Implikation von T“ (§4).** Er
>   erkennt Anteilsverschiebungen (Logit-Lecks). Gegen proportionale bzw. Sichtbarkeitsabweichungen
>   hat er Power ≈ α. T0 und die getestete Hypothese sind logisch unabhängig.
> * **Garantie (§5.2):** nur Satz 1 + Prop. 4 aus 04, unter den Bedingungen aus 07 §7 Nr. 7.
>   `ppi.estimate_ratio` ist approximativ (nur SRS, feste n).
> * **Neuheit (§9):** C1 ist „teilweise bekannt“ (07 §4).
> * **Evidenz:** Nichts in diesem Dokument ist auf echten Daten gemessen. Alle Zahlen stammen aus
>   kleinen Monte-Carlo-Läufen auf synthetischen Fanggeschichten ([`sims/`](sims/), Bericht 07). Die
>   Szenarien 0–D (§3.4) und die Leck-Fälle (§4.2) wurden in dieser Revision nachgerechnet und stimmen
>   im Rahmen des Monte-Carlo-Fehlers überein. Die übrigen Zahlen sind aus 07 bzw. den gespeicherten
>   Ausgaben in `sims/outputs/` übernommen.
> * **Umgesetzt:** 07 §7 Nr. 7, 9, 18, 19, 20 und 21, die Befunde F3, F5, F8, F9, F13, F14 und F15 sowie
>   das Neuheitsurteil zu C1. Die Zuordnung steht im Änderungsprotokoll am Ende.

---

## 1. Problem: Die bisherige Selbsteinschätzung ist blind für übersehene Defekte

Die Reviews 01–03 kommen unabhängig voneinander auf dieselbe Lücke. **Für Pixel-Auto-Labels fehlt eine
etablierte Methode, die die übersehenen Defekte (FN) direkt schätzt.** In Software-Test
(Fehler-Seeding), zerstörungsfreier Prüfung (POD), Astronomie (Injection–Recovery), Daten-Cleaning
(DQM) und Technology-Assisted Review (Chao-Stopp) existieren solche Verfahren (§9). Fassung 1 hatte
formuliert, dass „keine etablierte Methode“ dies leiste. Das war zu weit.

Die aktuelle Selbsteinschätzung (`quality.py`, `exp_missed`) summiert die kalibrierten
Wahrscheinlichkeiten nicht ausgewählter Kandidatenregionen **desselben** Modells, das die Labels
erzeugt hat. Ein Defekt, den das Modell gar nicht „sieht“ (p ≈ 0), trägt fast nichts bei. Das ist
strukturell, denn ein Detektor kann seine eigenen blinden Flecken nicht messen. Die Kalibrierung auf
OOF-Daten hilft nur, solange die Defekte in U denen in L ähneln. Sie versagt gerade bei unbekannten
Defekttypen, schwachem Kontrast und ungewohntem Kontext, also dort, wo übersehene Defekte entstehen.

Zwei Informationsquellen liegen teilweise außerhalb der blinden Flecken des Labelers:

1. **Andere Detektoren** mit anderem induktivem Bias: Normalitätsmodell (DINO-kNN), ein zweiter
   überwachter Member mit anderem Backbone, ein Zero-Shot-VFM. Wo sie Defekte finden, die der Labeler
   nicht findet, wird ein Teil des blinden Flecks sichtbar.
2. **Künstlich eingefügte Defekte** mit bekannter Lage. Bei ihnen ist auch das Ereignis „von allen
   Detektoren übersehen“ beobachtbar.

Beide Quellen haben eine Grenze. Andere Detektoren teilen einen Teil der blinden Flecken, und
eingefügte Defekte sind nicht echte Defekte. **Keine der beiden Quellen identifiziert die Zelle „von
allen übersehen“ für echte Defekte** (§2.2). Das leistet nur eine Inspektion nach einem schriftlichen
Protokoll P, also das Audit oder ein Datensatz mit vollständiger Ground Truth.

---

## 2. Modell, Notation und Identifikation

### 2.1 Modell und Notation

* **Population:** U enthält N echte Defekte d = 1…N, N unbekannt. Ein Defekt ist eine
  Zusammenhangskomponente nach der `merge_radius`-Konvention von `metrics.py`. Jeder Defekt hat
  Kovariaten x_d (Fläche, lokaler Kontrast bzw. Normalitäts-Score, Elongation, Klasse,
  Kontext-Cluster). Für die Identifikationsaussagen wird x in Zellen (Bins) diskretisiert.
* **Detektorpanel:** J ≥ 2 Detektoren. Detektor j „fängt“ Defekt d (δ_dj = 1), wenn eine seiner
  Kandidatenregionen den Defekt innerhalb der Toleranz trifft. Detektor 1 = A ist der Labeler, also
  die ausgelieferten Masken. Die Fanggeschichte ist h_d = (δ_d1, …, δ_dJ) ∈ {0,1}^J.
* **Beobachtet** wird nur, was mindestens ein Detektor fängt. Diese Regionen r ∈ R enthalten auch
  Fehlalarme. Jede Region bekommt q_r = P(echter Defekt | Merkmale) aus dem OOF-kalibrierten
  Regions-Stacker (01-R3, 03-R4b). n_h(x) ist die q-gewichtete Zahl der Regionen mit Geschichte h in
  Zelle x, und n_obs(x) = Σ_{h≠0} n_h(x).
* **Fangwahrscheinlichkeiten:** P_real(h | x) für echte Defekte, P_syn(h | x) für injizierte.
  π_real(x) = 1 − P_real(0 | x) und π_syn(x) = 1 − P_syn(0 | x) sind die Erfassungswahrscheinlichkeiten.
  **Notation:** π ohne Zusatz ist in diesem Dokument immer eine Fangwahrscheinlichkeit. Inklusions-
  wahrscheinlichkeiten des Audits heißen π_i^audit. Die Ziehungswahrscheinlichkeiten des Audits beim
  Ziehen mit Zurücklegen heißen wie in 04 q_t(i). Sie sind nicht mit den Regionswahrscheinlichkeiten
  q_r zu verwechseln.

**Annahmen**, getrennt benannt, weil sie verschieden prüfbar sind:

| Kürzel | Aussage | prüfbar mit Fanggeschichten? |
|---|---|---|
| T (Transport) | P_real(h \| x) = P_syn(h \| x) für alle h, einschließlich h = 0 | nur der Teil H_obs |
| T0 (Sichtbarkeit) | P_real(0 \| x) = P_syn(0 \| x), d. h. π_real(x) = π_syn(x) | nein |
| H_obs (beobachtbare Implikation) | P_real(h \| x, h ≠ 0) = P_syn(h \| x, h ≠ 0) für alle h ≠ 0 | ja (§4) |
| CI | Detektoren auf echten Defekten bedingt unabhängig gegeben x | für J = 2 nein (§3.2) |
| OR-Transport | OR_real(x) = OR_syn(x) (J = 2) | nein (§3.3) |

Es gilt T ⇔ T0 ∧ H_obs, denn P(h | x) = (1 − P(0 | x)) · P(h | x, h ≠ 0) für h ≠ 0.

**Gesucht** sind
(i) N, die Gesamtzahl der Defekte in U;
(ii) N_A, die Zahl der echten Defekte, die A erfasst, also der Defekt-Recall des Pools R_A = N_A/N;
(iii) pro Bild i die erwartete Zahl der von allen übersehenen Defekte (Prädiktor, §5.3);
(iv) K′, die Zahl der Restfehlbilder im Auditrahmen nach dem Review (Planungsgröße, §5.1).
(i) und (ii) sind ausschließlich Forschungsgrößen.

### 2.2 Identifikation: Was die Daten festlegen und was nicht

**Aussage I (Identifikation, 07 F3, [U, Kern B]).** Für die Zellzahlen gelte das Poisson-Modell
n_h(x) ~ Poisson(N(x) · P_real(h | x)). Dann legen die echten Fanggeschichten pro Kovariatenzelle x nur
zwei Dinge fest:

* den Erwartungswert von n_obs(x), also N(x) · π_real(x), und
* die bedingten Anteile P_real(h | x, h ≠ 0), also 2^J − 2 freie Parameter.

P_real(0 | x) und damit N(x) sind nicht identifiziert. Zu jedem Wert π ∈ (0, 1] passt bei gleichen
bedingten Anteilen die Population N(x) = E[n_obs(x)]/π mit genau derselben Verteilung der
beobachtbaren Daten.

**Folgerungen**

1. Bei kalibriertem q, fehlerfrei gemessenem x und bekanntem π_syn gilt

       E[N̂_HT] = Σ_x N(x) · π_real(x) / π_syn(x).

   N̂_HT nutzt die echten Daten also nur über n_obs(x). Die Anteile, die der Anpassungstest prüft,
   gehen in N̂_HT gar nicht ein.
2. N̂_HT ist erwartungstreu, wenn T0 zellweise gilt. „Genau dann“ wäre zu stark, weil sich
   Abweichungen verschiedener Zellen in der Summe aufheben können. T0 ist mit Fanggeschichten allein
   **untestbar**.
3. Der Anpassungstest (§4) prüft nur H_obs. **T0 und H_obs sind logisch unabhängig.** Beide Richtungen
   kommen in den Simulationen aus 07 vor (J = 2, drei Größenklassen, 300 echte Defekte,
   600 Injektionen, 2000 Wiederholungen, Testniveau 0,05):

   | Szenario | T0 | H_obs | Ablehnrate Test | rel. Bias N̂_HT | N̂_HG | Bell (Fassung 1: Hybrid) |
   |---|---|---|---|---|---|---|
   | A: 30 % der echten Defekte für beide Detektoren unsichtbar | falsch | wahr | 0,05 | −0,30 | −0,29 | −0,28 |
   | E: komplementäre Detektoren, P_real(11) = 0,4 · p_A · p_B, P(00) wie Synthetik | wahr | falsch | 1,00 | −0,000 | +0,827 | +0,905 |

   In A bemerkt der Test nichts, obwohl alle drei Schätzer 30 % zu niedrig liegen. Der geschätzte
   Recall von A beträgt dort 0,71, der wahre 0,50. In E lehnt der Test immer ab, obwohl N̂_HT
   unverzerrt ist. Die Regel aus Fassung 1 („bei Ablehnung auf Huggins ausweichen“) liefert in E
   +0,827. Ohne jeden gemeinsamen Fang (κ = 0 statt 0,4) wird N̂_HG um +31,6 verzerrt
   (`sims/outputs/redteam2_transport_identifiability.out`).
4. Die einzige Quelle, die P_real(0 | x) identifiziert, ist eine Inspektion nach Protokoll P: im
   Audit (designbasiert, für die gezogenen Bilder) oder auf Datensätzen mit vollständiger GT (§8, H5).
   Deshalb ist das Audit Pflicht und keine Option. Auch P ist kein Gold, das Zertifikat gilt relativ zu
   P (07 §3.1).
5. Einordnung: Link (2003) zeigt die Nicht-Identifizierbarkeit von N unter Heterogenitätsmodellen.
   Wachter & Freedman (2000, „The Fifth Cell“) diskutieren die unbeobachtbare Zelle der
   Dual-System-Schätzung. Die verschärfte These „kein Test auf beobachteten Zellen erkennt eine
   Verletzung der fünften Zelle“ ist nach 07 (Anhang A) zu stark. Verletzungen, die auch die Anteile
   verschieben, werden teilweise erkannt (Szenario B: 0,57, Szenario C: 0,69). Unsichtbar sind
   Verletzungen, die nur die Zelle h = 0 betreffen.

**Sensitivitätsband statt Punktwert.** Weil T0 untestbar ist, wird N̂ (im Forschungsprotokoll) als
Kurve über einen vorab registrierten Bereich berichtet:

    N̂(τ) = Σ_x n_obs(x) / ( π̂_syn(x) · (1 − τ) ),   τ ∈ [τ_min, τ_max]

Dabei ist τ = 1 − π_real/π_syn der Anteil echter Defekte, der *zusätzlich* zur Synthetik für alle
Detektoren unsichtbar ist. τ < 0 bedeutet, dass die Synthetik schwerer ist als die Realität. Hängt τ
von x ab und liegt τ(x) in [τ_min, τ_max] für alle x, gilt bei kalibriertem q, fehlerfreiem x und
bekanntem π_syn

    E[N̂(τ_min)] ≤ N ≤ E[N̂(τ_max)],

weil N̂(τ) in τ monoton steigt. Beispiel: In Szenario A (τ = 0,3) ist N̂(0,3) unverzerrt, der Punktwert
N̂(0) unterschätzt um 30 %. τ ist aus Fanggeschichten nicht schätzbar. Der Bereich muss begründet
werden, etwa aus GT-Datensätzen (§8, H5) oder aus Audit-Funden, die kein Detektor gefangen hatte. Mit
τ_max → 1 wird das Band unbeschränkt. Für die Planung (Rolle (a), §5.1) gilt das obere Bandende. Für
den Bell-Schätzer gibt es das analoge Band über das Odds Ratio (§3.3).

---

## 3. Schätzer (Forschungswerkzeug)

### 3.1 Horvitz–Thompson mit Injection–Recovery (Annahme T0)

Synthetische Defekte s mit bekannten Kovariaten x_s werden in zufällige U-Bilder eingefügt
(Residual-Paste aus der Defektbank, Gain und Größe gestreut). Alle Detektoren laufen auf den
modifizierten Bildern. So werden die **vollständigen** Fanggeschichten inklusive h = 0 beobachtet, und
daraus wird P_syn(h | x) geschätzt, insbesondere π̂_syn(x).

**Form von π(x).** Parametrisch wie in Probability-of-Detection-Studien der zerstörungsfreien Prüfung
(MIL-HDBK-1823A; MAPOD, model-assisted POD): ein Hit/Miss-GLM mit Logit- oder Probit-Link in
log(Größe), erweitert um Kontrast, Elongation, Klasse und Kontext. Das glättet π̂ und macht die
Extrapolation sichtbar. Kennwerte wie a90/95 (Größe mit 90 % POD bei 95 % Konfidenz) sind aus
Injektionen **nur modellbasiert**, weil sie T voraussetzen. Sie sind kein Zertifikat.

**Schätzer:**

    N̂_HT = Σ_{r ∈ R} q_r / π̂(x_r)
    N̂_A  = Σ_{r ∈ R, r ⊂ A} q_r            (direkt beobachtet, kein π nötig)
    R̂_A  = N̂_A / N̂_HT

**Bedingungen für Erwartungstreue** (bei bekanntem π):

1. **T0 zellweise** (§2.2). Untestbar.
2. **q bedingt auf (x, h) kalibriert:** E[1{r echt} | x_r, h_r, q_r] = q_r. Marginal kalibriertes q
   reicht nicht. In Szenario C (07) erhalten 300 falsche B-only-Regionen q = 0,1 statt 0. Das ergibt
   +0,14 Bias für N̂_HT, +0,29 für N̂_HG und +0,31 für Bell.
3. **x fehlerfrei und für echte wie synthetische Defekte gleich gemessen.** Für echte Defekte wird x aus
   Detektorausgaben gemessen, für injizierte ist es aus den Injektionsparametern bekannt. In
   Szenario D (Größe von B-only-Fängen zur Hälfte eine Klasse zu groß gemessen) ist N̂_HT kaum
   verzerrt (−0,01), N̂_HG um −0,05. Der Anpassungstest lehnt trotzdem in 0,57 der Läufe ab. Bei
   gefangenen Injektionen sind wahre und gemessene Kovariaten beide bekannt. Daraus lässt sich der
   Messfehler schätzen und korrigieren (Regressionskalibrierung, ungetestet).
4. **Positivität:** π(x) > 0 für alle echten Defekte. Defekte, die *kein* Detektor finden kann, sind aus
   Fanggeschichten prinzipiell nicht schätzbar. Für sie bleibt nur das Audit.

**Varianz** (Skizze, π bekannt, Regionen unabhängig, 07 F14 [N]):

    Var(N̂_HT − N) ≈ Σ_d (1 − π_d)/π_d  +  Σ_r q_r (1 − q_r) / π_r²  +  (Schätzfehler von π̂)

Der erste Term ist die Fangzufälligkeit, als Plug-in Σ_r q_r (1 − π_r)/π_r². Der zweite ist der
Fehlalarmterm, der in Fassung 1 fehlte.

**Schätzfehler von π̂.** Wegen E[1/π̂] > 1/π (Jensen) erzeugt Rauschen in π̂ eine Aufwärtsverzerrung. Ist
π̂ = 0 in einer Zelle, divergiert der Schätzer. Deshalb:

* π̂(x) geglättet schätzen (GLM wie oben) und mit einem vorab registrierten Floor π_min nach unten
  begrenzen;
* den Anteil von N̂ berichten, der aus Zellen am Floor stammt;
* die Unsicherheit per Bootstrap **über Bilder** schätzen, nicht über einzelne Injektionen. Gezogen
  werden ganze Injektionsbilder, weil mehrere Injektionen eines Bildes Bildeffekte teilen, und ganze
  U-Bilder, weil echte Defekte clustern.

Ein Synthetik-Leck überschätzt π und unterschätzt damit N (§6, Punkt 1).

### 3.2 Huggins–Alho: bedingte Likelihood auf echten Fanggeschichten (Annahme CI)

Ohne Synthetik gilt unter **bedingter Unabhängigkeit der Detektoren gegeben x**
P(h | x) = Π_j p_j(x)^{h_j} (1 − p_j(x))^{1 − h_j}. Die p_j(x) werden aus den echten Fanggeschichten über
die auf h ≠ 0 bedingte Likelihood geschätzt:

    ℓ(θ) = Σ_{r ∈ R} q_r · log[ P_θ(h_r | x_r) / (1 − P_θ(0 | x_r)) ],   p_j(x) = σ(θ_jᵀ x)

Danach ist N̂_HG = Σ_r q_r / π̂_θ(x_r), der Schätzer von Huggins (1989) bzw. Alho (1990). Für J = 2 und
diskrete Zellen ist das pro Zelle Lincoln–Petersen, N̂_HG(x) = n_obs + n₁₀ n₀₁ / n₁₁, mit
Chapman-Korrektur n_obs + n₁₀ n₀₁ / (n₁₁ + 1).

* **Für J = 2 ist auch CI untestbar.** Pro Zelle stehen drei beobachtbaren Zellen drei Parameter
  (N, p_A, p_B) gegenüber. Das Modell passt also immer. Mit stetigen Kovariaten ist nur die
  funktionale Form prüfbar.
* **Die Richtung des Bias ist nicht einseitig** (07 F8). Fassung 1 nannte nur die erste Richtung.
  Restabhängigkeit, die x nicht erklärt, wirkt in beide Richtungen:
  * positive Restabhängigkeit (gemeinsame blinde Flecken, OR_real > 1) führt zu **Unterschätzung**
    (Szenario B: −0,17 bei sd 1,5 logit, −0,29 bei sd 3);
  * negative Restabhängigkeit (komplementäre Detektoren, OR_real < 1) führt zu **Überschätzung**
    (Szenario E: +0,83, ohne gemeinsame Fänge +31,6).
* **J = 3 ist gesättigt.** Das log-lineare Modell mit allen Zweifach-Interaktionen hat 7 Parameter für
  7 beobachtbare Zellen. Es passt auf jeden Datensatz, und die Annahme „keine Dreifach-Interaktion“
  ist nicht testbar, bestimmt aber N̂. Fassung 1 („Annahme ist dann nur noch …“) hat das als Vorteil
  dargestellt. Erst ab J = 4 gibt es Modelle mit Restfreiheitsgraden, deren beobachtbare
  Implikationen testbar sind.
* **Doppelt robuste Schätzer** nach Art von `drpop` (Das, Kennedy, Jewell 2024) schätzen p_j(x) flexibel
  und verringern Fehlspezifikation in x. Sie **setzen weiterhin CI gegeben x voraus** und beseitigen
  die Korrelationsverzerrung nicht (07 M5).

### 3.3 Bell-Schätzer mit fixiertem Odds Ratio (Fassung 1: „Hybrid“)

Der Hybrid aus Fassung 1 („Abhängigkeitsstruktur aus der Injektion, Niveau aus echten Daten“) ist
Bells Dual-System-Schätzer mit fixiertem Odds Ratio (Bell 1993, JASA 88:1106; 07 F8 [N, algebraisch],
M5 [B]). Für J = 2 legen die drei beobachteten Zellen pro x die Größen e^{u+u_A}, e^{u+u_B} und
e^{u+u_A+u_B+u_AB} fest. Mit u_AB = log OR_syn(x) aus den Injektionen folgt

    N̂_Bell(x) = n_obs(x) + OR_syn(x) · n₁₀(x) · n₀₁(x) / n₁₁(x).

* **Exakt identifiziert und untestbar.** Drei Parameter stehen drei Zellen gegenüber, es bleibt kein
  Freiheitsgrad. Das Modell passt auf jeden Datensatz. Der Schätzer ist konsistent, wenn
  OR_real(x) = OR_syn(x) gilt (bis auf Auslöschung über Zellen). Dieser OR-Transport ist genauso
  untestbar wie T0.
* **Spezialfall:** Für OR_syn ≈ 1 ist Bell ≡ Huggins.
* **Gestrichen:** der Satz aus Fassung 1, die Abhängigkeitsstruktur sei „vermutlich robust“. Es gibt
  keinen Beleg, dass sich das Odds Ratio besser von Synthetik auf Realität überträgt als die
  Sichtbarkeit.
* **Stabilität:** Der Term 1/n₁₁ ist bei wenigen Doppelfängen instabil. Deshalb wird die
  Chapman-artige Form n_obs + OR · n₁₀ n₀₁ / (n₁₁ + 1) verwendet. Für OR = 1 ist sie identisch mit dem
  Chapman-Schätzer (n₁ + 1)(n₂ + 1)/(m + 1) − 1. OR_syn wird geglättet geschätzt (+0,5 je
  Injektionszelle wie in `redteam_capture_recapture.py`).
* **Sensitivität:** N̂_Bell steigt monoton in OR. Berichtet wird die Kurve über einen vorab
  registrierten Bereich [OR_lo, OR_hi] um OR_syn.

### 3.4 Verhalten unter Annahmeverletzungen (synthetisch)

Aufbau wie in §2.2 (J = 2, drei Größenklassen, N_real = 300, 600 Injektionen, 2000 Wiederholungen,
Testniveau 0,05, q perfekt außer in C). „Übereinstimmung“ heißt: max/min der drei Schätzer < 1,15.
Quellen sind `redteam_capture_recapture.py` (0–D, in dieser Revision nachgerechnet, identisch) und
`redteam2_transport_identifiability.py` (E).

| Szenario | Ablehnrate Anpassungstest | rel. Bias N̂: HT / HG / Bell | Übereinstimmung | Recall A geschätzt vs. wahr |
|---|---|---|---|---|
| 0 Null (T und CI gelten) | 0,05 | +0,00 / +0,01 / +0,02 | 0,93 | 0,71 vs. 0,71 |
| A 30 % für beide Detektoren unsichtbar | 0,05 | −0,30 / −0,29 / −0,28 | 0,89 | 0,71 vs. **0,50** |
| B gemeinsame Schwierigkeit, sd 1,5 logit | 0,57 | −0,08 / −0,17 / −0,17 | 0,77 | 0,73 vs. 0,67 |
| B′ dto., sd 3 logit | 1,00 | −0,16 / −0,29 / −0,29 | 0,09 | 0,75 vs. 0,62 |
| C 300 falsche B-only-Regionen, q = 0,1 fehlkalibriert | 0,69 | +0,14 / +0,29 / +0,31 | 0,46 | 0,62 vs. 0,71 |
| D Kovariate der B-only-Fänge zu groß gemessen | 0,57 | −0,01 / −0,05 / −0,03 | 0,98 | 0,72 vs. 0,71 |
| E komplementäre Detektoren (T0 gilt) | 1,00 | −0,000 / +0,827 / +0,905 | – | – |

Die Tabelle zeigt, warum R̂_A als Qualitätszahl gefährlich ist (A, E). Sie zeigt auch, dass Ablehnung
und Bias von N̂_HT nicht verlässlich zusammenhängen: In D lehnt der Test ab, obwohl der Bias klein ist.
In A lehnt er nicht ab, obwohl der Bias groß ist.

### 3.5 Übereinstimmung der Schätzer ist kein Beleg (07 F9 [N])

Alle drei Schätzer sehen die echten Daten nur über die beobachteten Zellen. Ihre Übereinstimmung ist
deshalb kein Validitätsbeleg: In A stimmen sie in 0,89 der Läufe überein, bei −30 % Bias. Eine
Mehrheitslogik („zwei von drei“) ist irreführend, weil N̂_HG und N̂_Bell keine unabhängigen Urteile
sind (beide hängen über n₁₀ n₀₁ / n₁₁ zusammen und fallen für OR ≈ 1 zusammen). In E liegen zwei von drei
„Stimmen“ falsch.

Zulässig ist allein eine **einseitige Falsifikationsprüfung** von {T0 ∧ CI}: Weichen N̂_HT und N̂_HG
stärker ab, als es der Bild-Bootstrap erklärt, ist mindestens eine der beiden Annahmen verletzt.
Übereinstimmung sagt nichts. Die Triangulation aus Fassung 1 (Übereinstimmung als Diagnosesignal für
das Auditbudget) ist gestrichen.

---

## 4. Anpassungstest der beobachtbaren Implikation von T (Fassung 1: „Transporttest“)

### 4.1 Was getestet wird

Aus der Injektion folgt eine Vorhersage für die **beobachtbaren** Zellen echter Defekte:

    P(h | x, h ≠ 0) = P_syn(h | x) / (1 − P_syn(0 | x))   für h ∈ {0,1}^J ∖ {0}

Für echte gefangene Defekte sind die q-gewichteten Anteile der Zellen „nur A“, „nur B“, „A und B“ usw.
beobachtbar. Der Test vergleicht beide Seiten pro Kovariatenzelle. Er prüft also **H_obs**. T0, die
einzige Annahme, die N̂_HT braucht, prüft er nicht (§2.2). Ein Testergebnis sagt daher nichts direkt
über die Verzerrung von N̂_HT. Nach dem Urteil zu C1 (07 §4) heißt er deshalb nicht mehr
„Transporttest“ und gilt nicht als Test der Transportannahme, sondern als Anpassungstest auf den
beobachtbaren Zellen. Er ist blind für Defekte, die kein Detektor sieht.

### 4.2 Was er erkennt und was nicht

Synthetische Abweichungen (N_real = 300, 600 Injektionen, 2000 Wiederholungen, Niveau 0,05). Quellen:
07 F3, `redteam2_verify_math_checks.py` (Teil B) und `redteam2_transport_identifiability.py` (G).

| Abweichung zwischen Synthetik und Realität | Ablehnrate | rel. Bias N̂_HT |
|---|---|---|
| Leck nur bei A: Synthetik logit_A + 1 | 0,81 | −0,105 |
| Leck bei beiden: Synthetik logit_A, logit_B + 1 | 1,00 | −0,139 |
| proportional: 20 % der echten Defekte für alle unsichtbar | 0,04 | −0,198 |
| gemeinsame Schwierigkeit sd 1,5 (B), 40 / 100 / 300 echte Defekte | 0,13 / 0,28 / 0,58 | ≈ −0,08 |

Bei der Nachrechnung von Teil B in dieser Revision (anderer Zufallsstrom) ergaben sich 0,81 / −0,106,
1,00 / −0,138 und 0,05 / −0,200.

Daraus folgt:

* Der Test **erkennt Anteilsverschiebungen**, etwa Logit-Lecks, die die Detektoren unterschiedlich
  oder auf der Logit-Skala gemeinsam treffen.
* Gegen **proportionale Abweichungen** P_real(h | x) = c(x) · P_syn(h | x) für alle h ≠ 0, also reine
  Sichtbarkeitsabweichungen, hat er **Power ≈ α**. Das ist genau der Fall „neuer Defekttyp“ oder „zu
  schwach für alle Detektoren“.
* Bei realistischen Zahlen echter Defekte ist die Power auch gegen erkennbare Abweichungen gering
  (0,13 bei 40 Defekten). Das trifft den Zielfall mit seltenen Defekten.

### 4.3 Clusterrobustheit und Multiplizität (07 F13 [N])

Der χ²-Homogenitätstest aus Fassung 1 setzt unabhängige Einheiten mit ganzzahligen Zählungen voraus.
Beides ist verletzt: Defekte clustern in Bildern mit gemeinsamen Bildeffekten (Reflexe, Textur), und
die Zählungen sind q-gewichtete Bruchzahlen. Simuliertes Niveau, wenn T marginal gilt und ein
Bildeffekt beide Logits senkt (3 Injektionen pro Bild, nominal 0,05,
`redteam2_transport_identifiability.py`, Teil F):

| sd des Bildeffekts | ≈ 1 echter Defekt pro Bild | ≈ 4 echte Defekte pro Bild |
|---|---|---|
| 0 | 0,044 | 0,049 |
| 1 | 0,050 | 0,066 |
| 2 | 0,057 | 0,089 |

Vorgaben für Fassung 2, alle **ungetestet**:

* **Bildcluster-robuster Test.** Wald-Test auf die Differenz der Zellanteile mit bildgeclusterter
  Kovarianz (Sandwich oder Bild-Bootstrap, echte und Injektionsbilder getrennt als Ziehungseinheiten,
  q-Gewichte mitgeführt). Alternative ist eine Rao–Scott-Korrektur des χ².
* **Multiplizität.** Ein vorab registrierter globaler Test über alle Zellen auf Niveau α. Zellweise
  Aussagen dienen nur der Lokalisierung und werden mit Holm korrigiert. Die Zellgrenzen werden
  festgelegt, bevor echte Fanggeschichten angesehen werden.

### 4.4 Was aus dem Ergebnis folgen darf

* **Die automatische Ausweichregel ist gestrichen.** In E treibt sie den Bias von 0 auf +0,827. Nach dem
  Gegenargument in 07 F3 ist sie nicht universell falsch: In den beiden Logit-Leck-Fällen senkt sie den
  Bias (−0,105 → −0,006 und −0,139 → +0,009), weil dort auf den echten Daten CI gilt. Ohne eine
  ausdrückliche Verknüpfungsannahme ist sie aber unbegründet.
* Ein **Wechsel des Schätzers** ist nur unter einer genannten, vorab registrierten Verknüpfungsannahme
  zulässig, z. B. „Abweichungen zwischen Synthetik und Realität sind Logit-Verschiebungen der
  Synthetik, und auf echten Daten gilt CI gegeben x“. Diese Annahme ist ihrerseits untestbar und wird
  im Forschungsprotokoll mitgeführt.
* **Zulässige Verwendungen:**
  * Diagnose der Synthetik, also ein Hinweis, Gain-, Größen- oder Kontextverteilung zu überarbeiten;
  * Hinweis auf Strata, die mehr Review verdienen, nur in Rollen, in denen Fehler Power kosten (§5.1).
    Eine nicht-uniforme Audit-Allokation setzt zusätzlich Go-Kriterium Nr. 7 aus 07 §6 und den Floor
    voraus;
  * Forschungsprotokoll.
* **Nicht zulässig:**
  * eine Aussage wie „Vollständigkeitsschätzung in Stratum X vertrauenswürdig“ im Qualitätsbericht;
  * Nicht-Ablehnung als Stütze für T0.

---

## 5. Produktrollen und Kopplung an das Audit

### 5.1 Zulässige Rollen: Modellfehler kosten nur Power

Eine modellbasierte Größe aus diesem Dokument darf in den Produktpfad nur dort, wo ihre Fehler Power
bzw. Effizienz kosten und nicht Gültigkeit (07 §6, Abschnitt „Rolle von C1 im Produktpfad“, [U]).
Unstrittig ist dort: C1 ist für die Gültigkeit redundant, und als Qualitätszahl ist C1 gefährlich.

**(a) K′-Prognose für die Auditplanung.** K′ ist die Zahl der Restfehlbilder (b_i = 1{M_i > 0}) im
Auditrahmen nach dem Review. Ein naheliegender Prognoseansatz (Vorschlag, ungetestet) ist

    K̂′ = Σ_{i im Rahmen} [ 1 − Π_{r ∈ i, r ⊄ A} (1 − q_r) · exp(−ĥ_i) ].

Dabei ist ĥ_i die vorhergesagte Zahl der von allen übersehenen Defekte in Bild i (§5.3). Weil das
Audit in CertAL-lite auch R einschließt, kommt für bereits reviewte Bilder ein Restanteil hinzu, der die
angenommene Reviewer-Sensitivität abbildet. Verwendung:

* In CertAL-lite (07 §6, L5) bestimmt K̂′ das fixe n aus der exakten Powertabelle. Die Schranke bleibt
  gültig, solange n vor der ersten Ziehung registriert ist. Eine zu optimistische Prognose kostet
  Power: Ein fixes n = 332, geplant für K′ = 10 (N = 2000, ζ\* = 0,02, Korrekturen angerechnet), hat
  Power 0,93 / 0,57 / 0,24 bei wahrem K′ = 10 / 20 / 30 (`certify.certification_power`). Die in
  07 §5.2 genannten 0,93 / 0,56 / 0,23 für n = 310 gelten im Rahmen N′ = 1850; im ganzen Pool hätte
  n = 310 nur 0,81 / 0,38 / 0,13. Geplant wird deshalb konservativ, mit dem oberen Ende des
  Sensitivitätsbands (§2.2).
* Bei der PPR-Martingale ohne Zurücklegen zentriert K̂′ den Prior. Das beeinflusst nur die Effizienz
  (Go-Kriterium Nr. 2 in 07 §6).
* Die Prognose K̂′(k) kann auch die Wahl der Review-Größe k informieren (Kostenform in 07 §5.3). Fehler
  kosten auch hier nur Effizienz, weil das Audit den ganzen Pool einschließlich R zieht.
* **Bedingungen:** K̂′ wird vor der ersten Auditziehung berechnet und eingefroren. Eine Neuplanung auf
  Auditdaten ist nur über den vorab registrierten Erweiterungsblick mit δ-Aufteilung zulässig (07 L5).
  K̂′ speist keine Schranke.

**(b) Härtekarte als Review-Risiko-Merkmal.** Injection–Recovery liefert pro Bildkontext eine
vorhergesagte Wahrscheinlichkeit, dass ein echter Defekt gegebener Größe von allen Detektoren übersehen
wird (02-N1). Als Merkmal des Review-Risikoscores (07 L3) lenkt sie das Review auf Bilder, in denen
unsichtbare Misses wahrscheinlich sind.

* Fehler kosten hier nur Effizienz, weil die Gültigkeit aus dem Audit über den ganzen Pool kommt,
  R eingeschlossen (07 §3.2, L5). Die Review-Reihenfolge bestimmt nur, welche Fehler vor dem Audit
  repariert werden.
* **Gate:** Go-Kriterium Nr. 8 aus 07 §6. Die 20 % Bilder mit der höchsten Härte müssen ≥ 50 % der
  echten Misses enthalten, die alle Detektoren übersehen haben, gemessen auf GT-Datensätzen (§8, H6).
  Ohne erfülltes Gate entfällt die Rolle.
* Eine Verwendung als Allokationskovariate im Audit (PPS) erfordert zusätzlich Go-Kriterium Nr. 7
  (n₉₀ sinkt um ≥ 15 % bei gleicher Gültigkeit) und den Floor aus §5.2.
* S3 wird nach gefundenen Misses pro Review-Minute bewertet, nicht nach dem Fehler von N̂ (07 A16, [N]).

**Nicht zulässig** im Produktpfad:

* N̂_HT, N̂_HG, N̂_Bell und R̂_A als Qualitätszahlen in Bericht oder Zertifikat;
* die Triangulation;
* der Status des Anpassungstests als Vertrauensaussage;
* jede Schranke (D_max, Wertebereich [a, b]), die aus diesen Schätzungen abgeleitet ist.

### 5.2 Garantieaussage (eingeschränkt)

Fassung 1 behauptete, die Gültigkeit des Audit-Intervalls hänge „nur vom Stichprobendesign ab“ und die
Vollständigkeitsschätzung sei „ein reiner Effizienzgewinn ohne Gültigkeitsrisiko“. In dieser
Allgemeinheit ist das falsch (07 F5, F7).

**Was gilt.** Die einzige Garantie im Produktpfad kommt aus dem designbasierten Audit. Gehen Größen
aus diesem Dokument als Prädiktor f, als Gewicht λ oder über die Ziehungswahrscheinlichkeiten q_t in das
Audit ein, beschränkt sich die Garantieaussage auf Satz 1 (IPW-Betting mit prädiktablen Größen) und
Prop. 4 (Fixed-Sequence-Fieller-Inversion für Verhältnismetriken) aus 04 §4.5. Sie gilt nur unter
folgenden Bedingungen (07 §7 Nr. 7):

1. f, λ und q sind vor dem Audit eingefroren oder lernen nur **prädiktabel** aus dem Audit, d. h.
   q_t, f^(t) und λ_t sind F_{t−1}-messbar. Bei fixem n ist nur **Cross-Fitting** zulässig.
2. q hat einen **von S3 unabhängigen Floor** q_t(i) ≥ γ/N′. Das ist eine **Gültigkeitsbedingung**, kein
   bloßer Schutz. Ohne Floor liefert Prop. 3 bei σ_i = 0 die Inklusionswahrscheinlichkeit 0
   (07 F7(e)). Im Code baut `certify.floored_inclusion` q mit Floor, und `certify.check_floor` lehnt
   jedes q mit q_i < γ/N ab; die Sims (`sim_recall_cert.py`, `sim2.py`) rufen die Prüfung auf. Ein
   q-gewichtetes Audit selbst ist in `autolabel/` nicht implementiert.
3. r, Ŷ (einschließlich der Wahrscheinlichkeitskarten), U′/R, die Konventionen (tol, merge_radius) und
   die Schranken (D_max, [a, b]) sind vor der ersten Ziehung eingefroren und gehasht. Die Schranken
   stammen nicht aus S3 bzw. aus diesem Dokument.
4. Die Messung ist nicht von S3 beeinflusst: Inspektion nach Protokoll P (blinde Vollsuche, dann
   Overlay, Vereinigung). Verify-first mit S3-Overlay gilt nicht als Messung (07 §3.1).

Dazu kommen Bedingungen, die nicht spezifisch für 06 sind (07 F1, F4, §3.2, §3.3):

* eine vorab registrierte harte Schranke y_i ∈ [a, b] für alle Einheiten. Für den Zähl-Recall ist das
  D_max, eine Modellannahme, die das Design nicht prüfen kann. Eine beobachtete Überschreitung macht
  das Zertifikat ungültig, ihr Ausbleiben belegt nichts;
* prädiktable Wettobergrenzen z^max_t;
* Zielgrößen auf dem ausgelieferten Pool, R im Rahmen;
* δ-Spending über Runden.

Unter diesen Bedingungen gilt der Satz aus Fassung 1 in eingeschränkter Form: **Schlechte Prädiktoren
machen das Intervall nach Satz 1 breiter, nicht falsch.**

Im Standarddesign CertAL-lite für ζ (exakte Fixe-n-Schranke bzw. PPR ohne Zurücklegen, 07 §6) geht aus
diesem Dokument nur die K′-Prognose ein, und zwar nur in die Wahl von n bzw. in den Prior (§5.1). Die
Schranke selbst nutzt keine Größe aus 06.

**`ppi.estimate_ratio` ist approximativ, nur für SRS und feste n.** Der Schätzer nutzt Delta-Methode,
t-Quantil und SRS-Endlichkeitskorrektur und ist nur asymptotisch gültig. Abdeckung der einseitigen
95-%-LCB in der Gegenprüfung (07 F5 [B für diesen Teil]):

| Prädiktor | n = 50 | n = 150 |
|---|---|---|
| gut | 0,962 | 0,955 |
| schlecht (schwere Ränder), nominal | 0,884 | 0,943 |
| schlecht, nur definierte Läufe (24 % NaN) | ≈ 0,85 | ≈ 0,93 |
| kein Prädiktor (λ = 0) | 0,594 | 0,824 |

* Schlechte Prädiktoren machen dieses Intervall also *falsch*, nicht nur breiter.
* Unter PPS ist der ungewichtete Schätzer stark verzerrt (mittlerer Bias −0,94 bei wahrem
  Recall ≈ 0,80; 07 F5 (ii), [U]).
* Fassung 1 verwies für nicht-uniforme Designs auf eine „IPW-Variante“. Diese gibt es in `ppi.py`
  nicht. Der Verweis ist entfernt. Auch eine IPW-CLT-Variante wäre nur approximativ.
* Bei t_b ≤ 0 soll die Funktion künftig „kein Intervall“ (triviales Intervall) statt NaN ausgeben
  (07 §7 Nr. 19). Die Änderung wird parallel zu dieser Revision in `ppi.py` umgesetzt und ist hier nicht
  geprüft.
* Verwendung: nur als gekennzeichnete approximative Beschreibung, nie als Zertifikat.

**Nachkalibrierung von q und f auf Auditdaten** ist nur prädiktabel (in Satz 1) oder cross-fitted
zulässig. Fassung 1 (§6.3: Fehlkalibrierung „über das Audit korrigiert, Gültigkeit bleibt“) war bei
fixem n eine In-Sample-Anpassung. Die Rangfolge r für Satz 2 bleibt in jedem Fall eingefroren. Selbst
prädiktables Nachlernen von r ergab 0,662 Falschzertifikate (07 F2).

### 5.3 Prädiktor pro Bild (07 F15 [N])

Fassung 1 schlug vor:

    f_i^N = Σ_{r ∈ i} q_r + m̂_i,    f_i^A = Σ_{r ∈ i, r ⊂ A} q_r,
    m̂_i   = (N̂ − Σ_r q_r) · w_i / Σ_k w_k,   w_i = 1 − s̄_i   (s̄_i: Recovery-Rate in Bild i)

Bewertung:

* **Eingefroren gültig, aber nahezu informationslos.** Jeder eingefrorene Prädiktor ist in Satz 1
  zulässig. Bei ≤ 1 Injektion pro Bild ist s̄_i ∈ {0, 1} aber Rauschen.
* **Richtige Zielgröße:** E[hidden_i] = λ_i · (1 − π_i). Dabei ist λ_i die erwartete Zahl echter
  Defekte in Bild i und π_i die Fangwahrscheinlichkeit im Kontext von Bild i. π_i kommt aus dem
  geglätteten POD-Modell (§3.1) mit Bildkontext-Kovariaten (Textur, Reflexe, Sättigung), gefittet
  über alle Injektionen. λ_i kommt aus einem Prävalenzmodell. Beides ist modellbasiert. Eingefroren
  kostet ein Fehler nur Power.
* **Der Skalar N̂ ist für die Effizienz irrelevant.** Eine Regressions-(GREG-)Kontrollvariate mit
  (Σ q_r, w_i) absorbiert jedes skalare N̂ exakt (07 §6). Eine nahezu konstante Verschiebung c in f,
  wie sie das gleichmäßige Verteilen von N̂ erzeugt, erhöht unter nicht-uniformen
  Inklusionswahrscheinlichkeiten die Varianz näherungsweise um c² · Σ_i (1/π_i^audit − 1).
* **Der Effizienzwert ist umstritten** [U, 07 §6]. Der Erstprüfer nennt höchstens 8–16 % Varianzgewinn
  durch die Härtekarte. Der Gegenprüfer findet mit genauer Karte 13–33 % (15 versteckte Misses) bzw.
  29–58 % (60–150) im Schätzregime. Im Nullfund-Zertifizierungsregime gibt es keinen Gewinn, weil
  uniform dort minimax-optimal ist (07 §5.2). Für ζ bei seltenen Defekten ist genau dieses Regime
  maßgeblich.

---

## 6. Fallstricke

1. **Synthetik-Leck.** Das Ensemble wurde mit Residual-Paste derselben Defektbank trainiert. Injiziert
   man dieselben Defekte, findet es sie leichter als echte. π wird überschätzt, N unterschätzt.
   Gegenmittel:
   (a) *cross-fitted Injection*: Für Member k nur Defekte aus Fold k injizieren, die Member k nie
   gesehen hat. Die Ensemble-Entscheidung wird über die Member-Scores mit OOF-Schwellen approximiert.
   (b) Eine reservierte Injektionsbank aus L_cal, also Defekte, die in keine Synthese eingehen (vgl.
   die L_cal/L_train-Trennung in 05).
   (c) Defekte aus *anderen* Aufgaben bzw. öffentlichen Datensätzen als „fremde“ Injektionen.
   **Korrektur gegenüber Fassung 1:** Der Anpassungstest deckt ein verbleibendes Leck nur auf, wenn es
   Anteile verschiebt (Logit-Leck: Ablehnung 0,81 bzw. 1,00). Ein proportionales Leck, bei dem die
   Synthetik für alle Detektoren gleichmäßig leichter ist, bleibt unsichtbar (0,04). Gepflanzte Fehler
   sind nachweislich leichter als echte (Kansal 2026; Mills-Seeding; 07 §3.1). Daher ist τ > 0 zu
   erwarten. Echte transplantierte Defekte sind synthetischen Seeds vorzuziehen (07 Anhang A, M10).
2. **Abhängigkeit durch Design.** Wird die Anomaliekarte als Eingangskanal in das U-Net gespeist
   (01-R4), sind A und B nicht mehr annähernd bedingt unabhängig. Das Detektorpanel wird deshalb auf
   *Unabhängigkeit* ausgelegt, nicht nur auf Genauigkeit. Mindestens ein Detektor bleibt vom Labeler
   informationell getrennt: anderer Backbone, anderes Lernprinzip, keine gemeinsamen Eingaben außer
   dem Bild. Das ist eine Heuristik, die das Risiko positiver Restabhängigkeit senkt. Eine Garantie
   ist sie nicht, denn für J = 2 ist Unabhängigkeit untestbar (§3.2, §3.3).
3. **Fehlalarme.** Ohne q_r würde jede Normalitäts-Reflexion als „von A übersehener Defekt“ zählen. q_r
   muss bedingt auf (x, h) kalibriert sein (§3.1). Gerade B-only-Regionen, wo der Stacker die wenigsten
   Trainingsdaten hat, sind kritisch (Szenario C). Nachkalibrierung über das Audit ist nur prädiktabel
   oder cross-fitted zulässig (§5.2).
4. **Regionsabgleich.** Kandidaten verschiedener Detektoren müssen demselben Defekt zugeordnet werden,
   über Überlapp mit Toleranz und `merge_radius` wie in `metrics.py`. Mehrdeutige Zuordnungen (ein
   großer B-Kandidat überdeckt zwei A-Komponenten) werden als ein Defekt gezählt. Diese Konvention muss
   in Schätzer und Metrik identisch sein und wird vor dem Audit eingefroren.
5. **Abhängigkeit innerhalb eines Bildes.** Für das *Audit* mit dem Bild als Stichprobeneinheit berührt
   die Abhängigkeit zwischen Komponenten desselben Bildes die Gültigkeit nicht. Für den Anpassungstest
   und die HT-Varianz gilt das nicht. Der χ²-Test überschreitet sein Niveau (bis 0,089 bei nominal
   0,05), und die Varianz wird unterschätzt. Abhilfe: bildcluster-robuster Test und Bild-Bootstrap
   (§3.1, §4.3). Fassung 1 hatte die Aussage ohne diese Einschränkung formuliert.
6. **Verzerrtes L** (07 A12). Echte Nutzer wählen „schöne“ Beispiele. Diese Wahl speist die
   Injektionsbank (Annahme T), die Fingerprint-Regeln und den Stacker. Fehlende Defektarten in L sind
   der klarste Fall einer T0-Verletzung (τ > 0 für diese Art). Fängt kein Detektor die Art, kann der
   Anpassungstest sie nicht sehen.

---

## 7. Designprinzip (eingeschränkt)

Fassung 1 formulierte: „Ein Auto-Labeler soll seine eigene Unvollständigkeit messbar machen.“ Nach dem
Red-Team gilt eine schwächere Fassung:

> **Messbar macht die Unvollständigkeit nur eine Inspektion nach Protokoll P, also das Audit.** Ein
> Panel von Detektoren mit möglichst unkorrelierten Fehlern und Injection–Recovery können sie
> *vorhersagen* und damit Review und Auditplanung lenken. Belegen können sie sie nicht.

Für die Ensemble-Konstruktion folgt daraus eine **Hypothese**, kein Befund: Neben OOF-Genauigkeit zählt
die Fehlerkorrelation zwischen Labeler und Kontrolldetektoren. Ein schwächerer, aber unabhängiger
Kontrolldetektor könnte für die Review-Lenkung wertvoller sein als ein stärkerer, korrelierter (§8,
H4). Weil Unabhängigkeit für J = 2 aus den Daten nicht prüfbar ist, wird sie auf GT-Datensätzen
gemessen.

---

## 8. Validierungsplan (falsifizierbar)

Datensätze mit vollständiger U-Ground-Truth: MT, BSData, DeepCrack, CFD (verfügbar); lokal KSDD2,
MVTec AD 2 (`test_public`), VisA. Dazu L = 15, 3 Seeds, 3 Labelbudgets.

**Querschnittsvorgaben:**

* Jede Hypothese wird zusätzlich **unter verzerrtem L** geprüft (07 A12): Top-k nach Fläche bzw.
  Kontrast, 1–2 fehlende Defektarten, eine einzige Charge.
* Schwellen, Zellgrenzen, τ- und OR-Bereiche werden vor den Läufen registriert.
* N̂-Ergebnisse erscheinen nur im Forschungsprotokoll.
* Gültigkeit wird nur mit Satz 1, Prop. 4 bzw. den exakten Schranken aus CertAL-lite gemessen, nicht mit
  `estimate_ratio`.

Herkunft der Schwellen: H1, H2a (80 %) und H5 stammen aus Fassung 1, H3 und H6 aus den Go-Kriterien
Nr. 7 und 8 in 07 §6. Die Niveaugrenze in H2a sowie H2b und H7 sind Vorschläge dieser Revision. Keine
dieser Hypothesen ist bisher geprüft worden.

| Hypothese | Messung | Verworfen, wenn |
|---|---|---|
| **H1** Die Vollständigkeitsschätzung schlägt naive Zählungen | MAE der geschätzten A-Misses (N̂ − N̂_A) gegen die wahre Zahl auf U, für N̂_HT, N̂_HG und N̂_Bell. Naive Baselines: (i) die rohe q-gewichtete Fangzahl Σ_{r ⊄ A} q_r der anderen Detektoren (wie bei Kansal 2026), (ii) Σ `exp_missed` aus `quality.py`. Beschreibend dazu: Vorzeichen des Bias und Anteil der Läufe, in denen das Band N̂(τ) die wahre Zahl enthält | keiner der drei Schätzer senkt den MAE um ≥ 25 % gegenüber der *besseren* der beiden Baselines |
| **H2a** Anteilsverschiebungen werden erkannt | absichtlich verschlechterte Injektion, die Anteile verschiebt: falsche Gain-Verteilung, nur kleine Defekte, Leck (Trainingsdefekte injiziert). Bildcluster-robuster Test mit Multiplizitätskontrolle (§4.3) | Ablehnung in < 80 % der verschlechterten Fälle, oder in > 2α der korrekten Fälle (Niveauverletzung) |
| **H2b** Proportionale Abweichungen bleiben unentdeckt, kosten im Produktpfad aber nur Power | proportionale Abweichung: Synthetik für alle Detektoren gleichmäßig leichter, bzw. eine Defektart nicht in der Injektionsbank. Gemessen werden die Power des Tests (erwartet ≈ α), die Falschzertifikatsrate und n₉₀ des Audits mit aktiven Rollen (a) und (b) | Falschzertifikatsrate > δ (plus Monte-Carlo-Fehler) in irgendeiner Konfiguration. Liegt die Power deutlich über α, ist das Blindheitsmodell falsch; das ist zu erklären, verwirft H2b aber nicht |
| **H3** Vollständigkeitsprädiktoren senken n₉₀ bzw. die LCB-Breite bei gleicher Gültigkeit | Satz 1/Prop. 4 (prädiktable Wettobergrenzen) bzw. PPR. Arme: f aus §5.3 (Σ q + Härteprädiktor) vs. f = Σ q vs. kein f, über ein Raster von n_audit. Falschzertifikatsrate am Rand | n₉₀ bzw. LCB-Breite sinkt um < 15 % gegenüber f = Σ q (entspricht Go-Kriterium Nr. 7), oder Falschzertifikatsrate > δ. Erwartung nach 07 §5.2: kein Gewinn im Nullfund-Regime für ζ, höchstens im Schätzregime |
| **H4** Unabhängigkeit schlägt Stärke | Panel A + B mit B = DINO-kNN (unabhängig) vs. B = zweiter U-Net-Member (stärker, korreliert). Primärmaß: gefundene echte Misses pro Review-Minute mit B als Risikomerkmal (07 A16). Ergänzend: Fehler aus H1, wahres OR_real auf GT | der korrelierte Partner liefert mehr echte Misses pro Review-Minute |
| **H5** Die Positivitätslücke ist klein | Anteil echter U-Defekte, die *kein* Detektor findet, pro Größenklasse. Dazu τ̂(x) = 1 − π̂_real(x)/π̂_syn(x) auf GT als empirische Grundlage für den τ-Bereich in §2.2 | > 10 % in einer Größenklasse: Das Panel braucht dort einen weiteren Detektor |
| **H6** Die Härtekarte lenkt das Review (Go-Kriterium Nr. 8, Rolle (b)) | Anteil der von allen Detektoren übersehenen echten Misses in den Top-20 % der Bilder nach Härte | < 50 %: Rolle (b) entfällt |
| **H7** Die K′-Prognose ist konservativ genug (Rolle (a)) | wahres K′ gegen K̂′ (Punktwert und oberes Bandende). Realisierte Power des mit K̂′ geplanten fixen n | realisierte Power < 0,8 bei Planung auf 0,9 in mindestens einer Datensatz-Konfiguration |

---

## 9. Einordnung und Neuheit

**Vorarbeiten.** Das Urteil der Neuheitsprüfung in 07 §4 zu C1 lautet „teilweise bekannt“
(gegenverifiziert [B]):

* **Capture–Recapture mit Kovariaten:** Huggins (1989, *Biometrika*), Alho (1990, *Biometrics*).
  Log-lineare Multiple-Systems-Schätzung: Fienberg (1972, *Biometrika*), Bishop/Fienberg/Holland
  (1975). Chapman-Korrektur: Chapman (1951). Software-Inspektion: Eick et al. (1992, ICSE), Briand et
  al. (2000, TSE). Korrektur des χ² unter komplexem Design: Rao & Scott (1981, JASA). Diese Verweise
  stammen aus Vorwissen und sind in keiner Session geprüft **(unverified)**.
* **Fehler-Seeding:** Mills (1972, IBM FSD). Gepflanzte Fehler werden leichter gefunden als echte.
* **Injection–Recovery in der Astronomie:** Kepler DR25 (Thompson et al., ApJS 235:38, 2018); Balrog für
  DES (Suchyta et al., MNRAS 2016).
* **Probability of Detection:** MIL-HDBK-1823A, MAPOD. Hit/Miss-GLM in log(Größe) als parametrische Form
  von π(x). a90/95 aus Injektionen ist modellbasiert (07 M6).
* **Daten-Cleaning:** DQM (Chung, Krishnan, Kraska, PVLDB 10, 2017). Die Zahl unentdeckter Fehler in
  Datensätzen wird aus überlappenden Funden mehrerer Prüfer geschätzt.
* **Technology-Assisted Review:** Chao-basierte Stoppregeln (Bron et al., ACM TOIS, 2024/2025).
* **Labelfehlerdetektoren:** Kansal 2026 (doi:10.5281/zenodo.23205993), Capture–Recapture über mehrere
  Labelfehlerdetektoren. Der MAE war dort schlechter als die Rohzählung. Das ist eine direkte Warnung
  für H1 und der Grund für die naive Baseline Σ q_r.
* **Dual-System-Schätzung im Zensus:** Bell (1993, JASA 88:1106), entspricht dem Hybrid aus Fassung 1;
  Wachter & Freedman, „The Fifth Cell“ (Eval. Rev. 24:191, 2000); Das, Kennedy, Jewell (JASA 119:1309,
  2024, `drpop`), das weiterhin CI gegeben x voraussetzt.
* **Nicht-Identifizierbarkeit:** Link (2003, Biometrics 59:1123).
* **Fehlende Labels in der Segmentierung:** Rottmann & Reese (WACV 2023, arXiv:2207.06104); Penquitt et
  al. (arXiv:2609.21822); REC✓D (arXiv:2508.06556, verpasst bis zu 66 %; 07 M9).

Zur Quellenlage: Die Angaben aus 07 §4 beruhen auf Suchmaschinen-Abstracts und -Snippets sowie auf Code
auf GitHub, darunter dem Code zu Kansal 2026, der gelesen wurde. arXiv, OpenReview und die meisten
Verlagsseiten waren in jener Session nicht abrufbar. In dieser Revision wurden die Angaben nicht erneut
geprüft.

**Neuheit.** Fassung 1 beanspruchte als neu die Kombination aus (a) Injection, die die Zelle „von allen
übersehen“ beobachtet, (b) einem GT-freien Test der Transportannahme, (c) einem PPI/IPW-Audit, das „das
Ganze annahmefrei gültig“ macht, und (d) einem Panel, das nach Unabhängigkeit gebaut ist. Nach 07 gilt:

* (a) ist bekannt (Mills, Kepler, Balrog, POD).
* (b) ist überzogen. Es handelt sich um einen Anpassungstest der beobachtbaren Implikation, blind für
  Defekte, die kein Detektor sieht (§4).
* (c) ist falsch formuliert. Das Audit ist unter den Bedingungen aus §5.2 für sich gültig, C1 ist dafür
  redundant, und `estimate_ratio` ist nur approximativ.
* (d) ist eine Konstruktionsheuristik, deren Grundannahme für J = 2 untestbar ist.

Umformulierter Beitrag (07 §4): **„Übertragung etablierter Vollständigkeitsmethoden (Seeding,
Injection–Recovery, Capture–Recapture mit Kovariaten) auf Pixel-Auto-Labels industrieller Defekte,
gekoppelt an ein designbasiertes Audit.“** Möglicherweise neu ist der GT-freie Vergleich vorhergesagter
beobachtbarer Zellanteile. Er ist ein Anpassungstest, kein Vollständigkeitstest. Die Suche war nicht
erschöpfend.

---

## Änderungsprotokoll Fassung 1 → Fassung 2

Die Abschnittsnummern von Fassung 1 bleiben erhalten, damit Verweise aus 00 und 07 weiter stimmen. Neu
ist §2.2 (Identifikation). §3.4 und §3.5 sowie die Unterabschnitte von §4 und §5 sind neu gegliedert.

| Vorgabe aus 07 | umgesetzt in |
|---|---|
| §7 Nr. 7 (S3-Fehler nur unter Bedingungen effizienzneutral; Floor als Gültigkeitsbedingung) | §5.2 |
| §7 Nr. 9 (Anpassungstest statt Transporttest, Nicht-Identifikation, Ausweichregel gestrichen, Sensitivitätsband, Clusterrobustheit, Multiplizität) | §2.2, §4 |
| §7 Nr. 18 (Bell-Schätzer, Chapman, OR-Sensitivität, Huggins-Bias beidseitig, J = 3 gesättigt, `drpop`-Hinweis) | §3.2, §3.3 |
| §7 Nr. 19 (Garantie auf Satz 1 + Prop. 4 beschränkt; `estimate_ratio` approximativ; IPW-Verweis entfernt; Nachkalibrierung) | §5.2 |
| §7 Nr. 20 (HT-Bedingungen, Fehlalarm-Varianzterm, π̂ geglättet mit Floor, POD/MAPOD, a90/95 modellbasiert) | §3.1 |
| §7 Nr. 21 (Validierungsplan: naive Baseline Σ q_r, H2 geteilt, H3 auf n₉₀/LCB-Breite, verzerrtes L) | §8 |
| F3 / F9 / F13 / F14 / F15 / F5 / F8 | §2.2 + §4 / §3.5 / §4.3 + §6.5 / §3.1 / §5.3 / §5.2 / §3.3 |
| Neuheitsurteil C1 (07 §4), Rolle von C1 (07 §6) | §9, Kopfbox, §5.1 |
| Formulierung „keine etablierte Methode“ (07 §7 Nr. 14, hier zur Konsistenz) | §1 |

Gestrichen aus Fassung 1:

* die Triangulation als Diagnosesignal;
* die automatische Ausweichregel;
* der Fit-Status im Qualitätsbericht;
* „Gültigkeit hängt nur vom Design ab“;
* „reiner Effizienzgewinn ohne Gültigkeitsrisiko“;
* „vermutlich robust“ (Hybrid);
* „Annahme ist dann nur noch, dass die Dreifach-Interaktion null ist“;
* der Verweis auf eine IPW-Variante von `estimate_ratio`.
