# 07 – Red-Team-Bericht zu CertAL (00, 04, 06)

Stand 19.10.2026. Geprüft wurden [00](00_SYNTHESE.md) (CertAL, S1–S5, C1–C7, Roadmap),
[04](04_guarantees_measurement.md) (Protokoll CALS) und [06](06_completeness_theory.md)
(Vollständigkeitsschätzung), dazu der Code in `autolabel/` und die Simulationen in [`sims/`](sims/).
Die Prüfung lief in zwei Durchgängen. Durchgang 1 bestand aus den Skripten `redteam_capture_recapture.py`
und `redteam_audit_checks.py`. Durchgang 2 hatte vier Linsen: Mathematik, Neuheit, Machbarkeit und
Alternativen. Jeder Befund aus Durchgang 2 wurde anschließend von einem zweiten Prüfer gezielt zu
widerlegen versucht.

**Legende der Marker**

* **[B]** bestätigt: Der Gegenprüfer hat versucht, den Befund zu widerlegen, und ist gescheitert.
  Abweichungen im Detail sind jeweils angegeben.
* **[U]** umstritten: Der Gegenprüfer hat Teile widerlegt oder abgeschwächt. Beide Argumente werden
  genannt.
* **[N]** nicht gegenverifiziert: Es gibt nur die Evidenz des Erstprüfers (Rechnung oder Simulation),
  keine adversariale Zweitprüfung.
* **[R]** in diesem Bericht nachgerechnet (exakte hypergeometrische Rechnung, Skript im Scratchpad).

Alle Zahlen stammen aus echten, kleinen Monte-Carlo-Läufen oder aus exakten Rechnungen auf
**synthetischen Pools**. Auf echten Daten ist nichts davon gemessen. Skripte in [`sims/`](sims/):

* aus Durchgang 1: `redteam_capture_recapture.py` und `redteam_audit_checks.py`;
* Erstprüfer Durchgang 2: `redteam2_betting_validity.py`, `redteam2_transport_identifiability.py` und
  `redteam2_satz2_tiles.py`;
* Gegenprüfer Durchgang 2: `redteam2_verify_math_checks.py`, `redteam2_verify_feasibility_checks.py`,
  `redteam2_verify_alternatives_checks.py`, `redteam2_verify_novelty_ltpd_prop3.py` und
  `redteam2_verify_novelty_tiles.py`.

Die Simulationen der Linsen Machbarkeit und Alternativen (`feas1–4.py`, `lite_zeta.py`,
`lite_recall.py`, `c1_value.py`, `n90_table.py`, `fixedn_vs_seq.py`, `run_ppr.py`, `p1_lin.py`) liegen
mit dem Präfix `redteam2_lens_` ebenfalls in [`sims/`](sims/), ihre Ausgaben in
[`sims/outputs/`](sims/outputs/).

---

## 1. Kurzfazit

* **Gültigkeit setzt „Mensch = Gold“ voraus, und CertAL verletzt diese Annahme selbst.** 00 S4 schreibt
  verify-first für *jede* geprüfte Einheit vor. Ein Reviewer mit 70 % Sensitivität erzeugt
  0,135 Falschzertifikate bei n=400 und 0,330 bei n=1000 (nominal 0,05). Der Fehler wächst mit n.
  Billige Korrekturen über Seed-Defekte oder einen kleinen Doppelannotationsanteil stellen die
  Gültigkeit nicht wieder her. [B]
* **Das Review-Stratum R liegt außerhalb jeder Messung.** Restfehler in R gehen nicht in das
  Zertifikat ein (U′ = U∖R). In einem Beispiel mit hoher Prävalenz ist *jedes* ausgestellte
  Zertifikat falsch (Rate 0,55 = Zertifizierungsrate). [B]
* **Satz 2 ist nur bei eingefrorenem r gültig.** 04 §4.5(i) erlaubt es aber, die Rangfolge aus dem
  Audit zu lernen. Bei nachgelerntem r sind 100 % der Zertifikate falsch, selbst bei *prädiktablem*
  Nachlernen sind es 66 %. Zweiphasig gelernt oder eingefroren ist es gültig (≤ 0,001). [B]
* **Der Recall-Schranke liegt eine unprüfbare Populationsannahme zugrunde, die harte Schranke D_max.**
  Ohne sie gibt es keine nichttriviale endliche LCB für Zähl-Recall. Eine einzige nie gezogene
  Einheit mit 100 Misses führt zu 0,71 Falschzertifikaten. „D_max ≈ 1–2 pro Kachel“ ist ein
  typischer Wert, keine Schranke. [B]
* **Der Code, der die Betting-Fieller-Spalte S1 erzeugt hat, ist ungültig.** Die Wettobergrenze ist
  nicht prädiktabel. Auf einem adversarialen Pool liegt der Fehler 1. Art bei 0,63–0,99, mit
  korrekter Obergrenze bei 0,000. [B]
* **Die exakte hypergeometrische Schranke ist nicht jederzeit stoppbar.** Bei Überwachung ergeben
  sich 0,168 Falschzertifikate. Die PPR-Martingale ohne Zurücklegen ist ein gültiger Ersatz. [B]
* **Der GT-freie Transporttest prüft etwas anderes, als N̂_HT braucht.** N̂_HT nutzt die echten Daten
  nur über n_obs(x). Die tragende Annahme T0, P_real(0|x) = P_syn(0|x), ist untestbar. Die
  automatische Ausweichregel auf Huggins kann den Bias von 0 auf +0,83 treiben. Übereinstimmung der
  drei Schätzer ist kein Beleg (0,89 Übereinstimmung bei −30 % Bias). [B]
* **Keiner der Beiträge C1–C7 ist in der behaupteten Form neu.** PAC-Labels korrigiert und
  zertifiziert bereits den korrigierten Pool. Rectifying Inspection (Dodge–Romig), TAR-Zertifizierung,
  Kartenvalidierung, Dual-System-Schätzung (Bell 1993) und NSA decken Kernteile ab. Neu bleibt die
  Übertragung auf pixelgenaue industrielle Auto-Labels sowie einige Details (geschlossene Form von
  Prop. 3, Ignore-Band). [B]
* **Für die Primärmetrik ζ bringt die volle S4-Maschinerie fast nichts.** Gemeint sind
  Prop.-3-π, PPS, PPI und IPW-Betting mit Zurücklegen. Gegenüber „Top-k-Review + uniformes Audit
  ohne Zurücklegen + exakte Schranke“ gewinnt sie 0–3 % und kostet 12–45 % mehr Inspektionen. Bei
  null Funden ist uniform minimax-optimal. [B]
* **Hoher Defekt-Recall ist bei seltenen Defekten nicht zertifizierbar.** Bei 1 % Prävalenz müssen
  für Recall ≥ 0,9 53–95 % der sauberen Einheiten inspiziert werden (ein Miss pro Einheit bzw.
  gepackt bei D_max = 6). Ein absolutes ζ-Ziel ist leer, sobald ζ* ≥ p: Ein
  Labeler ohne Ausgabe besteht es. [B]
* **Die Nordstern-Metrik ist manipulierbar und kostenmodellabhängig.** Bei einseitigem Ziel erreicht
  eine Vollvordergrund-Maske ζ = 0 und Recall = 1. Die dominierende Kostenart, die Vollinspektion,
  hat in 05 §3.1 keinen Anker. [B]
* **Was trägt:** der Kern von Satz 1 (IPW-Betting mit prädiktablen Größen), Prop. 4
  (Fixed-Sequence), die Fieller-Äquivalenzen, die geschlossene Form von Prop. 3, Lemma 0 auf
  Bildebene bei voller Gold-Korrektur und die Grundidee „Design kauft Gültigkeit“. Getragen wird
  sie von einer deutlich schlankeren Variante, **CertAL-lite** (§6). Diese ist nach Einschätzung der
  Machbarkeitslinse in 3–5 Wochen baubar (Schätzung, ungemessen).

---

## 2. Mathematische Prüfung

### 2.1 Was korrekt ist

Folgende Aussagen haben beide Durchgänge geprüft und für korrekt befunden:

* **Satz 1, Kern.** Unter Ziehung mit Zurücklegen, IPW und F_{t−1}-messbarem q_t, f^(t), λ_t gilt
  E[Z_t | F_{t−1}] = μ. Mit β_t ≤ c/(z^max − m), c < 1, bleibt jeder Faktor ≥ 1 − c > 0. Der Prozess
  ist unter μ ≥ m ein Supermartingal, und Villes Ungleichung greift. Eine Mischung über ein β-Gitter
  mit festen Gewichten ist zulässig.
* **Prop. 4.** Ablehnungen dürfen zu verschiedenen Zeiten erfolgen, und es ist keine
  Multiplizitätskorrektur nötig, weil {∃t: ρ⁰ < LCB_t} ⊆ {∃t: K_t(r_{j*}) ≥ 1/δ} gilt.
* **Weitere korrekte Teile:**
  * die Fieller-Äquivalenzen für Recall, Precision und Dice;
  * die CRC-Algebra in Prop. 2, einschließlich der Reduktion auf s_(⌊α(n+1)⌋) bei D ≡ 1;
  * der deterministische Schritt in Prop. 1;
  * Lemma 0 auf Bildebene bei vollständigem Gold-Ersatz Ỹ_i = Y_i;
  * die Beta(11,1)-Zahlen.
* **Exakte Zahlen.** Die Zahlen 188/57/18 in 04 S4 (Bestätigung [R]) und die geschlossene Form von
  Prop. 3 stimmen. Gegen SLSQP auf 20 Instanzen liegt die relative Zielfunktionslücke bei ≤ 1e−13.

### 2.2 Ergebnisse aus Durchgang 1

**Capture–Recapture (06).** Parameter: N_real = 300, 600 Injektionen, 2000 Wiederholungen,
Testniveau 0,05.

| Szenario | Ablehnrate Transporttest | rel. Bias N̂: HT / HG / Hybrid | „Übereinstimmung“ | Recall A geschätzt vs. wahr |
|---|---|---|---|---|
| 0 Null (T und CI gelten) | 0,05 | +0,00 / +0,01 / +0,02 | 0,93 | 0,71 vs. 0,71 |
| A 30 % für beide Detektoren unsichtbar | 0,05 | −0,30 / −0,29 / −0,28 | 0,89 | 0,71 vs. **0,50** |
| B gemeinsame Schwierigkeit, sd 1,5 logit | 0,57 | −0,08 / −0,17 / −0,17 | 0,77 | 0,73 vs. 0,67 |
| B′ dto., sd 3 logit | 1,00 | −0,16 / −0,29 / −0,29 | 0,09 | 0,75 vs. 0,62 |
| C 300 falsche B-only-Regionen, q = 0,1 fehlkalibriert | 0,69 | +0,14 / +0,29 / +0,31 | 0,46 | 0,62 vs. 0,71 |
| D Kovariate x der B-only-Fänge zu groß gemessen | 0,57 | −0,01 / −0,05 / −0,03 | 0,98 | 0,72 vs. 0,71 |

**Audit (04).**

| Prüfung | Ergebnis |
|---|---|
| (a) Prop. 3 geschlossen vs. SLSQP | Lücke ≤ 1e−13, korrekt |
| (b) Optional Stopping, wahres ζ = 0,0205 > 0,02 | fixes n = 600 hypergeometrisch 0,042; **alle 20 Ziehungen überwacht 0,168**; Betting-CS 0,022 |
| (b) wahres ζ = 0,005 | fix 1,000; überwacht 1,000 (Median-Stopp 220); Betting 0,932 (Median 339) |
| (c) Kacheln vs. Bilder, Recall-LCB (wahr ≈ 0,91) | 10 % Fläche: 0,384 → 0,624; 30 %: 0,650 → 0,777; geclustert 0,371 → 0,527 bzw. 0,630 → 0,744; Abdeckung 1,00 |
| (c) Heuhaufen, k = 10 Misses ohne Signal | nötiger Anteil 1 − δ^{1/k} = 0,259 für Bilder *und* Kacheln |
| (d) Satz 2 | r eingefroren: 0,00 Zertifikate; r aus demselben Audit nachgelernt: 1,00 Zertifikate, alle falsch |
| (e) Mensch ≠ Gold, ζ ≤ 0,025 bei wahrem 0,03 | Gold n=400: 0,011, n=1000: 0,002; verify-first mit s = 0,7: **0,135 bzw. 0,330** |

### 2.3 Befunde zu 04

**F1 – Designbasierte Gültigkeit braucht eine harte Schranke für *alle* Einheiten.** [B, Schwere hoch
für 04, kritisch für die Formulierung in 00]

Satz 1 und Prop. 4 verlangen Z_t ≤ z^max für jede ziehbare Einheit. Die Schranke b = r·D_max ist eine
Aussage über *unbeobachtete* Misses.

* **Gegenbeispiel** (`redteam2_betting_validity.py` (2)): N = 2000, ein einziges Bild mit 100
  übersehenen Mikrodefekten, wahres ρ = 0,792, registriertes D_max = 6. Das falsche Zertifikat
  „ρ ≥ 0,80“ wird in 0,707 der Läufe ausgestellt (1500 Ziehungen), bei 3000 Ziehungen in 0,737. Das
  schwere Bild wurde in 0,447 bzw. 0,250 der Läufe nie gezogen, und in all diesen Läufen war das
  Zertifikat falsch. Ein Abbruch bei beobachteter Überschreitung hilft daher nicht.
* **Unmöglichkeitsaussage** (vom Gegenprüfer exakt gemacht): Ohne Schranke gilt für jede gültige
  LCB des Zähl-Recalls auf *jeder* Population
  P(LCB > ε) ≤ δ·N/(N − n).
  Für N = 2000 sind das 0,059 bei n = 300 und 0,100 bei n = 1000. Es gibt also keine nichttriviale
  LCB.
* **Physikalisch sichere Kachelschranke.** Sie ist leer: (256/5)² ≈ 2601 Komponenten pro Kachel
  erfordern ≥ 9,2·10⁵ Ziehungen (Jensen-Schranke).
* **Abschwächung.** 04 §4.2 führt die Beschränktheit y_i ∈ [a, b] im G-D-Tabelleneintrag bereits
  als Annahme. Falsch sind (i) die Prosa „Gültigkeit kommt allein aus der Randomisierung“, (ii) §6,
  das eine Überschreitung als *entdeckbar* behandelt, und (iii) 00 („nie falsch“, „D_max ≈ 1–2“ als
  Eigenschaft).

*Korrigierte Aussage (Satz 1′).* Gültig unter folgenden Voraussetzungen:
(i) bekanntes q_t ≥ γ/N′ > 0;
(ii) eine **vorab registrierte harte Schranke** y_i ∈ [a, b] für alle i ∈ U′, als Modellannahme,
die das Design nicht prüfen kann;
(iii) Gold-Messung bzw. Messung nach Protokoll P;
(iv) eine eingefrorene Zielgröße;
(v) prädiktable Wettobergrenzen (F4).
Eine *beobachtete* Überschreitung macht das Zertifikat ungültig, ihr *Ausbleiben* belegt nichts.

Zielgrößen ohne unprüfbare Zählschranke:
* ζ (b_i ∈ {0, 1});
* Precision bzw. φ mit der bekannten Schranke F_i ≤ P_i aus Ŷ;
* explizit gekappter Recall mit min(M_i, D_max), so beschriftet;
* ergänzend eine designbasierte UCB auf den Anteil der Einheiten mit D_i > D_max.

**F2 – Satz 2 verlangt, dass r F₀-messbar ist.** [B]

Der Beweis braucht eine feste Funktion A(u). Ist r datenabhängig, werden die Nullhypothesen zufällig.

* **Nachgelerntes r:** 1,000 Falschzertifikate (`redteam2_satz2_tiles.py` (1)).
* **Prädiktables Nachlernen** (Gegenprüfer, `redteam2_verify_math_checks.py` A): Beta-Posterior der
  Missrate einer Gruppe G, Budgetstopp bei 600 Ziehungen, Review mit r^(600). Ergebnis: **0,662**
  Falschzertifikate. Der Grund ist, dass die abgelehnten Nullhypothesen r^(s) für s < τ betreffen,
  das Review aber r^(τ) verwendet.
* **Gültige Varianten:**
  * zweiphasig: r auf den Ziehungen 1..t₀ lernen, einfrieren, S_{t₀} in das bekannte Stratum
    verschieben, frische Ziehungen nur aus U″ = U′∖S_{t₀}. Ergebnis 0,000 Falschzertifikate,
    Power 0,577;
  * eingefrorenes r mit der Identität Σ_{U′∖S_τ} M·1{r<u} = Σ_{U′} M·1{r<u} − Σ_{S_τ} M·1{r<u}.
    Ergebnis 0,001.
* **Abschwächung:** Der Pseudocode in 04 friert r in Stufe 0 bereits ein. 04 ist also inkonsistent,
  nicht durchgängig falsch. Der Widerspruch liegt bei §2.1, §4.5(i) und 00 S3.

*Korrigierte Aussage (Satz 2′).* Voraussetzung ist, dass vor Ziehung 1 eingefroren und gehasht sind:
r, Ŷ einschließlich der Wahrscheinlichkeitskarten, die Konventionen (tol, merge_radius,
Kachel-Heimatregel), U′, R, Gitter und Testreihenfolge, ε, die δ-Aufteilung und D_max. Nur die
Familie {r ≥ u} zählt, streng monotone Transformationen sind also erlaubt. Prädiktabel dürfen sein:
q_t (mit Floor), f^(t), λ_t, β_t und die Stoppzeit, **nicht** r.

**F4 – Die Wettobergrenze in `sims/sim_recall_cert.py` ist nicht prädiktabel.** [B für die
Hauptaussage]

Zeile 78 berechnet z^max aus q und f der *aktuellen* Ziehung i_t. Daraus folgt

E[1 + β_t(c − Z_t) | F_{t−1}] = 1 + Σ_i q_i β(i)(c − Z(i)),

und dieser Ausdruck kann unter H0 größer als 1 sein. Auf dem adversarialen Pool aus 04 §4.10 (Misses
am Floor-q) werden die *wahren* Hypothesen H(0,80) und H(0,82) fälschlich abgelehnt:

| | n = 300 | n = 1000 |
|---|---|---|
| Obergrenze pro Ziehung (aktueller Code) | 0,628 / 0,506 | 0,989 / 0,969 |
| prädiktable Obergrenze | 0,000 | 0,000 |

Die Drift beträgt analytisch +0,0103 pro Schritt. Die Abdeckung 1,00 in S1 belegt deshalb nur
Konservativität auf einem gutartigen Pool, keine Gültigkeit.

Korrekte Obergrenze:
z^max_t = λ_t F̄^(t) + max_{i∈U′} (b − λ_t f_i^(t))/(N′ q_t(i))
≤ λ_t F̄^(t) + (b − λ_t min_i f_i^(t))/γ.

**F6 – Die hypergeometrische Schranke gilt nur bei fixem n und uniformem q.** [B]

04 §4.5 nennt diese Bedingungen, 00 S4 hat sie gestrichen. Unter Überwachung entstehen 0,168
Falschzertifikate.

Gültiger jederzeit stoppbarer Ersatz ist die Posterior-Prior-Ratio-Martingale (PPR) für Ziehen ohne
Zurücklegen (Waudby-Smith & Ramdas 2020, arXiv:2006.04347) mit Beta-Binomial(1,100)-Prior:
* 0,031 Falschzertifikate am Rand;
* ein wahres ζ = 0,005 wird in 0,995 der Läufe zertifiziert (Median-Stopp n = 300).

Bei uniformer Ziehung mit Zurücklegen bilden die Erstauftritte verschiedener Einheiten eine
Stichprobe ohne Zurücklegen. PPR ist damit anwendbar, wenn der uniforme Strom separat gelost und
protokolliert wird.

**F10 – Die Heuhaufenformel in N5 ist der günstigste Fall.** [N, Zahlen [R]]

* Packt man k = εD_tot + 1 Misses in ⌈k/D_max⌉ Einheiten ohne Signal, steigt der nötige Anteil
  auf 1 − δ^{1/⌈k/D_max⌉}. Für k = 10 sind das 0,259 (D_max = 1), 0,451 (D_max = 2) und 0,776
  (D_max = 6).
* Für Recall gilt im schlechtesten Fall n_c ≈ N_c·ln(1/δ)·D_max/(ε·D_tot). Die Formel mit D_max = 1
  bleibt für ζ richtig.
* Kleinere Einheiten helfen im schlechtesten Fall also doch, aber nur über ein kleineres *hartes*
  D_max (F1).
* Die Zahl „640“ in 04 §0.8/S4 und 00 §7 ist **624** [R].

**F11 – „Um Größenordnungen schärfer“ ist für beschränkte Betting-CS unmöglich.** [B über C5; die
Mathematik selbst N]

Die Halbbreite ist ≈ σ√(2 log(1/δ)/n) + O(range·log(1/δ)/n). Kacheln ändern den führenden Term nur
um √(Designeffekt) und den Term zweiter Ordnung um das D_max-Verhältnis (≈ 3).

Empirische Faktoren:
* LCB: 1,2–1,6 (Durchgang 1);
* nötige Fläche: 1,1–2,5 (Gegenprüfer, `redteam2_verify_novelty_tiles.py`);
* Recall-LCB bei gleicher *Zeit* statt Fläche (`feas4.py`, [N]):
  * Overhead c₀ = 0 s pro Kachel: Bilder 0,332, Kacheln 0,489;
  * c₀ = 1 s: 0,349 vs. 0,390;
  * c₀ = 3 s: Kacheln verlieren (0,312 vs. 0,240).

Im geclusterten Pool hatte eine Kachel 5 Defekte. „D_max ≈ 1–2“ ist damit auch empirisch nicht
gesichert.

**F12 – Lemma 0 versagt auf Kachelebene.** [N, explizite Gegenbeispiele]

* `redteam2_satz2_tiles.py` (2), lokale Gold-Korrektur nur der Kachel T:
  * Eine U-förmige Falschkomponente mit Heimat T′ zerfällt, F_{T′} steigt von 1 auf 2.
  * Eine GT-Komponente in T′, deren einziger Toleranztreffer in T lag, wird zum Miss, M_{T′} steigt
    von 0 auf 1.
* „Maximum-Pixel“ ist für binäre GT-Komponenten undefiniert.
* Die Überlappungszone muss mindestens Defektausdehnung + merge_radius + tol betragen. Für große
  Defekte wird eine Kachelprüfung damit zur Bildprüfung.
* Teilkorrekturen auf Bildebene („Grenze falsch“) können Komponenten ebenfalls teilen.

**F16 – Implizite Bedingungen in Prop. 4 und Satz 1.** [N]

* Gitter und Reihenfolge sind vorab fixiert.
* Das Gitter wird von unten gescannt. Bisektion ist nur bei monotonen Ablehnungen gültig, und diese
  sind bei r-abhängigen Obergrenzen nicht garantiert.
* Die Gewichte der β-Mischung sind fest. Ein Maximum über das β-Gitter braucht Bonferroni.
* Ist die Ablehnmenge leer, gilt LCB = 0.
* y ist der Wert der Erstinspektion vor der Korrektur.
* `sim3.py` Zeile 28 gibt nr.max() als UCB aus. Das ist um bis zu einen Gitterschritt
  antikonservativ; korrekt ist der Gitterpunkt *über* dem größten nicht abgelehnten.

**F17 – Prop. 3 ist für die Poisson-Zielfunktion optimal, Stufe C zieht aber mit Zurücklegen.**
[N, betrifft nur Effizienz]

* Stufe C zieht mit Zurücklegen nach Hansen–Hurwitz, dort gilt q ∝ |y − λf|. „Optimal“ überträgt
  sich deshalb nicht.
* Prop. 1 (CV+) braucht austauschbare Einheiten. Defekte clustern aber in Bildern, und die Folds
  sind bildweise. Außerdem muss die Nachverarbeitung monoton in der Karte sein, was Top-k- oder
  Größte-Komponente-Filter nicht sind.

### 2.4 Befunde zu 06

**F3 – Identifikation und Testbarkeit.** [U, Kern B]

*Satz (Identifikation).* Pro Kovariatenzelle x identifizieren die echten Daten nur n_obs(x) und die
bedingten Anteile P_real(h | x, h ≠ 0), also 2^J − 2 Parameter. P_real(0|x) und damit N(x) sind nicht
identifiziert.

Es gilt E[N̂_HT] = Σ_x N(x)·π_real(x)/π_syn(x). HT nutzt die echten Daten also nur über n_obs(x) und
ist konsistent, wenn T0 gilt: P_real(0|x) = P_syn(0|x), zellweise. „Genau dann“ ist wegen möglicher
Auslöschung über Zellen zu stark.

Der Test prüft H_obs: P_real(·|x, h≠0) = P_syn(·|x, h≠0). **T0 und H_obs sind logisch unabhängig:**

* Szenario A: H_obs wahr, T0 falsch.
* Neues Szenario E: T0 wahr, H_obs falsch. Ergebnis: Ablehnung 1,00, Bias HT −0,000, HG +0,827,
  Hybrid +0,905. Die Regel aus 06 (bei Ablehnung auf Huggins ausweichen) liefert **+0,827**.
* Power bei realistischer Zahl echter Defekte (Fall B): 0,13 (40 Defekte), 0,28 (100), 0,58 (300).

*Gegenargument des Prüfers:*
* Logit-Lecks werden erkannt: Ein Leck nur bei A (+1 logit) gibt Ablehnung 0,81 bei HT-Bias −0,105;
  ein Leck bei beiden Detektoren gibt Ablehnung 1,00 bei −0,139.
* Unsichtbar ist nur die *proportionale* Abweichung P_real(h|x) = c(x)·P_syn(h|x) für h ≠ 0.
  Dort ist die Ablehnrate 0,04 bei HT-Bias −0,198.
* In beiden Leck-Fällen senkt die Ausweichregel den Bias (−0,105 → −0,006, −0,139 → +0,009), weil
  dort CI gilt.
* Die Behauptung „ein Leck, das alle Detektoren gleich anhebt, liegt in der 00-Richtung“ ist damit
  widerlegt. Die Regel ist also nicht universell falsch, aber ohne explizite Verknüpfungsannahme
  unbegründet. 06 §4, letzter Absatz, räumt die Blindheit für h = 0 bereits ein.
* Neu ist, dass die getesteten Größen in N̂_HT gar nicht eingehen.

**F5 – „Gültigkeit hängt nur vom Design ab“ (06 §5) gilt nicht für `ppi.estimate_ratio`.** [B für (i),
U für (ii)]

(i) Der Schätzer nutzt Delta-Methode, t-Quantil und SRS-FPC und ist damit nur asymptotisch gültig.
Abdeckung der 95-%-LCB beim Gegenprüfer:

| Prädiktor | n = 50 | n = 150 |
|---|---|---|
| gut | 0,962 | 0,955 |
| schlecht (schwere Ränder), nominal | 0,884 | 0,943 |
| schlecht, nur definierte Läufe (24 % NaN) | ≈ 0,85 | ≈ 0,93 |
| kein Prädiktor (λ = 0) | 0,594 | 0,824 |

Schlechte Prädiktoren machen das Intervall also *falsch*, nicht nur breiter.

(ii) Unter PPS ist der Schätzer stark verzerrt (−0,94). 06 verweist dafür aber auf eine
„IPW-Variante“. Die gibt es in `ppi.py` nicht, und als CLT-Variante wäre sie ebenfalls nur
approximativ.

Zusätzlich gilt: 06 §6.3 (q-Rekalibrierung „über das Audit“) ist bei fixem n in-sample.

**F8 – Der Hybrid ist exakt identifiziert.** [N, algebraisch]

Für J = 2 legen die drei beobachteten Zellen e^{u+u_A}, e^{u+u_B} und e^{u+u_A+u_B+u_AB} fest. Daraus
folgt N̂_hyb = n_obs + OR_syn·n₁₀·n₀₁/n₁₁. Der Schätzer ist genau dann konsistent, wenn
OR_real(x) = OR_syn(x). Das ist ebenso untestbar wie T0.

* Ist OR_syn ≈ 1, gilt Hybrid ≡ Huggins.
* Die Richtung des Huggins-Bias ist nicht einseitig: Positive Restabhängigkeit führt zu
  Unterschätzung, negative (komplementäre Detektoren) zu Überschätzung (Szenario E: +0,83).
* Für J = 3 ist das Modell mit allen Zweifach-Interaktionen gesättigt (7 Parameter, 7 Zellen). Die
  Annahme „keine Dreifach-Interaktion“ ist nicht testbar.
* Der Term 1/n₁₁ ist instabil und braucht eine Chapman-Korrektur.
* Literaturhinweis (M5, [B]): Das ist Bells Dual-System-Schätzer mit fixiertem Odds Ratio
  (JASA 88:1106, 1993).

**F9 – Triangulation.** [N]

Alle drei Schätzer sehen die echten Daten nur über die beobachteten Zellen. Übereinstimmung ist
daher kein Validitätsbeleg (A: 0,89 bei −30 %). Mehrheitslogik ist irreführend, weil HG und Hybrid
keine unabhängigen Urteile sind. In E liegen zwei von drei „Stimmen“ falsch. Zulässig ist allein
eine einseitige Falsifikationsprüfung von {T0 ∧ CI}.

**F13 – Clusterabhängigkeit.** [N]

§6.5 („Abhängigkeit innerhalb eines Bildes berührt die Gültigkeit nicht“) gilt nur für das Audit.
Der χ²-Transporttest überschreitet sein Niveau unter Bild-Zufallseffekten: 0,057 bzw. 0,089 bei
sd 2 und ≈ 4 Defekten pro Bild (nominal 0,05). Hinzu kommen die q-gewichteten Bruchzählungen und die
Mehrfachtests über Bins.

**F14 – HT-Varianz.** [N]

* Der Varianzformel fehlt der Fehlalarmterm Σ q_r(1−q_r)/π_r².
* Für Erwartungstreue muss q bedingt auf (x, h) kalibriert sein.
* E[1/π̂] > 1/π (Jensen). Ist π̂ = 0 in einem Bin, divergiert der Schätzer.
* π̂(x) ist mit Floor zu glätten, und das Bootstrap sollte über Bilder clustern.

**F15 – Der Prädiktor m̂_i = (N̂ − Σq)·w_i/Σw ist gültig, wenn eingefroren, aber nahezu
informationslos.** [N]

* Bei ≤ 1 Injektion pro Bild ist s̄_i Rauschen.
* Die richtige Zielgröße ist E[hidden_i] = λ_i(1 − π_i).
* Eine nahezu konstante Verschiebung c erhöht unter Nicht-Uniformität die Varianz um
  c²·Σ(1/π_i − 1).

### 2.5 Befunde zu 00

**F7 – „Fehler in S3 kosten nur Effizienz, nie Gültigkeit“ stimmt nur bedingt.** [B, mit Überlappung
zu F1, F2 und F5]

Die Aussage ist falsch für folgende Kanäle:

* (a) r in Satz 2;
* (b) aus S3 abgeleitete Schranken (D_max);
* (c) Ŷ oder Akzeptanzregeln, die während des Audits geändert werden;
* (d) Messung unter verify-first mit S3-Overlay. Der Fehler aus (e) wurde mit einer von S3
  unabhängigen Sensitivität simuliert. Die Kopplung über Automation Bias ist argumentiert, nicht
  simuliert;
* (e) Prop. 3 liefert π_i = σ_i√(κ/(ν−m_i)) = 0 bei σ_i = 0. Der Floor ist deshalb eine
  *Gültigkeitsbedingung*. 04 §4.5 fordert ihn zwar, §4.4 stellt ihn aber als bloßen Schutz dar;
* (f) eine Erweiterung von R während des Audits.

Korrigierte Fassung in §7, Nr. 7.

---

## 3. Versteckte Annahmen und Fehlermodi

**3.1 Mensch = Gold, und verify-first im Messpfad.** [B; A1 Machbarkeit, A5 Alternativen]

Verify-first misst s·ζ statt ζ. Falschzertifikate entstehen mit Wahrscheinlichkeit → 1 für n → ∞,
sobald s·ζ < ζ* < ζ (in (e): 0,021 < 0,025 < 0,03). Die Gegenprüfung (`redteam2_verify_feasibility_checks.py`
P2) zeigt:

* **Seed-Defekt-Korrektur** mit UCB(obs)/LCB_CP(s) bei n = 1000:
  * s_real = s_seed: 0,001–0,004 (gültig);
  * Seeds leichter als echte Misses (0,6/0,85): 0,073 (100 Seeds) und 0,107 (300 Seeds);
  * 0,5/0,9: 0,43–0,50.
  * Mehr Seeds verschlechtern das Ergebnis, weil die verzerrte Schätzung schärfer wird. Gepflanzte
    Fehler sind nachweislich leichter als echte (Kansal 2026; Mills-Seeding).
* **Zweiphasiges Gold-Nachprüfen** ist gültig (≤ 0,001), aber nicht billig. Bei wahrem ζ = 0,005,
  s = 0,7 und Ziel 0,025:
  * n = 1000 verify-first + 300 Expertenprüfungen: Power 0,874;
  * 250 Expertenprüfungen allein (exaktes Gold-Audit mit festem n): Power 0,881.
  * Bei seltenen Fehlern ist das direkte Gold-Audit also billiger. Allgemein braucht die
    Expertenstichprobe ≥ ln(1/δ′)/(ζ* − UCB_found) Einheiten, d. h. mindestens die
    Nullfehlergröße (113 bei ζ* = 0,025).
* **Kleiner Doppelannotationsanteil** (00 §7, 04 §6), exakt bei 0 Diskrepanzen, N = 2000,
  δ/2 = 0,025: Die UCB der Reviewer-Misses ist 34 / 16 / 10 Bilder bei n₂ = 200 / 400 / 600. Bei
  n₂ = 200 verbraucht dieser Term allein 0,017 eines Ziels von 0,025.
* **Einschränkung:** s = 0,7 ist eine *Annahme* aus `redteam_audit_checks.py` Z. 200 und keine
  Messung. Auch eine „Gold-Modus“-Blindsuche ist nicht Gold. Ein Zertifikat ist deshalb nur relativ
  zu einem schriftlichen Protokoll P sinnvoll. Das entspricht der Logik von Risk-Limiting Audits und
  den Composite Reference Standards nach Reitsma et al. 2009 (J Clin Epidemiol 62:797). Doppeltes
  Sampling mit fehlbarem Klassifikator: Tenenbein 1970 (JASA 65:1350).

**3.2 Das Review-Stratum liegt außerhalb des Stichprobenrahmens.** [B; A2 Machbarkeit, A1 Alternativen]

Lemma 0 setzt M_i = 0 für i ∈ R, und das Audit zieht nur aus U′ = U∖R. Restmisses in R, die unter
verify-first unentdeckt bleiben, fehlen im Zähler und in D_R. Der Netto-Effekt ist antikonservativ:
(a + M_R)/(b + M_R) > a/b.

* **Beispiel** (`redteam2_verify_alternatives_checks.py`): N = 2000, 600 defekt, 180 Fehlbilder,
  Top-700-Review, s = 0,7.
  * In R bleiben 45 ungefixt, in U′ liegen 30.
  * Ein Audit mit n = 400 findet ≈ 9. Die UCB ist 47, also gilt 47 − 9 = 38 ≤ 40, und es wird
    zertifiziert.
  * Wahres ζ_final = 0,033. Die Falschzertifikatsrate ist 0,550 und damit gleich der
    Zertifizierungsrate.
  * Bei s = 0,9 liegt sie bei 0,123, das sind 22 % der Zertifikate.
* **Folge:** „Schlechte Modelle machen das Zertifikat teurer, aber nie falsch“ (00 §1) ist
  widerlegt. Ein schlechteres Modell schiebt mehr Fehlbilder nach R.
* **Abhilfe (geprüft):**
  * Audit nach dem Review aus dem *ganzen* Pool, mit R als eigenem Stratum. Gültig (0,043 bzw. 0,049
    im schlechtesten Fall), kostet +20 % (sauberes R) bis +70 % (s = 0,7);
  * oder R unter Protokoll P inspizieren;
  * oder zufällige P-Nachprüfung von R mit δ-Aufteilung. Bei |R| = 150 begrenzen ≈ 43 Nachprüfungen
    ohne Befund den Rest in R auf ≤ 10 Einheiten bei δ′ = 0,025.

**3.3 Mehrere Runden ohne δ-Spending.** [B; A5]

Die Schleife Audit → Futility → Retraining → neues Audit → zertifizieren beim ersten Erfolg begrenzt
den Fehler nur durch Σ_r δ_r. Exakt: Bei wahrem ζ = 0,0205 in jeder Version und frischem Audit mit
fixem n = 600 ergeben sich 0,044 / 0,086 / 0,127 / 0,202 Falschzertifikate für
R_max = 1 / 2 / 3 / 5. Mit δ/R_max bleibt der Wert ≤ 0,044.

Unterscheidung: Labeländernde Retrainings brauchen eine neue Version mit neuem δ_r. Eine reine
Review-Erweiterung (Korrekturen über die Identität angerechnet) braucht kein neues δ. Werkzeuge:
α-Spending nach Lan–DeMets, α-Investing.

**3.4 Leere oder unerreichbare Zertifikate bei seltenen Defekten.** [B; A4, Zahlen [R]]

Bei einem Miss pro Einheit müssen für Recall ≥ 0,9 von den sauberen Einheiten inspiziert werden:

| Prävalenz p | 0,5 % | 1 % | 3 % | 10 % | 30 % |
|---|---|---|---|---|---|
| ein Miss pro Einheit | 78 % | 53 % | 26 % | 9 % | 3 % |
| gepackt, D_max = 6 | 95 % | 95 % | 78 % | 39 % | – |
| gepackt, D_max = 2 | 78 % | 78 % | 45 % | 16 % | – |

* **ζ* = 0,02** braucht n = 140 (N = 2000). Daraus folgt ein Bild-Recall ≥ 0,93 bei p = 30 %,
  ≥ 0,80 bei 10 %, ≥ 0,33 bei 3 % und 0 bei ≤ 1 %. Ein leerer Labeler hat ζ = p und besteht das
  Ziel, wann immer p ≤ ζ*.
* **Benchmarks:** MT (28 % defekt), BSData (≈ 36 %) und KSDD2 (≈ 11 %) liegen im machbaren Bereich.
  Das Problem trifft den eigentlichen Zielfall mit seltenen Defekten.

**3.5 Weitere Fehlermodi** [alle N, Evidenz aus der Machbarkeitslinse]

* **Unbekannte Defekttypen (A11).** Bei 90 uniformen Ziehungen wird ein Typ mit 1 % Prävalenz mit
  Wahrscheinlichkeit 0,60 getroffen, einer mit 0,3 % mit 0,24. Ein Typ mit 1,5 % besteht ζ ≤ 2 %
  und wird nie repariert. Abhilfe: eine Rule-of-Three-Schranke ≈ 3/n_uniform berichten und ein
  eigenes Entdeckungsbudget mit bekannten Inklusionswahrscheinlichkeiten vorsehen.
* **L–U-Verschiebung im Benchmark selbst (A12).** `data.make_split` zieht L stratifiziert
  round-robin über alle Defektarten. Echte Nutzer wählen „schöne“ Beispiele. Diese Wahl speist die
  Injektionsbank (Annahme T), die Fingerprint-Regeln und den Stacker. Alle Hypothesen sind daher auch
  unter verzerrtem L zu prüfen: Top-k nach Fläche oder Kontrast, 1–2 fehlende Arten, eine einzige
  Charge.
* **Subjektive Grenzen und subjektive Existenz (A13).** Verify-first-Flags liefern keine
  TP/FP/FN-Pixel. Dice-Zertifikate brauchen deshalb volle Grenzkorrektur. Abhilfe: eine
  Defektspezifikation pro Klasse, die Antwort „unsicher“ (zwei zertifizierte Versionen) und ein
  Orakelrauschen in der Simulation.
* **Mehrklassen (A15).** Für ζ_c ist die Bonferroni-Strafe moderat, n = 140/171/219/241/270 für
  J = 1/2/6/10/20 [R]. Klassen-Recall ist dagegen durch D_c begrenzt (Heuhaufen). Eine
  Zählregel für Klassenverwechslung fehlt. Empfehlung: klassenagnostisches ζ bei vollem δ, danach
  Klassen in fester Reihenfolge.
* **Retrainende Politiken (A14)** sind aus gespeicherten Läufen nicht simulierbar. 500
  Wiederholungen × 75 min ≈ 26 CPU-Tage pro Konfiguration.
* **Doppeltes Cross-Fitting (A8).** Es ist nicht strikt leckfrei: Der Lehrer j hat U_k als
  Negativ-Hintergrund gesehen. Strikt wird es erst mit K(K−1)/2 paarweise ausschließenden Lehrern
  (+ ≈ 83 min CPU). Außerdem fällt das Ensemble auf U weg.
* **Elongierte Defekte.** Mit any-hit-Toleranz gilt ein zu 10 % getroffener Riss als erkannt. Der
  Recall ist für Risse also nachsichtig (A9).

---

## 4. Neuheitsprüfung C1–C7

Einschränkung: arXiv, OpenReview, CVF, ACL Anthology, Semantic Scholar und die meisten Verlagsseiten
waren nicht abrufbar. Die Evidenz stammt aus Suchmaschinen-Abstracts und -Snippets sowie aus Code auf
GitHub (PAC-Labels, NSA, Kansal), der gelesen wurde. Nicht bestätigte Angaben tragen „(unverified)“.
Alle sieben Urteile sind vom Gegenprüfer bestätigt [B].

| # | Urteil | Nächstliegende Vorarbeit | Erforderliche Umformulierung |
|---|---|---|---|
| C1 Vollständigkeitsschätzung | teilweise bekannt | Mills, Fehler-Seeding (IBM FSD 1972); Kepler DR25 Injection (Thompson et al., ApJS 235:38, 2018); Balrog (Suchyta et al., MNRAS 2016); DQM (Chung, Krishnan, Kraska, PVLDB 10, 2017); Chao-Stopp in TAR (Bron et al., ACM TOIS, 2024/2025); Kansal 2026, Capture–Recapture über Labelfehlerdetektoren (doi:10.5281/zenodo.23205993, MAE schlechter als Rohzählung); Nicht-Identifizierbarkeit: Link 2003 (Biometrics 59:1123); fehlende Labels in Segmentierung: Rottmann & Reese (WACV 2023), Penquitt et al. (arXiv:2609.21822) | „Übertragung etablierter Vollständigkeitsmethoden (Seeding, Injection-Recovery, Capture–Recapture mit Kovariaten) auf Pixel-Auto-Labels industrieller Defekte, gekoppelt an ein designbasiertes Audit.“ Statt „testbare Transportannahme“: „Anpassungstest auf den beobachtbaren Zellen; blind für Defekte, die kein Detektor sieht“ |
| C2 Measure-and-Repair | teilweise bekannt (Kern bekannt) | PAC-Labels (Candès, Ilyas, Zrnic, ICML 2026, arXiv:2506.10908; Code `pac_utils.py` geprüft: korrigiert gezogene Punkte, Review-Stratum, Garantie auf dem korrigierten Datensatz, enthält auch squared_loss); Dodge–Romig Rectifying Inspection (AOQ/ATI, 1944/1959); Ijiri & Kaplan (J. Accounting Research 9(1):73–87, 1971); TAR-Zertifizierung (Lewis, Yang, Frieder, CIKM 2021, doi:10.1145/3459637.3482415) | „Geschlossene KKT-Form der Inklusionswahrscheinlichkeiten für das bekannte Measure-and-Repair-Prinzip.“ Lemma 0 als Beitrag streichen. Die Vorarbeitsspalte korrigieren (PAC-Labels ist nicht „nur Klassifikation“). Ein abgestimmtes „Top-k + Neyman-Audit“ erreicht die Prop.-3-Zielfunktion bis auf ≤ 1,2 % |
| C3 Segmentierungs-PAC-Labels | teilweise bekannt | PAC-Labels/LTT; Fieller-Linearisierung bzw. SHANGRLA-Assorter (Stark, arXiv:1911.10035); IPW-Testmartingale mit adaptiven Kontrollvariaten (Shekhar, Xu, Lipton, Liang, Ramdas, UAI 2023, arXiv:2305.06884); Xu, Karampatziakis, Mineiro (NeurIPS 2024); Csillag et al. (ICML 2025); Unterdeckung von Normal-Recall-Intervallen: Webber (arXiv:1202.2880) | „Anwendung von PAC-Labeling bzw. TAR-Recall-Zertifizierung auf geclusterte Segmentierungseinheiten; Verhältnis-Metriken über bekannte Linearisierung; prädiktables Tuning wie bei WSR/ALPHA/Shekhar.“ Die Wahl der Pixelschwelle ist LTT auf IPW-e-Prozessen. „Löst das Unterdeckungsproblem“ als Beitrag streichen |
| C4 Doppeltes Cross-Fitting | teilweise bekannt (Ingenieursbeitrag) | CV+/Jackknife+ (Barber et al., Ann. Stat. 2021); Cross-Prediction-PPI (Zrnic & Candès, PNAS 2024, doi:10.1073/pnas.2322083121); INCV (Chen et al., ICML 2019, PMLR 97); Confident Learning (Northcutt et al., JAIR 70, 2021); DML-Cross-Fitting (unverified) | „Konsequente Out-of-Fold-Disziplin über L ∪ U einschließlich Synthese-Hintergründen und Pseudo-Labels.“ Der Satz „Schwellen übertragen sich ehrlich“ setzt L~U-Austauschbarkeit voraus und ist so zu qualifizieren. Der Phase-1-Test „U-Schwelle vs. U-Oracle“ wird Akzeptanzkriterium |
| C5 Kacheln als Einheit | teilweise bekannt | Stark 2010, „Why small audit batches are more efficient“; Kartenvalidierung (Stehman & Czaplewski 1998); Proportionator (Gardi, Nyengaard, Gundersen 2008, Venue Comput. Biol. Med. vs. J. Microsc. unverified); MetaAT (Su et al., ECCV 2024) | „Übertrag des bekannten Prinzips kleiner Einheiten; moderat schärfer (Faktor 1,1–2,5), nur bei harter D_max pro Kachel; hilft nicht gegen die Heuhaufen-Grenze bei verteilten Misses.“ „Um Größenordnungen“ streichen |
| C6 Synthese-Labelvertrag | teilweise bekannt (Kern bekannt) | NSA (Schlüter, Tan, Hou, Kainz, ECCV 2022; Code `self_sup_tasks.py` geprüft: Label = geschwellte Intensitätsdifferenz, auch im Modus „binary“); FPI/PII (Tan et al., arXiv:2011.04197; MICCAI 2021); Null-Negative: Dwibedi et al. (ICCV 2017), Dvornik et al. (arXiv:1809.02492); Stärkekontrolle: RealNet (arXiv:2403.05897) | „Verallgemeinerung der aus NSA/FPI/PII bekannten signalbasierten Labels auf alle Generatoren, ergänzt um Ignore-Band und Verwerfungsregel; Null-Negative nach Dwibedi/Dvornik; Detektierbarkeits-Matching am realen Normalitäts-Score.“ Den Satz „nicht gesehen“ in 02-N2 korrigieren. Die Ablation braucht den NSA-Arm |
| C7 Nordstern-Metrik | teilweise bekannt | PAC-Labels „budget save“; TAR-Gesamtkosten bei zertifiziertem Recall (Lewis et al. 2021; Callaghan & Müller-Hansen, Syst. Rev. 9:273, 2020); WSS@95 (Cohen et al., JAMIA 2006); Dodge–Romig ATI; ASN sequentieller Tests; NoC@90; Angelopoulos et al. (arXiv:2506.07949) | „Erwartete Menschminuten bis zu einem *gültigen* Zertifikat, eine kostengewichtete ATI/ASN.“ Definition als Paar mit der Falschzertifikatsrate unter fehlbarem Reviewer, als Verteilung berichtet (§5.3) |

**Übersehene Felder** [alle B]:

* **M1 Kartenvalidierung in der Fernerkundung.** Olofsson et al., RSE 148:42–57 (2014); Stehman,
  RSE 2009. Das ist die nächste Vorarbeit für „designbasierte Zertifizierung pixelgenauer Labels“:
  Stichproben-, Response- und Auswertungsdesign; User's/Producer's Accuracy als
  Verhältnisschätzer. Unterschiede zu CertAL: Diese Arbeiten zählen Flächen statt Objekte, nutzen
  CLT-Intervalle statt endlicher bzw. anytime-Schranken und korrigieren nicht.
* **M2 Technology-Assisted Review (E-Discovery, Systematic Reviews).** Ein isomorphes Problem:
  Dokument ↔ Bild, relevant ↔ defekt, Review-Cutoff ↔ û, QBCB/Target-Methode ↔ Satz 2/N5. Daraus
  folgt, dass N5 bekannt ist.
* **M3 Akzeptanzprüfung.** Klie et al., ACL 2024 (arXiv:2405.11919); ISO 2859-1:2026 und 2859-2. Das
  fixe-n-hypergeometrische ζ-Zertifikat ist **identisch** mit einem LTPD-Plan bei Konsumentenrisiko
  β = δ (0 von 620 (n, k)-Zellen weichen ab). ATI-optimale Rectifying-Pläne: ζ̄ = 0,0025 → n = 290,
  c = 2; ζ̄ = 0,005 → n = 420, c = 4. Die Formulierung sollte LTPD/LQ verwenden, nicht AQL (AQL ist
  ein Produzentenrisiko).
* **M4 Risk-Limiting Financial Audits.** Shekhar et al. 2023 ist die Basis von N4: gewichtete
  Ziehung ohne Zurücklegen und adaptive Kontrollvariaten. Das löst auch den offenen Punkt aus 04 §6.
  Für die Neyman-Auditpolitik: Ao et al., arXiv:2601.21471.
* **M5 Census Dual-System Estimation.** Bell 1993 (fixiertes Odds Ratio = Hybrid); Wachter & Freedman,
  „The Fifth Cell“, Eval. Rev. 24:191 (2000); Das, Kennedy, Jewell, JASA 119:1309 (2024, `drpop`).
  `drpop` setzt weiterhin CI gegeben x voraus und beseitigt die Korrelationsverzerrung nicht.
* **M6 Probability of Detection (MIL-HDBK-1823A, MAPOD).** Liefert die parametrische Form von π(x)
  als Hit/Miss-GLM in log(Größe). Ein a90/95 aus Injektionen ist *modellbasiert* (Annahme T) und kein
  Zertifikat.
* **M7 Proportionator.** PPS über Kacheln nach automatischer Bildanalyse; Boyce & Gundersen,
  Front. Neuroanat. 12:19 (2018).
* **M8 Label-effiziente Evaluation.** Sawade et al. (NIPS 2010, unverified in dieser Session);
  Marchant & Rubinstein, KDD 2021 (arXiv:2006.06963); Poms et al. (ICCV 2021); MetaAT. Alle nur
  asymptotisch gültig.
* **M9 Labelfehlerdetektion für fehlende Labels.** Rottmann & Reese; Penquitt et al.; REC✓D
  (arXiv:2508.06556, verpasst bis zu 66 %). Die Spalte „MetaSeg (nur FP)“ unterschlägt diese Linie.
* **M10 Gold-/Honeypot-Fragen.** Oleson et al., HCOMP 2011. Synthetische Seeds erben das
  Transportproblem, siehe §3.1.
* **M11 Begriffskollision „Zertifikat“.** ISO/IEC 5259-4:2024 (Prozesszertifizierung).

**Was an Neuheit übrig bleibt:**
* die Übertragung auf pixelgenaue industrielle Auto-Labels mit korrigiertem Pool;
* die exakte KKT-Form von Prop. 3, korrekt, aber mit geringem Zusatznutzen;
* das Ignore-Band und die Verwerfungsregel;
* das Detektierbarkeits-Matching am realen Normalitäts-Score;
* möglicherweise der GT-freie Vergleich vorhergesagter beobachtbarer Zellanteile. Er ist ein
  Anpassungstest, kein Vollständigkeitstest.

---

## 5. Machbarkeit und Prioritäten (Mac mini M4, eine Person)

### 5.1 Rechen- und Entwicklungsaufwand [U; A6]

Gemessene Basis (Remote-CPU, MT):

| Posten | Aufwand |
|---|---|
| Training: 0,46 s/it × 600 it × 8 Member | 37 min |
| Inferenz auf U | ≈ 11 min |
| Inferenz auf T | ≈ 3 min |
| Rest | ≈ 24 min |
| **Summe** | **75 min** |

Schätzungen (FLOP-Verhältnisse, ungemessen):

| Komponente | Aufwand |
|---|---|
| Doppeltes Cross-Fitting mit 2 heterogenen Membern pro Fold | ≈ 110 min |
| Paarweise ausschließende Lehrer | + 83 min |
| Injektionsdurchgang | 11–27 min |
| **voller CertAL-Lauf** | **2,0–3,6 × heutiger Lauf**, nicht 2–4,5 × (Korrektur des Gegenprüfers) |
| S5 | ≈ 15 h pro Datensatz und Seed auf der CPU, über Nacht machbar |
| LODO-Validierung (≈ 630 Läufe) | lokal nicht praktikabel |

Gegenargument des Prüfers:
* „27 Läufe ≈ 34 h“ ist überzogen. Mit echten Datensatzgrößen sind es ≈ 24 h, plus ≈ 7 h für KSDD2
  und MVTec AD 2. „≈ 1 Tag“ verlangt daher einen M4, der ≈ 1,3–1,5 × so schnell ist wie die
  Remote-CPU. Plausibel, aber ungemessen.
* „Rechenschätzungen optimistisch“ ist nicht belegt.
* Entwicklungszeit für eine Person: 11–18 Wochen insgesamt. Das ist eine **unüberprüfbare
  Schätzung**.

### 5.2 Wo Aufwand Wert erzeugt

Gültigkeit entsteht nur aus S4, dem Einfrieren und dem Messprotokoll. S1–S3 kosten nur Effizienz,
unter den Bedingungen von F7. Effizienz für ζ entsteht fast ausschließlich im **Review-Stratum**,
nicht in der nicht-uniformen Auditallokation:

* **Uniform minimax-optimal bei null Funden** [B; A2 Alternativen]: Ein Design, das bei null Funden
  zertifiziert, braucht Π(1 − π_i) ≤ δ über die K* + 1 Einheiten mit kleinstem π. Daraus folgt
  E[n] ≥ N′(1 − δ^{1/(K*+1)}), mit Gleichheit nur bei uniformem π. Für N′ = 1850 und K* = 40 sind
  das 129 [R] bzw. 130 in der Tabelle. Das widerlegt auch den Hebel (i) aus 04 N5: Eine
  Proposal-Verteilung kann die Nullfund-Stichprobe nicht verkleinern.
* **PPS gegen uniform bei ζ** [B mit Korrektur; A3 Machbarkeit]. Wahres ζ = 0,005, Ziel 0,02:

  | Design | verschiedene Einheiten bis Zertifikat (Median) |
  |---|---|
  | uniform | 314 |
  | PPS, γ = 0,3, AUC 0,7 | 680 |
  | PPS, γ = 0,3, AUC 0,9 | 554–560 |
  | PPS, γ = 0,1 | 880–1088 bzw. kaum Zertifizierung |

  Nahe am Ziel (ζ = 0,010–0,015) liegt PPS mit AUC 0,9 gleichauf. PPI mit λ = 1 ändert ≤ 5 pp.
  Ursache ist die Wettobergrenze 1/(z^max − m) mit z^max ≈ 1/γ. Diese endliche Bereichsgrenze
  übersieht die CLT-Zielfunktion von Prop. 3. Vgl. PPEB (Aslam, Popa, Rivest, EVT 2008): Bei binären
  Fehlern pro Einheit reduziert sich PPEB auf uniform.
* **Volle S4 gegen CertAL-lite** (`lite_zeta.py`, [B mit Korrektur]). Kosten bei ζ* = 0,02:

  | Variante | Inspektionen |
  |---|---|
  | lite, Orakelplanung (90 % Power) | 700 |
  | lite, ohne Orakel (gruppensequentiell), Median | 650 |
  | volle S4, Median / q90 | 810 / 1014 |

  Fair verglichen kostet die volle S4 am Median 12–25 % mehr.
* **Kosten der jederzeitigen Gültigkeit** [U; A4 Alternativen]. K = 10/2000:

  | Verfahren | Inspektionen |
  |---|---|
  | fixes n, 50 % / 90 % Power | n₅₀ = 220, n₉₀ = 355 |
  | PPR mit Prior | q50 300, q90 550 |
  | PPR mit uniformem Prior | q50 665 |
  | WOR-Betting-Mischung | q50 311 |

  „PPR so billig wie fixes n“ ist **widerlegt**. Die Mehrkosten stammen überwiegend aus der
  anytime-Eigenschaft selbst. Andererseits ist der Hauptnutzen jederzeitiger Verfahren die
  Robustheit gegen eine falsche K′-Prognose: Fixes n = 310, geplant für K′ = 10, hat Power 0,93 /
  0,56 / 0,23 bei K′ = 10 / 20 / 30.
* **Grenzwert eines Fehlbilds.** Jedes vor dem Audit entfernte Rest-Fehlbild spart ≈ 18 Audits bei
  ζ* = 0,02 und ≈ 45 bei ζ* = 0,01 (nichtlinear: die ersten 3 kosten 215). [B mit Korrektur von 40
  auf 45]

**Prioritäten** (Nutzen pro Aufwand, Machbarkeitslinse, vom Gegenprüfer mit Einschränkungen
bestätigt):

1. echte Baseline-Zahlen (Phase 1), mit Gate: Recall in Runde 0 vs. Runde 1 auf U nach
   Risikostratum, U vs. U-Oracle-Schwelle;
2. ein gültiges einfaches ζ-Zertifikat mit korrekter Buchführung des korrigierten Pools;
3. HITL-Simulation mit Aktionszählungen;
4. ein DINO-kNN-Reviewsignal;
5. verify-first-Flags für die Reparatur;
6. die Korrektur des Synthese-Labelvertrags (unsichtbare Positive sind ein Korrektheitsfehler).

Senkt Self-Training den U-Recall in den Niedrigrisikostrata, wird doppeltes Cross-Fitting
vorgezogen. Laut 04 §4.8(2) unterdrückt In-Sample-Self-Training genau die übersehenen Defekte.

### 5.3 Nordstern-Metrik: Definition [B; A10, C7]

Für eine Aufgabeninstanz werden vorab registriert:
* ein **zweiseitiges** Ziel (ζ*, dazu φ* pro sauberem Bild oder ein Ziel für Pixel- bzw.
  Toleranzpräzision);
* δ;
* eine Politik π;
* ein Budget B.

Begründung für zweiseitig: Mit `metrics.image_stats` (tol = 2, merge_radius = 2) erreicht eine
Vollvordergrund-Maske ζ = 0, Recall = 1 und sogar Komponentenpräzision 1/1 auf defekten Bildern.
Nur φ bzw. die Pixelpräzision entlarvt sie.

Berichtet werden:

1. die Erfolgskurve P(Zertifikat nach ≤ B Minuten);
2. Median und 90-%-Quantil der Minuten, gegeben Zertifizierung;
3. die Falschzertifikatsrate (≤ δ), **auch unter simuliertem fehlbarem Reviewer** mit
   s ∈ {1; 0,9; 0,7} als Funktion von n, größen- und kontrastabhängig und positiv mit der
   Modellschwierigkeit korreliert;
4. die wahre GT-Qualität des ausgelieferten Pools;
5. die Fixkosten für L und die Rechenzeit, getrennt ausgewiesen;
6. die Futility-Rate.

**Kostenmodell und Sensitivität.** Die Simulation speichert kostenfreie Aktionsvektoren n_a. Gezählt
werden Verify-Ansichten, Vollinspektionen pro Einheitentyp, korrigierte Komponenten,
Grenzkorrekturen und Ziehungen. Daraus folgt

Minuten = Σ_a c_a · n_a,

ausgewertet über einen Latin-Hypercube auf folgenden Spannen:

| Aktion | Spanne |
|---|---|
| Verify | 2,7–6 s |
| Flag + Klick | + 3–5 s |
| Korrektur | 10–120 s pro Komponente |
| Polygon / Riss | 60–300 s |
| **Vollinspektion** | 10–120 s pro Bild (Annahme, kein Anker in 05 §3.1; 2,72 s ist ImageNet-Ein-Bit-Verifikation) |
| Kachel-Overhead c₀ | 1–6 s |

Berichtet wird, wo im Kostenraum H* gilt, und zwar über die komponentenweise Dominanz der Zählungen
oder über Break-even-Verhältnisse:
* **c_inspect/c_verify.** Baseline (b) skaliert linear damit. Ein Gold-Audit kostet pro Bild
  vermutlich 5–10 × verify-first.
* **c₀.** Kacheln lohnen nur bei c₀ ≲ 1–2 s [N].
* **c_review/c_audit.** Bestimmt die Ertragsschwelle für das Review: Reviewen lohnt, solange der
  Ertrag > (c_review/c_audit)/(dn₉₀/dK′) ist. Das sind ≈ 5,5 % bei ζ* = 0,02 und ≈ 2,5 % bei 0,01
  [N]. (Korrektur nach Abschluss des Berichts: Die erste Fassung hatte das Kostenverhältnis
  invertiert. Ein geprüftes Bild kostet c_review und spart y·(dn₉₀/dK′)·c_audit; so steht es auch in
  00 §4.5.)

γ und κ werden vorab fixiert und nie auf den H*-Instanzen getunt. Alle Baselines bekommen dieselben
gültigen Schranken. **Vor Phase 4** steht eine einstündige Selbst-Zeitmessung: 50 MT- und 50
KSDD2-Einheiten, jeweils Verify, Vollinspektion und Korrektur.

Gegenargument zu „mit GT als Orakel kann Gültigkeit in der Simulation nicht scheitern“: Unsichtbar
wird dann nur der Menschfehler-Pfad. Designfehler wie Optional Stopping, D_max, Rückkopplung,
Mehrrunden-δ und gelerntes r können und müssen in der Simulation scheitern.

Für ζ existiert außerdem eine geschlossene Form [N; A12 Alternativen]:
Kosten(k) = k·c_review + n₉₀(K′(k))·c_audit,
mit der Restkurve K′(k) aus einem GT-Lauf. Für ζ ist dann keine sequentielle Simulation nötig.

---

## 6. Einfachere bzw. stärkere Alternativen und „CertAL-lite“

Beide Linsen (Machbarkeit A7, Alternativen) kommen unabhängig zu demselben Kern. Der Gegenprüfer hat
ihn bestätigt, mit dem Hinweis, dass CertAL-lite ein **klassisches Akzeptanzprüfungsdesign**
(Rectifying LTPD/AOQL) ist und keine Neuheitsbehauptung tragen kann.

**CertAL-lite (minimaler gültiger Satz)**

* **L0 Einfrieren.** Vor der ersten Auditziehung werden Labels, Risikoscores, Konventionen
  (tol, merge_radius), Ziele, die δ-Aufteilung, R_max und der Seed eingefroren und gehasht.
* **L1 Labeler.** Die bestehende Pipeline (K = 4, Runde 0), plus die Korrektur der unsichtbaren
  Positiven in der Synthese. Kein doppeltes Cross-Fitting, solange das Gate aus §5.2 es nicht
  verlangt.
* **L2 Ein Kontrolldetektor.** anomalib 2.7.0 AnomalyDINO (DINOv2-S, Apache-2.0, laut METADATA),
  nur als Risiko-Merkmal, nie als Labeler-Input.
* **L3 Risikoscore.** Das bestehende `review_risk`-Rangmittel oder ein logistischer Stacker auf
  L-OOF. Keine Vollständigkeitsschätzer.
* **L4 Reparatur.** Verify-first-Review in Risikoreihenfolge, in Batches zu 25. Stopp, sobald der
  Ertrag an Fehlbildern pro Batch unter die Schwelle aus §5.3 fällt. Gültig, weil der Rahmen nur von
  Ergebnissen geprüfter Einheiten abhängt [N]. Verify-first-Korrekturen werden **nicht** als
  gemessen angerechnet.
* **L5 Zertifizierung.**
  * Nach der Reparatur wird ein Audit ohne Zurücklegen aus dem **ganzen Pool** gezogen, R
    eingeschlossen (optional als eigenes Stratum mit exakter kombinierter Schranke). Ausnahme: R
    wurde unter P inspiziert.
  * Jede Audit-Einheit wird unter **Protokoll P** inspiziert: blinde Vollsuche, als UI-Hilfe
    kachelweise gescannt (statistisch ohne Rolle), dann Overlay einblenden, Vereinigung bilden.
  * n kommt aus der exakten Powertabelle mit konservativer K′-Prognose. Die Prognose beeinflusst nur
    die Power.
  * Vorab registriert ist ein Erweiterungsblick beim pessimistischen n mit δ-Aufteilung. Die
    Fixe-n-Schranke wird nie überwacht. Frühes Stoppen ist nur zum Nicht-Zertifizieren erlaubt.
  * Zertifikat: ζ_final ≤ (UCB_hyp − gefunden)/N, ausgedrückt als **LTPD/LQ-Plan mit
    Konsumentenrisiko δ**.
  * Zweitens φ (Fehlalarme auf sauberen Bildern) für die Zweiseitigkeit.
* **L6 Abgeleitete Schranke** (ohne zusätzliches δ, kein D_max) [N, Algebra hier nachvollzogen].
  Seien U die UCB der Restfehlbilder und A die Zahl der verifizierten defekten Bilder (aus Review
  und Audit). Dann gilt für den Recall defekter Bilder ρ_b ≥ 1 − U/(A + U), weil
  #defekt ≥ A + |B| und x ↦ x/(A + x) monoton ist. Beispiel: A = 150, U = 40 ergibt ρ_b ≥ 0,79.
  Das gilt nur, wenn die A Bilder unter P korrigiert wurden.
* **L7 Bericht.** Das Zertifikat mit Annahmeliste: Design, Protokoll P, eingefrorene Labels, keine
  Zählschranke. Dice und Recall nach Größenklassen sind als *approximativ* zu kennzeichnen. Die
  Begriffe „statistisches Pool-Zertifikat, keine Prozesszertifizierung nach ISO/IEC 5259“ sind
  festzuhalten.
* **L8 Runden.** Höchstens ein Retraining, mit Σδ_r ≤ δ. Das zweite Audit zieht aus U″ und behält
  die geprüften Einheiten als bekanntes Stratum.
* **L9 Regime-Weiche** vor jeder Menschminute, mit LCB(p) und UCB(p) aus der Triage (§3.4):
  * hohe Prävalenz: ζ + φ + ρ_b;
  * mittlere Prävalenz: stratifizierte Aussage („Missrate ≤ UCB im signalfreien Stratum“);
  * niedrige Prävalenz: Vollinspektion mit modellgestützter Beschleunigung.
  * Gilt UCB(p) ≤ ζ*, wird das ζ-Zertifikat als leer markiert.

Der Codeaufwand liegt bei ≈ 100 Zeilen in `ppi.py`, `cli.py` und `evaluate.py`, plus
`scipy.stats.hypergeom`. Für den anytime-Fall gibt es `confseq.betting_cs(x, N=N′)`; die
Unterstützung von Ziehen ohne Zurücklegen ist im Quellcode von confseq 0.0.11 geprüft. `ppi_py` ist
nur CLT-basiert und damit **nicht** für Zertifikate geeignet.

**Später hinzufügen, nur mit gemessenem Gewinn** (Go-Kriterien gemessen über L7 bzw. die
geschlossene Kostenform auf GT-Datensätzen; alle [N]):

| # | Komponente | Go-Kriterium |
|---|---|---|
| 1 | stärkerer Encoder (Labeler und Kontrolldetektor) | K′(k*) oder n₉₀ sinkt um ≥ 10 % |
| 2 | anytime-Stopp via PPR ohne Zurücklegen, Prior auf Prognose zentriert | Median-n ≤ 0,9 × fixes n₉₀ bei Falschzertifikatsrate ≤ δ am Rand |
| 3 | Synthese v2 über die Fehlerkorrektur hinaus (C6) | OOF-Recall pro Größenklasse steigt, K′ sinkt |
| 4 | unabhängige Zusatzsignale (Sequenzkonsistenz, zweite Ansicht, zweiter Detektor) | Anteil unsichtbarer Misses sinkt |
| 5 | Self-Training Runde 1, dann C4 | K′(k) sinkt um ≥ 10 % |
| 6 | Recall-Zertifikat auf Defektebene | nur wenn vertraglich nötig (siehe unten, umstritten) |
| 7 | PPS/PPI im Audit | n₉₀ sinkt um ≥ 15 %, getrennt auch für Recall geprüft (für Recall ungeprüft) |
| 8 | Injection-Härtekarte als Allokationskovariate | Top-20-%-Härte-Bilder enthalten ≥ 50 % der von allen Detektoren übersehenen echten Misses |
| 9 | Kacheln als Stichprobeneinheit | gemessene Kachel-Prüfzeit pro Fläche ≥ 30 % niedriger **und** gepoolter Recall pro Größenklasse gefordert |
| 10 | Satz 2 (zweiphasig) und C1-Populationsschätzer | Forschungsspur, nie im Zertifikat |

**Umstritten: bildgewichteter Recall als Ersatz für Zähl-Recall.** Die Alternativenlinse (A7) und
die Machbarkeitslinse (A9) schlagen 1 − Σ(M_i/D_i)/#defekte Bilder vor. Dieses Maß ist auf [0, 1]
beschränkt, braucht keine D_max und ist auf Bildebene so scharf wie gepoolter Recall auf Kacheln.
Die Lücke zur LCB beträgt 0,29–0,32 bei 10 % Fläche (`lite_recall.py`, [N]). Der Gegenprüfer der
Mathematiklinse (F1) zeigt aber: Es ist eine **andere Zielgröße**, die gegen genau den
Pitting-Fehlermodus blind ist. Im Gegenbeispiel beträgt sie 5/401 ≈ 0,0125, das Zertifikat ist
„wahr“, während der Zähl-Recall bei 0,79 liegt. Empfehlung: Den bildgewichteten Recall nur
verwenden, wenn er unter eigenem Namen berichtet wird, mit dem Hinweis, dass er schwere Einheiten
ignoriert. Er ist bildweise zu definieren, nicht kachelweise, weil die Kachelung willkürlich ist.

**Rolle von C1 im Produktpfad** [U; A3 Alternativen]. Unstrittig:
* C1 ist für die Gültigkeit redundant.
* Als Qualitätszahl ist C1 gefährlich (A; E).
* Eine GREG-Kontrollvariate mit (Σq_r, w_i) absorbiert jedes skalare N̂ exakt.

Umstritten ist der Effizienzwert. Der Erstprüfer nennt höchstens 8–16 % Varianzgewinn durch die
Härtekarte. Der Gegenprüfer findet mit genauer Karte 13–33 % (15 versteckte Misses) bzw. 29–58 %
(60–150) im *Schätzregime*, im Nullfund-Zertifizierungsregime dagegen keinen Gewinn (Minimax-Grenze).

Kompromiss:
* N̂_HT/HG/Hybrid, Triangulation und die Ausweichregel aus Berichten und Zertifikat entfernen.
* Zwei Rollen behalten, in denen Modellfehler nur Power kosten: (a) die K′-Prognose zur
  Auditplanung bzw. als PPR-Prior; (b) die Injection-Recovery-Karte als Review-Risiko-Merkmal hinter
  dem Go-Kriterium Nr. 8.
* S3 wird nach Misses pro Review-Minute bewertet, nicht nach dem Fehler von N̂ (A16, [N]).

---

## 7. Erforderliche Änderungen an 00 und 06 (priorisiert)

Änderungen an 04 und am Code stehen dabei, wo sie für 00 und 06 nötig sind. In diesem Durchgang wurde
nur diese Datei geändert.

1. **00 §1, S4, §7; 04 §4.1/§4.2: Messprotokoll.**
   * „Mensch = Gold“ ersetzen durch „Zertifikat relativ zum registrierten Protokoll P (blinde
     Vollsuche, dann Overlay, Vereinigung)“.
   * Verify-first nur für Reparatur, deren Korrekturen nicht als gemessen gelten.
   * Den Satz zum „kleinen Doppelannotationsanteil“ streichen bzw. als Bias-Monitor
     (Punktschätzung) umformulieren.
   * Grundlage: A1, A5 und Durchgang 1 (e).
2. **00 S4; 04 §4.1/§4.5: Review-Stratum in den Rahmen.**
   * Zielgrößen auf dem ausgelieferten Pool definieren: ζ_final = (1/N)Σ_{i∈U} b_i^final, ρ_final
     mit M_R^res.
   * R unter P inspizieren oder in den Auditrahmen aufnehmen (eigenes Stratum, δ-Aufteilung).
   * Grundlage: A2 (Machbarkeit), A1 (Alternativen).
3. **00 §1: „nie falsch“.** Ändern zu „nie falsch, sofern die registrierten Annahmen gelten (Protokoll
   P, eingefrorene Größen, Schranken, δ-Aufteilung über Runden)“. Den Satz „Schlechte Modelle machen
   das Zertifikat teurer, aber nie falsch“ entsprechend qualifizieren. (F1, A1, A2)
4. **04 Satz 2, §2.1, §4.5(i); 00 S3/S4: r einfrieren.**
   * Satz 2′ aus §2.3 übernehmen: r ist F₀-messbar, auch prädiktables r^(t) ist verboten.
   * Zulässig ist zweiphasiges Lernen oder eingefrorenes r mit der Korrekturidentität.
   * In 00 festhalten: „Der Stacker darf für f und q nachkalibriert werden, r für Satz 2 bleibt
     eingefroren.“
   * Grundlage: F2, Durchgang 1 (d).
5. **04 §4.2/§6; 00 S4: D_max.**
   * Die Prosa „Gültigkeit allein aus Randomisierung“ ersetzen durch Satz 1′.
   * Die Unmöglichkeitsaussage P(LCB > ε) ≤ δN/(N − n) aufnehmen.
   * „D_max ≈ 1–2“ ersetzen durch „typisch 1–2; als harte Schranke vorab zu begründen, sonst ist
     das Recall-Zertifikat modellbasiert“.
   * Zielgrößen ohne unprüfbare Zählschranke bevorzugen (F1). Das Flächenmaß als „gültig, aber
     praktisch leer“ markieren.
6. **`sims/sim_recall_cert.py`, `ppi.py` v2; 04 S1: prädiktable Wettobergrenze.**
   * Wettobergrenze aus F_{t−1} berechnen.
   * Zur Laufzeit Z_t ≤ z^max_t prüfen; eine Verletzung macht das Zertifikat ungültig, kein
     NaN-Überspringen.
   * Adversarialen Unit-Test hinzufügen und die S1-Spalte neu erzeugen.
   * `sim2.py` Z. 20 und die Gitterkonvention in `sim3.py` Z. 28 korrigieren.
   * Grundlage: F4, F16.
7. **00 S3; 06 §5; 04 §4.4: S3-Fehler.** „Fehler in S3 kosten nur Effizienz“ ersetzen durch: „…
   sofern
   (i) f, λ, q nur eingefroren oder prädiktabel aus dem Audit lernen (bei fixem n nur
   cross-fitted);
   (ii) q einen S3-unabhängigen Floor q ≥ γ/N′ hat, als Gültigkeitsbedingung;
   (iii) r, Ŷ, U′/R, Konventionen und Schranken vor dem Audit eingefroren sind und die Schranken
   nicht aus S3 stammen;
   (iv) die Messung nicht von S3 beeinflusst ist (Protokoll P)“.
   Den Floor im Code erzwingen. (F7)
8. **00 S4, §6 Phase 0; 04 §0.8: hypergeometrische Schranke.** Die Schranke nur bei vorab
   registriertem fixem n und uniformer Ziehung verwenden. Jederzeit stoppbar über PPR ohne
   Zurücklegen auf einem separat gelosten uniformen Strom. Unter nicht-uniformem q über Satz 1.
   (F6, C3)
9. **06 §4: Transporttest.**
   * Umbenennen in „Anpassungstest der beobachtbaren Implikation von T“.
   * Festhalten: N(x) ist nicht identifiziert, HT nutzt nur n_obs(x), T0 ist untestbar; der Test
     erkennt Anteilsverschiebungen, aber nicht proportionale bzw. Sichtbarkeitsabweichungen.
   * Die automatische Ausweichregel streichen. Ein Wechsel ist nur unter einer genannten
     Verknüpfungsannahme zulässig.
   * Ein Sensitivitätsband N̂(τ) = Σ_x n_obs(x)/(π_syn(x)(1 − τ)) berichten.
   * Einen bildcluster-robusten Test verwenden und Multiplizität über Bins kontrollieren.
   * Grundlage: F3, F9, F13, M5.
10. **00 §5 H\*: Baselines.**
    * Baseline (a) zweifach definieren: naiv nach 04 §1, und stark (Top-k-Review, danach SRSWOR
      nach dem Review, exakte Schranke, angerechnete Korrekturen, Planung ohne Orakel).
    * Neu (c): Dodge–Romig-LTPD- bzw. ISO-2859-2-Plan mit Rectifying.
    * CertAL-lite und eine TAR-Target-Methode als weitere Baselines.
    * Grundlage: A2 (Alternativen), A3/A7 (Machbarkeit), C2, M2, M3.
11. **00 S4: Standarddesign für ζ.** CertAL-lite wird der Standard. Prop. 3/PPS/PPI kommen nur mit
    Gate (n₉₀ −15 % bei gleicher Gültigkeit). 04 N5 Hebel (i) korrigieren. (A3, A2, A4)
12. **00 §1, C7; 06 §8: Nordstern.** Als Paar mit der Falschzertifikatsrate unter fehlbarem
    Reviewer definieren, mit zweiseitigem Ziel, als Verteilung, über Aktionszählungen und
    Kosten-Sweep. Als kostengewichtete ATI/ASN einordnen. (A10, C7)
13. **00 §5: C1–C7.** Gemäß der Tabelle in §4 umformulieren und die Vorarbeitsspalten korrigieren
    (PAC-Labels, MetaSeg, Kartenvalidierung, TAR, Bell).
14. **00 B10, 03 Z. 59–60.** „Keine etablierte Methode schätzt übersehene Defekte direkt“ ersetzen
    durch „Für Pixel-Auto-Labels fehlt eine etablierte Methode. In Software-Test, NDE (POD),
    Astronomie, Daten-Cleaning (DQM) und TAR existieren direkte Verfahren“. Damit entfällt der
    Widerspruch zu 06 §9. (C1, M9)
15. **00 S4, C5; 04 N6: Kacheln.** „Um Größenordnungen schärfer“ streichen. Das Bild bleibt
    Stichprobeneinheit, die Kachel ist UI-Scanhilfe. Die Lemma-0-Gegenbeispiele und die
    Heimatregel auf GT (argmax der eingefrorenen Karte) dokumentieren. (F11, F12, C5, A9)
16. **00 §6 Roadmap.**
    * Die Spalte „Rechenbedarf“ aufteilen in „Rechenzeit (gemessen auf M4)“ und
      „Entwicklungstage“, mit Exit-Kriterium pro Phase.
    * Phase 1 vor jeder CertAL-Komponente, mit dem Self-Training-Gate.
    * R_max und δ_r vorab registrieren, den Freeze-Ablauf festschreiben.
    * Phase 4 auf Einzelschuss-Politiken beschränken.
    * S5 lokal auf 2–3 Regeln × 3 Datensätze verkleinern.
    * Grundlage: A5, A6, A14.
17. **00 §7 Heuhaufen; 04 N5/§0.8.**
    * 640 → 624.
    * Den gepackten schlechtesten Fall mit dem Faktor D_max angeben.
    * Den Machbarkeitscheck mit LCB(p) und UCB(p) aus der Triage führen, einschließlich der
      Leerheitsregel für ζ* ≥ p.
    * Die Regime-Weiche aus L9 übernehmen.
    * Grundlage: F10, A4.
18. **06 §3.2/§3.3: Hybrid und Huggins.**
    * Den Hybrid als Bell-Schätzer mit fixiertem Odds Ratio (OR-Transport) bezeichnen, für J = 2
      exakt identifiziert und untestbar. Sensitivitätsanalyse über OR, Chapman-Korrektur.
    * Den Satz „vermutlich robust“ streichen.
    * §3.2: „positive Restabhängigkeit → Unterschätzung, negative → Überschätzung“; „J = 3:
      gesättigt“.
    * Einen `drpop`-artigen DR-Schätzer nur mit dem Hinweis „setzt weiterhin CI gegeben x voraus“.
    * Grundlage: F8, M5.
19. **06 §5, §6.3: Garantie.** Auf Satz 1 + Prop. 4 mit den Bedingungen aus Nr. 7 beschränken.
    `ppi.estimate_ratio` als „approximativ, nur SRS, feste n“ kennzeichnen; bei tb ≤ 0 „kein
    Intervall“ statt NaN ausgeben. Den Verweis auf die nicht existierende IPW-Variante entfernen.
    Rekalibrierung im Audit nur prädiktabel oder cross-fitted. (F5)
20. **06 §3.1: HT.**
    * Die Bedingungen nennen: q bedingt kalibriert, x fehlerfrei gemessen.
    * Den q-Varianzterm ergänzen und π̂(x) geglättet mit Floor verwenden.
    * POD/MAPOD zitieren, Hit/Miss-GLM in log(Größe) als Form von π(x).
    * a90/95 nur als „modellbasiert“ berichten.
    * Grundlage: F14, M6.
21. **06 §8: Validierungsplan.**
    * H1: die rohe q-gewichtete Fangzahl Σq_r als naive Baseline aufnehmen (MAE schlagen, wie bei
      Kansal 2026).
    * H2 in zwei Familien teilen: Anteilsverschiebungen sollen erkannt werden (≥ 80 %),
      proportionale Abweichungen haben Power ≈ α und sind über das Audit zu begrenzen.
    * H3 auf n₉₀ bzw. LCB-Breite bei gleicher Gültigkeit umstellen.
    * Hypothesen unter verzerrtem L prüfen.
    * Grundlage: F3, C1, A3, A12.
22. **04 §2: Literatur.** TAR, Kartenvalidierung, Akzeptanzprüfung, Shekhar et al. 2023,
    label-effiziente Evaluation, Dual-System-Schätzung und PPEB ergänzen. N4 umformulieren zu
    „Kombination aus Shekhar et al. 2023 und SHANGRLA/ALPHA-Linearisierung“. N5 als bekannt
    einstufen. (M1–M5, M8)
23. **00 S4/S5: Mehrklassen und unbekannte Typen.** Klassenagnostisches ζ als primäres Zertifikat;
    Klassen in fester Reihenfolge; eine Rule-of-Three-Schranke für unentdeckte Typen; ein
    Entdeckungsbudget mit bekannten Inklusionswahrscheinlichkeiten. (A11, A15)
24. **00 §1/§7: Begriff.** „Statistisches Pool-Zertifikat (designbasiert), keine
    Prozesszertifizierung nach ISO/IEC 5259“. (M11)

---

## Anhang A – Verworfene bzw. abgeschwächte Einwände

* **F4, Pseudocode-Vorwurf.** Der Vorwurf, `fit_predictable(seen_before_t)` in 04 §4.9 sei eine
  Gültigkeitsfalle, ist widerlegt: Der Aufruf schränkt explizit auf Daten vor t ein. Offen bleibt
  nur, dass die β-Obergrenze in `BettingCS` unspezifiziert ist.
* **F4, Nebenbemerkung zu `sim2.py`.** Formal richtig, praktisch ohne Wirkung: In 120 gutartigen
  S1-Pools gab es 0 Verletzungen.
* **F5 (ii).** „Unter dem von 06 vorgeschriebenen π verzerrt“ greift eine Verwendung an, die 06 nicht
  macht (06 verweist auf eine IPW-Variante). Sie fehlt allerdings.
* **F3, Teilbehauptung.** Dass „ein Leck, das alle Detektoren gleich anhebt“, in der untestbaren
  00-Richtung liege, ist widerlegt. Logit-Lecks werden erkannt, nur proportionale Abweichungen nicht.
  „Ausweichregel löschen“ ist zu stark, weil sie in Leck-Fällen hilft. Sie ist unbegründet, aber
  nicht universell falsch.
* **F1, Lösungsvorschläge.** Der Flächenanteil ist gültig, aber leer (≥ 1,9·10⁵ Ziehungen). Die
  mittlere Missquote pro Einheit ist blind für Pitting (siehe §6, umstritten). Die Schwere „kritisch“
  gilt nur für 00, nicht für 04.
* **M5, Lösungsvorschlag.** `drpop` behebt keine Korrelationsverzerrung. „Kein Test auf beobachteten
  Zellen erkennt die fünfte Zelle“ ist zu stark (B 0,57, C 0,69 Ablehnung). Nur reine h = 0-Verletzungen
  sind unsichtbar.
* **M3, Lösungsvorschlag.** AQL-Formulierung verworfen, LTPD/LQ ist korrekt.
* **M6, Lösungsvorschlag.** a90/95 aus Injektionen als Zertifikat verworfen, nur modellbasiert.
* **M10, Lösungsvorschlag.** Synthetische Seeds als Korrekturfaktor verworfen, nur Monitor. Echte
  transplantierte Defekte sind vorzuziehen. Achtung Prävalenzeffekt (Wolfe et al., Nature 2005,
  unverified).
* **C5.** Die Rettung von „um Größenordnungen“ über die nötige Fläche statt die LCB ist gescheitert
  (Faktor 1,1–2,5).
* **A1 (Machbarkeit), Vergleichspunkt.** „Gold-only n = 140“ gehört zu einem anderen Ziel. Der
  korrekte Vergleich (n = 250 für Power 0,881) stärkt die Aussage.
* **A2 (Machbarkeit).** Gleiche Ursache wie A1, daher eher „hoch“ als eigenständig „kritisch“.
* **A3 (Machbarkeit).** „PPS deutlich schlechter“ gilt nur fern vom Ziel. Nahe am Ziel liegt PPS
  gleichauf.
* **A5.** „Mehrtägige Schleifen“ ist überzogen (Retraining 2–5 h CPU). Eine Review-Erweiterung
  braucht kein neues δ.
* **A6.** „27 Läufe ≈ 34 h“ überzogen (≈ 24 h). „S5 auf M4 unmöglich“ überzogen, nur LODO ist es.
* **A10.** „Gültigkeit kann in der Simulation nicht scheitern“ ist zu stark, nur der Menschfehler-Pfad
  ist unsichtbar.
* **A1 (Alternativen).** Red-Team (e) hat 70 % Sensitivität *angenommen*, nicht gezeigt.
* **A2 (Alternativen).** „Restmisses sind per Konstruktion unrankbar“ ist überzogen, die Empirie trägt
  die Aussage. LITE-SRS nutzte Orakelplanung, der faire Vergleich ist LITE-GS.
* **A3 (Alternativen).** „Höchstens 8–16 %“ ist keine Obergrenze. „< 0,2 pp N̂-Skaleneffekt“ gilt nur
  bei 15 versteckten Misses.
* **A4 (Alternativen).** „PPR so billig wie fixes n“ ist widerlegt. „Frühes Stoppen ist der einzige
  Nutzen“ übersieht die Robustheit gegen eine falsche Prognose.
* **A7 (Machbarkeit).** Doppeltes Cross-Fitting aufzuschieben ist eine Wette, deshalb das
  Phase-1-Gate.

## Anhang B – Reproduktion

```bash
python docs/research/sims/redteam_capture_recapture.py        # Durchgang 1, 06
python docs/research/sims/redteam_audit_checks.py             # Durchgang 1, 04
python docs/research/sims/redteam2_betting_validity.py        # F1, F4, F6, F10
python docs/research/sims/redteam2_transport_identifiability.py  # F3, F13 (Szenarien E, F, G)
python docs/research/sims/redteam2_satz2_tiles.py             # F2, F12
python docs/research/sims/redteam2_verify_math_checks.py      # Gegenprüfung F2–F5
python docs/research/sims/redteam2_verify_feasibility_checks.py   # Gegenprüfung A1–A6, A10
python docs/research/sims/redteam2_verify_alternatives_checks.py  # Gegenprüfung A1–A5 (Alternativen)
python docs/research/sims/redteam2_verify_novelty_ltpd_prop3.py   # M3 (LTPD), C2 (Prop. 3 vs. Neyman)
python docs/research/sims/redteam2_verify_novelty_tiles.py        # C5
```

Alle Skripte laufen auf der CPU in unter einer Minute bis wenigen Minuten. Die Skripte der Linsen
Machbarkeit und Alternativen (`feas*.py`, `lite_*.py`, `c1_value.py`, `n90_table.py`,
`fixedn_vs_seq.py`, `run_ppr.py`, `p1_lin.py`) liegen mit dem Präfix `redteam2_lens_` in
[`sims/`](sims/) (Korrektur: die erste Fassung sagte hier, sie lägen nicht im Repository; das
widersprach dem Kopf des Berichts). Ihre Zahlen sind, soweit nicht gegenverifiziert, weiterhin mit [N]
markiert, weil keine adversariale Zweitprüfung stattfand.
