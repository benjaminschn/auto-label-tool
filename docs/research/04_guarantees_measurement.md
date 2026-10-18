# 04 – Statistische Garantien und Messung der Auto-Label-Qualität mit minimalem menschlichem Aufwand

Stand: 18.10.2026 · Literatur-Review, eigene Theorie und Protokollvorschlag für `auto-label-tool`.

> **Quellenlage (bitte lesen).** In dieser Umgebung waren arXiv/OpenReview/Verlagsseiten per Proxy gesperrt. Das
> Web-Such-Budget (geteilt mit parallelen Reviewern) war nach ca. 20 Suchen erschöpft. Verifiziert wurden
> deshalb (a) Existenz, Autoren und Kernaussagen per Websuche und (b) Algorithmen direkt im **offiziellen Code**
> auf GitHub: PAC-Labels (`tijana-zrnic/pac-labels/pac_utils.py`), CRC (`aangelopoulos/conformal-risk/core/get_lhat.py`),
> PPI++ (`aangelopoulos/ppi_py/ppi_py/ppi.py`) und Active Inference (`tijana-zrnic/active-inference`).
> Theoreme anderer Arbeiten sind aus Fachwissen wiedergegeben und im Text bzw. in den Referenzen mit **(unverified)**
> markiert. Die Simulationen in §3 sind echte, kleine Läufe (Skripte nur im Scratchpad, nicht im Repo).

---

## 0. Executive Summary

1. **Zwei Arten von Garantie müssen getrennt werden.** *Modellbasierte* Garantien (Conformal/CRC auf L) gelten nur,
   wenn L eine Zufallsstichprobe aus dem Pool ist und die Vorhersagen auf L und U vom selben Mechanismus stammen.
   Beides ist bei uns verletzt. *Designbasierte* Garantien (randomisiertes Audit mit bekannten
   Auswahlwahrscheinlichkeiten) brauchen **keine** Austauschbarkeit und **keine** Annahme über Komponenten-Abhängigkeit.
   Sie sollten die zertifizierende Säule des Tools sein.
2. **Die aktuelle Recall-Garantie ist für U nicht gültig.** Gründe: (i) Die Schwelle wird auf Einzel-Member-OOF
   kalibriert, aber auf den Ensemble-Mittelwert angewandt; es gilt max-of-mean ≤ mean-of-max. In einem Toy-Modell
   steigt die Missrate dadurch bei Soll α=0,1 auf 0,12–0,32. (ii) Ab Runde 1 und für Synthese-Hintergründe sind die
   U-Vorhersagen *in-sample*, mit genau den übersehenen Defekten als Negativ-Label. (iii) Defekte im selben Bild sind
   abhängig. (iv) `t_seed` wird datenabhängig gewählt.
3. **Mit n≈11 Defekten in ≈10 Bildern ist keine nicht-triviale PAC- oder bildweise CRC-Garantie möglich.** Die marginale
   Garantie hat bedingt auf L eine Abdeckung ~Beta(11,1): P(Recall < 0,9) = 0,31. Das Tool sollte das so berichten.
4. **Die aktuellen Audit-Intervalle (t-Quantil + Delta-Methode) unterschreiten die Soll-Abdeckung bei seltenen Defekten.**
   Eigene Simulation, einseitig 95 %: audit-only teils nur 0,23–0,69, PPI (λ=1) bis hinab auf 0,83–0,88. Exakte bzw. Betting-Schranken
   (hypergeometrisch, WSR) halten 0,95–1,00.
5. **PAC-Labels (Candès, Ilyas, Zrnic; ICML 2026) ist das passende Grundgerüst.** Es nutzt IPW-Stichproben mit
   bekannten π, eine Betting-Obergrenze und eine Schwelle auf der Unsicherheit. Die Garantie lautet „Fehler des
   *gesamten* finalen Datensatzes ≤ ε mit W. ≥ 1−α“. Wir erweitern es auf Segmentierung: Bild = Einheit,
   Verhältnis-Metriken über Fieller-Inversion, Pixelschwelle und Review-Schwelle per Fixed-Sequence-Testing.
6. **Review und Audit sind dieselbe menschliche Handlung.** Jedes geprüfte Bild wird korrigiert *und* misst. Ein
   einheitliches Design mit Inklusionswahrscheinlichkeiten π_i (π_i=1 ⇒ Review-Stratum) plus der Identität
   „finale Fehler = Populationssumme − beobachtete korrigierte Fehler“ liefert unverzerrte, zertifizierte Aussagen
   über den **korrigierten** Pool. Die optimale Aufteilung hat eine geschlossene Form (§4.4, Idee N1).
7. **Anytime-valid Auditing** mit Betting-Konfidenzsequenzen (Waudby-Smith & Ramdas) erlaubt es, jederzeit zu stoppen,
   sobald das Ziel zertifiziert ist. λ (PPI-Power-Tuning), Kalibrierung der Selbsteinschätzung und
   Auswahlwahrscheinlichkeiten dürfen dabei *prädiktabel* (nur aus der Vergangenheit) angepasst werden. Das löst die
   Unterdeckung des getunten λ.
8. **Harte Grenze: das Heuhaufen-Problem.** Hohen Defekt-Recall in einem Pool mit seltenen Defekten zu zertifizieren,
   verlangt im „vorhergesagt sauber“-Stratum Audits der Größenordnung N_c·ln(1/δ)/(ε·D_tot). Beispiel: 2000 saubere
   Bilder, ≤7 Fehlbilder ⇒ 640 Audits. Primär zertifizierbar ist daher die **Rate der Bilder mit übersehenem Defekt**
   (binär, effiziente exakte Schranken). Defekt-Recall ist sekundär (nichtasymptotisch konservativ, asymptotisch
   approximativ).
9. **Downstream:** Eine zertifizierte Label-Fehlerrate begrenzt die Lücke zwischen gemessenem und wahrem Fehler eines
   Produktivmodells (|R_clean − R_noisy| ≤ η). Sie garantiert aber nicht dessen Qualität, weil Pseudo-Label-Rauschen
   instanzabhängig ist. Ehrlicher Weg: Produktivmodell auf einem zurückgehaltenen Teil des Audits designbasiert messen.

---

## 1. Ausgangslage im Code (präzise)

* `conformal.py`: `defect_scores` = max p in jeder GT-Komponente (dilatiert, tol=2) auf **OOF-Vorhersagen eines
  Members**. `conformal_threshold` = s_(⌊α(n+1)⌋) ist korrekt für eine *fest vorab* gewählte Rangregel. `pac_threshold`
  = größte Schwelle mit Clopper-Pearson-UCB ≤ α (RCPS-artig, korrekt unter iid Defekten). `certified_recall(scores, t)`
  meldet 1−(m+1)/(n+1) für **beliebiges** t.
* `pipeline.py`: Schwellen (`select_thresholds`, F1-Gridsuche über 7 Seeds × 4 Ratios × 3 Flächen) und konforme Schwelle
  auf OOF. Masken auf U aus `mean_U` (Ensemble-Mittel über K Member + Flip-TTA). Runde ≥1: Member k trainiert auf
  Pseudo-Labels **aller** U-Bilder (Teacher k). Runde 0: die 70 % normalsten U-Bilder dienen als Synthese-Hintergründe
  mit Label „negativ“ (außer 98-%-Anomalie-Ignore).
* `ppi.py`: Differenzschätzer für endliche Populationen, λ=1 Default, λ-Tuning = Cov/Var (in-sample), FPC (1−n/M),
  t-Quantil mit n−1 FG, Ratio via Delta-Methode.
* `cli.py`: Audit = uniforme Stichprobe fester Größe (Default 30) aus U, vor dem Review gezogen. Das Review ist
  separat, und die Qualität des *korrigierten* Pools wird nicht geschätzt.

---

## 2. Literatur-Review

### 2.1 Auto-Labeling mit Garantien

**Threshold-based auto-labeling (TBAL).** Vishwakarma, Lin, Sala, Vinayak (NeurIPS 2023) analysieren iterative
Systeme, die einen Teil der Daten aktiv labeln, ein Modell trainieren und dann auf einem menschlich gelabelten
Validierungsset pro Klasse eine Konfidenzschwelle wählen. Alles oberhalb der Schwelle wird automatisch gelabelt.
Kernaussagen (verifiziert per Abstract): Schon schwache Modelle können große Teile korrekt labeln, aber **TBAL kann
prohibitiv viel Validierungsdaten benötigen**. Die Autoren geben Sample-Complexity-Schranken für die Qualität der
maschinell gelabelten Daten an. Form der Schwellenwahl (unverified im Detail): kleinste Schwelle t mit
ε̂(t) + C·σ̂(t) ≤ ε_a auf den Validierungspunkten oberhalb t. Die Garantie betrifft den **Fehler unter den
auto-gelabelten Punkten** (selektives Risiko), mit Zusatzterm O(√(log(1/δ)/n_v(t))), wobei n_v(t) die Zahl der
Validierungspunkte oberhalb t ist. Für uns entscheidend: Die Validierungsdaten stammen aus *demselben* Pool und werden
mit *demselben* Modell ausgewertet. Das ist genau die Eigenschaft, die unserem OOF→Ensemble-Transfer fehlt.

**Pearls from Pebbles / Colander** (Vishwakarma et al., NeurIPS 2024). Die Arbeit lernt post-hoc eine
Konfidenzfunktion, die die Abdeckung unter einer Fehlerschranke maximiert, und berichtet bis +60 % Abdeckung bei
Fehler < 5 % (verifiziert). Standard-Kalibrierung reicht für TBAL nicht. Lehre für uns: Die Rangfolge der
Selbsteinschätzung (Review-Score) sollte direkt für das Zielkriterium gelernt werden, z. B. für „Bild enthält
übersehenen Defekt“. Das kostet allerdings gelabelte Daten, die wir nur aus Audits haben (→ prädiktables Lernen,
§4.5).

**Selective classification with guaranteed risk (SGR)** (Geifman & El-Yaniv, NeurIPS 2017; unverified): binäre Suche
über Konfidenzschwellen mit Binomial-Tail-Schranke und Bonferroni über ⌈log₂ m⌉ Schritte. Garantie:
P(selektives Risiko > r*) < δ. Laut Conformal-Labeling-Paper hält SGR in ihren Experimenten die FDR nicht ein
(42,6 % statt 10 %; verifiziert per Zusammenfassung).

**PAC Labels** (Candès, Ilyas, Zrnic; arXiv 2506.10908, ICML 2026). Im offiziellen Code verifiziert:

* Es werden `num_draws` Indizes i_k **gleichverteilt mit Zurücklegen** gezogen, je mit ξ_k ~ Bernoulli(π_{i_k}).
  π darf gleichförmig oder unsicherheitsproportional sein. Gezogene Punkte mit ξ_k=1 erhalten Expertenlabels.
* Für eine Unsicherheitsschwelle u gilt Z_k(u) = ℓ(Y_{i_k}, Ŷ_{i_k})·ξ_k/π_{i_k}·1{U_{i_k} < u}. Dann ist
  E Z_k(u) = L(u) = (1/n)Σ_i ℓ_i·1{U_i<u}: der Fehler des gesamten Datensatzes, wenn alles mit U ≥ u vom Experten
  kommt.
* Die obere Schranke ist nichtasymptotisch per WSR-Betting (Bereich [0, max ℓ/π_min]) oder asymptotisch per CLT.
  Binärsuche liefert das größte u mit UCB(u) ≤ ε. Danach werden **alle** Punkte mit U ≥ û vom Experten gelabelt.
* Garantie: P(L(û) ≤ ε) ≥ 1−α, also „höchstens ε-Anteil des *gesamten* Datensatzes falsch gelabelt“. Erweiterungen:
  „PAC router“ über mehrere Modelle und gelernte Kalibrierung der Unsicherheit.

Strukturell ist das RCPS (Bates et al. 2021; unverified) auf einer **endlichen Population** mit IPW. Die Autoren
nennen selbst die Verwandtschaft zu *risk-limiting audits*. Für uns ist das die richtige Zielgröße: die Qualität des
Trainingssets, nicht der auto-gelabelten Teilmenge.

**Conformal Labeling** (arXiv 2510.14581, 2025; Autoren unverified). Für jeden Testpunkt wird ein konformer p-Wert
gebildet, indem seine Modellkonfidenz mit Kalibrierpunkten verglichen wird, die das Modell **falsch** gelabelt hat.
Dann folgt BH auf einem über die geschätzte Modellgenauigkeit angepassten Niveau. Ergebnis: verteilungsfreie,
endliche FDR-Kontrolle der auto-gelabelten Teilmenge (verifiziert per Zusammenfassung). Basis ist *Conformal
Selection* (Jin & Candès 2023). Für Segmentierung passt das als **bildweise Akzeptanzregel**: „Unter den als korrekt
akzeptierten Bildern ist der erwartete Anteil mit übersehenem Defekt ≤ q.“ Voraussetzung ist ein austauschbares
Kalibrierset mit Wahrheit, bei uns also Audit-Bilder (nicht L). Verwandte Arbeiten: Selektion aus hierarchischen Daten
mit konformen e-Werten (arXiv 2501.02514) und konforme Selektion nach datenabhängiger Selektion (Wang et al.,
NeurIPS 2024).

**Engines.** *Conformal Risk Control* (Angelopoulos, Bates, Fisch, Lei, Schuster; ICLR 2024). Im Code verifiziert:
λ̂ = inf{λ : (n/(n+1))·R̂_n(λ) + B/(n+1) ≤ α}. Garantie: E[L_{n+1}(λ̂)] ≤ α für monotone, nach oben durch B
beschränkte Verluste unter Austauschbarkeit, inklusive FNR-Kontrolle für Polypen-Segmentierung.
*Learn-then-Test* (Angelopoulos, Bates, Candès, Jordan, Lei; AoAS 2025): p-Werte für H_λ: R(λ) > α plus
FWER-Kontrolle (Bonferroni, Fixed-Sequence, Graph) ergibt eine simultan gültige Parametermenge, auch für nicht-monotone
Risiken und Instanzsegmentierung. *Pareto Testing* (Laufer-Goldshtein et al., ICLR 2023; unverified) kontrolliert
mehrere Risiken (FN und FP) effizient.

**Übertrag auf Segmentierung.** Drei Einheiten sind möglich:

* Pixel: viele, stark abhängig. Als Garantieeinheit ungeeignet.
* Komponente: Zieleinheit für Recall, aber geclustert.
* Bild: iid-/Austauschbarkeitseinheit.

Alle Garantien sollten **auf Bildebene** formuliert werden, mit **pro-Bild-Summen** von Komponenten- und Pixelgrößen
(Cluster-Stichprobe). Damit ist die Abhängigkeit innerhalb eines Bildes automatisch korrekt behandelt.

### 2.2 Konformes Vorgehen für Segmentierung

* **CRC-FNR** (s. o.): Schwelle λ auf Pixel-Wahrscheinlichkeiten, Verlust = FN-Anteil pro Bild. Kontrolliert wird der
  *mittlere Bild-FNR*. Garantieeinheit ist das Bild.
* **Mossina, Dalmau, Andéol (CVPR-W 2024)**: post-hoc CRC für semantische (Mehrklassen-)Segmentierung. Ergebnis sind
  Multi-Label-Masken bzw. Heatmaps mit Abdeckung der GT-Maske bei gewähltem Niveau (verifiziert).
* **Mossina & Friedrich (MICCAI 2025)**: *morphologische* Vorhersagemengen. Die Zahl der **Dilatations-Iterationen**
  wird konform so kalibriert, dass die dilatierte Vorhersage die GT abdeckt. Die Mengen sind wegen Extensivität
  geschachtelt; die Methode braucht nur Masken, keine Wahrscheinlichkeiten (verifiziert). Das duale
  **Erosions**-Verfahren für FP-Kontrolle (Mossina & Friedrich, arXiv 2511.15406) ist ebenfalls verifiziert.
  → Direkt verwendbar für unsere **Ignore-Ringe** (Idee N7).
* **Kandinsky CP**: zwei verschiedene Arbeiten. (a) Brunekreef et al. (CVPR 2024): Nichtkonformitätsscores werden über
  *ähnliche Bildregionen* gepoolt, zwischen bild- und pixelweiser Kalibrierung. Das ist dateneffizient bei wenigen
  Kalibrierbildern (verifiziert). (b) Bairaktari, Wu, Wu (ICML 2025): gruppenbedingte Abdeckung für überlappende,
  fraktionale Gruppen über Kovariaten *und* Labels, minimax-optimal (verifiziert). Für uns relevant als
  Recall-Garantie *pro Defektklasse/-größe* ohne disjunkte Mondrian-Zellen.
* **Hierarchische Austauschbarkeit**: Lee, Barber, Willett (2023; verifiziert) leiten eine hierarchische Form der
  Austauschbarkeit her (Gruppen iid, innerhalb austauschbar) und erweitern Conformal/Jackknife+ darauf. Dunn,
  Wasserman, Ramdas (JASA 2023; unverified) geben Mengen für Zwei-Ebenen-Modelle mit Garantie für eine zufällige
  Einheit einer *neuen Gruppe*. Praktisch heißt das: Jede Gruppe (Bild) erhält Gewicht 1, ihre Einheiten (Defekte)
  je 1/D_i. Das entspricht dem mittleren Bild-FNR, nicht dem gepoolten Recall.
* **Mehrklassen**: Mondrian-/klassenbedingtes CP; bei wenigen Beispielen pro Klasse hilft *clustered conformal*
  (Ding et al., NeurIPS 2023; unverified).
* **Kovariatenshift**: gewichtetes CP (Tibshirani et al., NeurIPS 2019; unverified) setzt ein **bekanntes**
  Likelihood-Verhältnis voraus. Bei uns ist L typischerweise *nutzerselektiert* (Bilder mit gut sichtbaren Defekten).
  Das Verhältnis ist unbekannt, und das Gewichtungsargument greift nicht. Ein Audit aus U mit bekannten π umgeht das.
* **Cross-Fitting**: Jackknife+/CV+ (Barber, Candès, Ramdas, Tibshirani, AoS 2021; unverified). Der Test vergleicht
  für jedes i die Vorhersage *des Modells, das i nicht sah*, und erreicht ≥ 1−2α (CV+ mit Korrekturterm
  min{2(1−1/K)/(n/K+1), (1−K/n)/(K+1)}). Die Minmax-Varianten erreichen ≥ 1−α. Jackknife+-after-Bootstrap
  (Kim, Xu, Barber, NeurIPS 2020; unverified) aggregiert über Bootstrap-Modelle, die i nicht sahen, und erreicht
  1−2α. CV-CRC (Cohen, Park, Simeone, Shamai 2024; verifiziert) überträgt die Jackknife-Minmax-Idee auf
  Risikokontrolle.
  **Antwort auf die Projektfrage:** Schwellen aus OOF-Vorhersagen von K Modellen, angewandt auf deren *Mittelwert*,
  haben **keine** bekannte Garantie. Gültig sind: (i) Vergleich pro Kalibrierpunkt mit *seinem* Fold-Modell (CV+);
  (ii) die **Union/Minmax-Regel** (Pixel-Maximum über Member, Prop. 1); (iii) Subsampling-Ensembles mit
  OOB-Aggregat (J+aB-Logik); (iv) Kalibrierung auf zufälligen U-Audits mit dem eingesetzten Ensemble.

### 2.3 Messung: PPI und aktive Stichproben

* **PPI** (Angelopoulos et al., Science 2023), **PPI++** (Angelopoulos, Duchi, Zrnic 2023). Im Code verifiziert:
  Zähler = symmetrisierte Kovarianz (2·Cov), Nenner = 2(1+n/N)·Var, also λ̂ = Cov/((1+n/N)·Var) für den Mittelwert. Der Faktor
  (1+n/N) entsteht, weil der Mittelwert der Vorhersagen auf N ungelabelten Punkten selbst geschätzt ist. **Für unsere
  endliche Population ist Σ_U f_i bekannt.** Optimal ist dann der Regressionsschätzer λ* = Cov(y,f)/Var(f) *ohne*
  diesen Faktor; `ppi._tuned_lambda` ist insoweit korrekt. Seine Unterdeckung kommt vom In-Sample-Tuning bei kleinem n.
* **Cross-PPI** (Zrnic & Candès, PNAS 2024): Prädiktor per Cross-Fitting auf den gelabelten Daten trainieren. Relevant,
  wenn die Selbsteinschätzung aus Audit-Daten gelernt wird.
* **Stratified PPI** (Fisch et al., NeurIPS 2024; unverified): λ und Allokation pro Stratum, Gewinn bei heterogener
  Modellgüte.
* **PPI mit nicht-uniformer Stichprobe** (arXiv 2501.18577; verifiziert über `ppi_py`-README) und
  **Active Multiple-PPI** (arXiv 2605.08429, 2026; Existenz verifiziert, Inhalt nicht).
* **Active Statistical Inference** (Zrnic & Candès, ICML 2024; Code verifiziert, inkl. sequentieller Variante mit
  Feintuning). Schätzer θ̂ = (1/n)Σ_i [f(X_i) + (Y_i − f(X_i))·ξ_i/π(X_i)] mit ξ_i ~ Bern(π(X_i)) und Budget
  E Σ ξ_i ≤ n_b. Unverzerrt für beliebiges bekanntes π. Varianzminimierend ist π(x) ∝ √E[(Y−f(X))² | X=x], praktisch
  ∝ Unsicherheit, gemischt mit Gleichverteilung (Theorem-Details unverified).
* **Active Testing** (Kossen et al., ICML 2021) und **Active Surrogate Estimators** (Kossen et al., NeurIPS 2022;
  beide unverified): sequentielle, nicht-uniforme Auswahl der Testpunkte mit Gewichtskorrektur (LURE, Farquhar et al.
  2021). Das ist das ML-Analogon zu Horvitz–Thompson.
* **Confidence-driven inference** (Gligorić, Zrnic, Lee, Candès, Jurafsky 2024; unverified): LLM-Konfidenz steuert,
  welche Punkte Menschen labeln, plus aktive Inferenz.
* **Semi-supervised risk control via PPI** (Einbinder, Ringel, Romano 2024; unverified): PPI-verschärfte
  LTT/RCPS-Schranken.

### 2.4 Anytime-valid / sequentielles Auditing

* **Betting-Konfidenzintervalle/-sequenzen** (Waudby-Smith & Ramdas, JRSS-B 2024; Implementierung im PAC-Labels-Code
  verifiziert). Für X_t ∈ [0,1] mit bedingtem Mittel μ ist K_t(m) = Π_{s≤t}(1 + λ_s(X_s − m)) bei prädiktablen λ_s
  ein nichtnegatives Martingal unter m = μ. Nach **Ville** gilt P(∃t: K_t(μ) ≥ 1/δ) ≤ δ. Daraus folgt die
  Konfidenzsequenz {m : K_t(m) < 1/δ}, gültig zu **jeder Stoppzeit**.
* **CS für Ziehen ohne Zurücklegen** (Waudby-Smith & Ramdas, NeurIPS 2020; unverified): endliche Populationen, wird
  schmal, wenn t → N.
* **Risk-limiting audits** (Stark: SHANGRLA 2020, ALPHA 2023; unverified): gewichtete, stratifizierte, sequentielle
  Wahlprüfungen mit Supermartingalen. Konzeptionell identisch mit einem „risk-limiting label audit“.
* **Active, anytime-valid RCPS** (Xu, Karampatziakis, Mineiro, NeurIPS 2024; unverified): Label-Anfragen mit
  prädiktablen Wahrscheinlichkeiten und Prädiktoren als Kontrollvariate, e-Prozesse für Risikokontrolle. Am
  nächsten an unserem Protokoll.
* **Asymptotische CS** (Waudby-Smith et al., time-uniform CLT; unverified): breiten-effizient ohne
  Beschränktheitsannahme, aber nur approximativ.

### 2.5 Downstream-Garantien (kurz)

* Für 0-1-Verlust und jeden Klassifikator h gilt (Dreiecksungleichung):
  |P(h(X)≠Y) − P(h(X)≠Ỹ)| ≤ P(Ỹ≠Y) = η. Eine zertifizierte Label-Fehlerrate begrenzt also die
  **Messverzerrung**, wenn das Produktivmodell auf Pseudo-Labels evaluiert wird. Sie begrenzt **nicht** dessen
  wahren Fehler.
* Klassenbedingtes Rauschen (Natarajan et al., NeurIPS 2013; unverified): η̃(x) = (1−ρ₊−ρ₋)η(x) + ρ₋. Die
  Bayes-Schwelle verschiebt sich bekannt, und Verlustkorrektur ist konsistent. Konsequenz bei FN-lastigen
  Pseudo-Labels (ρ₊ > 0): Die Produktivschwelle muss **sinken**, und eine zertifizierte Schranke für ρ₊ begrenzt die
  nötige Verschiebung.
* Pseudo-Label-Fehler sind aber **instanzabhängig**: Übersehen werden genau die schweren Defekte. Dafür gibt es keine
  allgemeine Transferschranke. Self-Training-Theorie (Wei et al., ICLR 2021; unverified) braucht Expansionsannahmen,
  die nicht prüfbar sind.
* **Empfehlung:** Das Produktivmodell auf einem **nicht trainierten** Teil des Audits designbasiert messen (gleiche
  Maschinerie wie §4.5).

---

## 3. Eigene Vorab-Simulationen (echte Läufe, klein)

**Modell.** N=2000 Bilder. Defektprävalenz p ∈ {0,03; 0,10; 0,30}, D_i = 1+Poisson(0,6) bei Defektbildern, D_max=6.
Jeder Defekt wird mit Wahrscheinlichkeit „Sichtbarkeit“ ~ Beta(4;1,2) erkannt, sodass der wahre Recall ≈ 0,77 ist.
Die Selbsteinschätzung kommt aus Kandidatenregionen; 30 % der übersehenen Defekte haben keinen Kandidaten, und die
Selbsteinschätzung nimmt bewusst falsch 9 % an. Es gibt 300 Wiederholungen, und alle Schranken sind einseitig mit
Sollniveau 95 %. Die Tabellen zeigen Abdeckung (mittlere Schranke).

**(S1) Defekt-Recall ρ = 1 − ΣM/ΣD, untere Schranke (LCB)**

| p | n | wahres ρ | audit-only (t, Delta) | PPI λ=1 (aktuell) | IPW-CLT-Fieller | Betting-Fieller (PPS, γ=0,3) |
|---|---|---|---|---|---|---|
| 0,03 | 30 | 0,77 | 0,69 (0,40)* | 1,00 (0,21) | 0,89 (0,59) | 1,00 (0,00) |
| 0,03 | 100 | 0,76 | 0,66 (0,59) | 0,99 (0,44) | 0,94 (0,62) | 1,00 (0,15) |
| 0,10 | 30 | 0,77 | 0,67 (0,56) | 0,95 (0,56) | 0,83 (0,65) | 1,00 (0,13) |
| 0,10 | 100 | 0,77 | 0,86 (0,61) | 0,91 (0,65) | 0,89 (0,70) | 1,00 (0,35) |
| 0,30 | 30 | 0,77 | 0,85 (0,59) | 0,88 (0,65) | 0,86 (0,68) | 1,00 (0,28) |
| 0,30 | 100 | 0,77 | 0,94 (0,67) | 0,93 (0,69) | 0,88 (0,72) | 1,00 (0,51) |

\*43 % der Läufe undefiniert (kein Defekt im Audit). Diese Läufe sind als LCB=0 gezählt, die echte Abdeckung ist also
noch schlechter. Die Spalte IPW-CLT-Fieller stammt aus einem separaten Lauf mit demselben Modell und PPS-Design
(γ=0,3, 200 Wdh.).

**(S2) Rate der Bilder mit übersehenem Defekt ζ, obere Schranke (UCB)**

| p | n | ζ | audit-only t | PPI λ=1 (aktuell) | exakt hypergeom. | PPI + WSR-Betting |
|---|---|---|---|---|---|---|
| 0,03 | 30 | 0,010 | 0,23 (0,023) | 0,98 (0,038) | 1,00 (0,108) | 1,00 (0,141) |
| 0,03 | 100 | 0,010 | 0,62 (0,023) | 0,94 (0,027) | 1,00 (0,045) | 1,00 (0,055) |
| 0,10 | 30 | 0,033 | 0,66 (0,080) | 0,83 (0,078) | 1,00 (0,147) | 1,00 (0,168) |
| 0,10 | 100 | 0,033 | 0,88 (0,062) | 0,90 (0,059) | 0,98 (0,079) | 1,00 (0,083) |
| 0,30 | 30 | 0,10 | 0,81 (0,181) | 0,87 (0,168) | 0,95 (0,229) | 1,00 (0,243) |
| 0,30 | 100 | 0,10 | 0,89 (0,147) | 0,89 (0,137) | 0,95 (0,160) | 0,98 (0,160) |

**(S3) Schwellentransfer, Toy-Modell.** Ein dünner Defekt hat 30 Pixel und K=4 Member. Das pixelweise Rauschen hat
einen gemeinsamen Anteil s ∈ {0; 0,5; 0,8}. Die Schwelle ist das α=0,1-Quantil der Einzel-Member-Scores (= OOF).
Gemessene Missrate:

* **Ensemble-Mittel:** 0,32 / 0,17 / 0,12
* **Union** (Pixel-Max über Member): 0,035 / 0,040 / 0,053

Das Toy-Modell ignoriert, dass das Ensemble auf mehr Daten trainiert ist. Die Richtung (antikonservativ) folgt aber
aus max-of-mean ≤ mean-of-max, sobald die Peaks der Member räumlich streuen.

**(S4) Exakte Zahlen.**

* Split-Conformal mit n=11, α=0,1: bedingte Abdeckung ~ Beta(11,1). P(<0,9)=0,31, P(<0,8)=0,086, P(<0,7)=0,020.
* Clopper-Pearson-PAC mit 0 Fehlern braucht n ≥ 22 (δ=0,1) bzw. ≥ 29 (δ=0,05) *unabhängige* Defekte.
* Heuhaufen: 2000 vorhergesagt-saubere Bilder, 0 Fehlbilder im Audit. Die 95-%-UCB der Zahl der Fehlbilder ist 188
  (n=30), 57 (n=100), 18 (n=300). Für ≤7 braucht man n=640.

**Interpretation.**

* t-/Delta-Intervalle sind im relevanten Regime (kleines n, seltene Defekte) nicht vertrauenswürdig. PPI λ=1 ist
  besser als audit-only, aber nicht durchgehend gültig.
* Exakte bzw. Betting-Schranken sind gültig. Für **binäre Bildmetriken** sind sie praktisch brauchbar, für
  **Defektzahl-Verhältnisse** mit D_max=6 bei n ≤ 100 fast leer. Ein einzelnes Bild könnte 6 übersehene Defekte
  tragen; die Betting-Schranke muss das zulassen, die CLT-Schranke ignoriert es.
* Daraus folgen im Protokoll die Primärmetrik ζ, kleine D_max über Kachel-Einheiten (N6) und eine klare
  Kennzeichnung approximativer Recall-Intervalle.

---

## 4. Protokoll „Certified Auto-Labeling for Segmentation“ (CALS)

### 4.1 Einheiten, Notation, Zielgrößen

Pool U mit N Bildern; gelabelte Menge L. Für jedes i ∈ U gibt es das Pseudo-Label Ŷ_i, die unbekannte Wahrheit Y_i,
den Risikoscore r_i und Selbsteinschätzungen f_i^• (erwartete Misses f^M, erwartete Defekte f^D, erwartete falsche
Komponenten f^F, P(übersehener Defekt) f^b) sowie Varianzproxies σ_i. Pro Bild mit Toleranz τ=2 px und Merge-Radius:

* D_i: GT-Komponenten; M_i: davon nicht getroffen
* P_i: vorhergesagte Komponenten; F_i: davon ohne GT
* b_i = 1{M_i > 0}
* Toleranz-Pixel TP_i, FP_i, FN_i

Vorab registrierte Schranke: D_i ≤ D_max.

Menschliche Prüfung einer Menge S ⊆ U: Das Bild wird vollständig korrigiert (Annahme: Mensch = Goldstandard).
R ⊆ S ist das deterministische Review-Stratum, U' = U \ R mit N' = |U'|.

Finale Labels: Ỹ_i = Y_i für i ∈ S, sonst Ŷ_i. Zielgrößen auf Pool-Ebene:

* Defekt-Recall ρ = 1 − Σ_{i∉S} M_i / Σ_{i∈U} D_i
* Fehlbildrate ζ = (1/N) Σ_{i∉S} b_i
* falsche Komponenten pro Bild φ = (1/N) Σ_{i∉S} F_i
* gepoolter Toleranz-Dice

**Lemma 0 (Monotonie).** Korrekturen verringern M, b und F und lassen D unverändert. Also gilt
ρ_final ≥ ρ⁰ := 1 − Σ_{U'} M_i / (D_R + Σ_{U'} D_i), wobei ρ⁰ der Zustand nach Review und vor Audit-Korrekturen
ist. Analog gilt ζ_final ≤ ζ⁰ und φ_final ≤ φ⁰. ρ⁰, ζ⁰ und φ⁰ sind **feste** Populationsgrößen. Schranken für sie
sind konservative Schranken für den finalen Pool. Für Komponenten-*Präzision* gilt die Monotonie wegen
Fragmentierung nicht. Dort wird die Präzision vor Korrektur zertifiziert (Fieller wie unten), oder man nutzt φ.

### 4.2 Garantietypen und Annahmen

| Typ | Aussage | Annahmen |
|---|---|---|
| **G-D (designbasiert, primär)** | P(∀t: ρ⁰ ≥ LCB_t, ζ⁰ ≤ UCB_t, …) ≥ 1−δ für den **konkreten Pool** | Zufallsziehung nach dokumentiertem Design mit bekannten q_t(i) ≥ q_min > 0 im Rahmen U'; Mensch korrekt; Beschränktheit y_i ∈ [a,b] (z. B. D_max); keine Rückkopplung der Audit-GT in die zu zertifizierenden Labels |
| **G-P (PAC-Akzeptanz)** | Mit Review aller r_i ≥ û gilt P(ρ_final ≥ 1−ε) ≥ 1−δ | wie G-D, plus Fixed-Sequence-Struktur |
| **G-M (modellbasiert, sekundär)** | E[M_neu] ≤ α·E[D_neu] für ein neues Bild des Pools bzw. P(Miss) ≤ 2α + c | Austauschbarkeit der Bilder zwischen Kalibrierung und Ziel; bei L-Kalibrierung zusätzlich voller Cross-Fit (Idee N2) |

Wichtig: G-D braucht **keine** Austauschbarkeit von Bildern oder Defekten. Die Gültigkeit kommt allein aus der
Randomisierung; Abhängigkeit innerhalb eines Bildes beeinflusst nur die Varianz. G-M ist ein Komfortprodukt für die
`labels_recall_safe/`-Ausgabe *vor* jedem Audit und muss als annahmebasiert gekennzeichnet werden.

### 4.3 Stufe A – modellbasierte Schwellen (vor dem Audit)

**A1. Bildweise CRC statt Defekt-Pooling.** Die Verlustfunktion pro Bild ist ℓ_i(t) = M_i(t) − α·D_i ≤ (1−α)·D_max =: B.
Sie ist monoton: Sinkt t_seed (mit t_ext ← min(t_ext, t_seed), fester min_area), sind die Hysterese-Masken
geschachtelt, und M_i(t) fällt. Regel:

$$\hat t = \max\Big\{t:\ \textstyle\sum_{i=1}^{n} M_i(t) + B \le \alpha \sum_{i=1}^n D_i\Big\}.$$

**Prop. 2 (aus CRC).** Unter Austauschbarkeit der n+1 Bilder gilt E[M_{n+1}(t̂)] ≤ α·E[D_{n+1}]. Das ist eine
Recall-Garantie „im Verhältnis der Erwartungen“, der natürliche Grenzwert von ΣTP/ΣD.

*Beweisskizze:* CRC mit λ = −t auf ℓ ∈ (−∞, B]. Für D_i ≡ 1 reproduziert die Regel die bisherige Split-Conformal-
Schwelle s_(⌊α(n+1)⌋).

*Ehrlichkeitsfolge:* Mit 10 Bildern, 11 Defekten und α=0,1 verlangt die Regel ΣM ≤ 1,1 − 0,9·D_max. Schon für
D_max=2 ist das unerfüllbar ⇒ t̂=0. Das Tool muss dann „keine modellbasierte Garantie möglich“ melden (heute meldet
es 0,086 als 90-%-Schwelle auf Defektebene). Alternative ist der mittlere Bild-FNR ℓ_i = M_i/D_i mit B=1, der der
hierarchischen Gewichtung entspricht.

**A2. Transfer auf U korrekt machen.** Zwei Optionen:

* (a) **Union-Regel:** Maske aus p^∪(x) = max_k p^k(x) mit gepoolter OOF-Schwelle t = s_(ℓ), ℓ = ⌊α(n+1)⌋.
* (b) **Voller Cross-Fit über L ∪ U** (Idee N2): Member k berührt L_k ∪ U_k in keiner Rolle, also weder als Label noch
  als Synthese-Hintergrund noch als Pseudo-Label, und sagt L_k sowie U_k vorher.

**Prop. 1 (Union ⇒ CV+-Ereignis).** Sei s_i der OOF-Score von Kalibrierbild i (Modell k(i)), s^k_neu der Score des
neuen Bildes unter Member k, und keine Bindungen. Wird der Defekt mit der Union-Regel verpasst
(max_k s^k_neu < s_(ℓ)), dann gilt Σ_i 1{s_i > s^{k(i)}_neu} ≥ n − ℓ + 1 ≥ (1−α)(n+1).

*Beweis:* Für jeden Fold k ist s^k_neu < s_(ℓ). Höchstens ℓ−1 Kalibrierscores liegen unter s_(ℓ), alle anderen sind
größer als das jeweilige s^{k(i)}_neu. □

Das ist exakt das Fehlüberdeckungsereignis im CV+-Beweis von Barber et al. Daraus folgt
P(Miss) ≤ 2α + min{2(1−1/K)/(n/K+1), (1−K/n)/(K+1)}, Konstante unverified. In (S3) liegt die realisierte Missrate
sogar unter α. Kosten: mehr FP, daher nur für die `recall_safe`-Ausgabe, nicht für die Hauptlabels.

**A3. Ensemble-Mittel nur mit eigener Kalibrierung.** Soll das Ensemble-Mittel die Hauptmaske bleiben, muss seine
Schwelle auf Vorhersagen *desselben Mechanismus* kalibriert werden: entweder auf Audit-Bildern (Stufe D, Satz 2) oder
mit Subsampling-Ensembles (B Member auf je ~50–60 % von L). Im zweiten Fall ist jedes L-Bild für ~B/2 Member OOB, und
das OOB-Mittel ähnelt dem U-Mittel; das ist die J+aB-Logik.

### 4.4 Stufe B – Budgetaufteilung Review ↔ Audit („Measure-and-Repair“)

Jedes inspizierte Bild wird korrigiert *und* liefert Wahrheit. Unter Poisson-Design mit Inklusionswahrscheinlichkeiten
π_i und Differenzschätzer T̂ = Σ f_i + Σ_{i∈S} (y_i − f_i)/π_i gilt:

* erwartete Restfehler nach Korrektur: Σ (1−π_i)·m_i mit m_i = E[M_i] ≈ f^M_i
* Varianz der Schätzung: Σ (1/π_i − 1)·σ_i² mit σ_i² = E[(y_i−f_i)²]

**Prop. 3.** Das Problem

$$\min_{\pi\in(0,1]^N}\ \sum_i (1-\pi_i)\,m_i\ +\ \kappa \sum_i \Big(\frac1{\pi_i}-1\Big)\sigma_i^2\quad\text{s.t.}\ \sum_i \pi_i = B$$

hat die Lösung

$$\pi_i^* = \min\Big\{1,\ \sigma_i\sqrt{\kappa/(\nu - m_i)}\Big\}\ \text{für}\ m_i < \nu,\qquad \pi_i^*=1\ \text{sonst},$$

wobei ν (Lagrange-Multiplikator) das Budget löst.

*Beweis:* Die Zielfunktion ist konvex in π_i (κσ²/π ist konvex). KKT liefert −m_i − κσ_i²/π_i² + ν = 0 im Inneren. □

**Folgerungen.**

* (i) Das **Take-all-Stratum** (= Review) ist {i : m_i + κσ_i² ≥ ν}. Die Review-Priorität ist also
  „erwartete Fehler + κ·Unsicherheit“, eine begründete Alternative zum heutigen Rang-Mittel.
* (ii) Alle übrigen Bilder werden mit π ∝ σ_i/√(ν−m_i) gezogen, dem Neyman-/Active-Inference-Teil.
* (iii) κ→0 ergibt reines Greedy-Review ohne Messmöglichkeit; κ→∞ ergibt reine aktive Inferenz.
* κ wird so gewählt, dass die vorhergesagte Halbbreite z_δ·√V(κ) mit V(κ)=Σ(1/π_i*−1)σ_i² das Ziel h erreicht.
* Fehler in m_i und σ_i kosten nur Effizienz, nie Gültigkeit. Ein Uniform-Floor π_i ≥ π_min (z. B. 30 % des
  Ziehungsbudgets gleichverteilt) schützt gegen *unsichtbare* Misses, für die f nichts weiß.

**Varianzproxies aus vorhandenen Größen** (CandidateModel). Für Kandidaten c mit p_c = P(echter Defekt | max p)
gilt M_i ≈ Σ_{c unsel} Bern(p_c) + N_i (Misses ohne Kandidat), D_i ≈ Σ_c Bern(p_c) + N_i. Für die
Recall-Ratiofunktion g_i(r) = M_i − (1−r)D_i folgt

$$\sigma_i^2(r) = r^2\sum_{c\ \text{unsel}}p_c(1-p_c) + (1-r)^2\sum_{c\ \text{sel}}p_c(1-p_c) + r^2\,\widehat{\mathrm{Var}}(N_i).$$

Bei r ≈ 0,9 dominieren *unselektierte* Kandidaten (Gewicht 0,81 gegenüber 0,01). Die Audit-Stichprobe für Recall
gehört also zu Bildern mit unsicheren, *nicht* gelabelten Kandidaten und anomalen Restregionen.

### 4.5 Stufe C – sequentielles, designbasiertes Audit (G-D)

**Design.** R wird zuerst reviewt. Danach wird in Schritt t = 1, 2, … ein Bild i_t ∈ U' mit Zurücklegen nach q_t
gezogen; q_t ist F_{t−1}-messbar, und es gilt q_t(i) ≥ γ/N'. Bei neuem Bild wird inspiziert und korrigiert, bei
Wiederholung ist die Ziehung kostenlos. Das ist die PAC-Labels-Ziehung mit nicht-uniformem q. Protokolliert werden
Seed, q_t und Ziehfolge (Audit-Trail wie bei RLAs).

**Terme.** Für eine Zielgröße μ = (1/N')Σ_{U'} y_i mit y_i ∈ [a,b], prädiktablen Vorhersagen f^{(t)} und λ_t ∈ [0,1]:

$$Z_t = \lambda_t \bar F^{(t)} + \frac{y_{i_t} - \lambda_t f^{(t)}_{i_t}}{N' q_t(i_t)},\qquad \bar F^{(t)}=\tfrac1{N'}\textstyle\sum_{j\in U'} f^{(t)}_j .$$

Dann gilt E[Z_t | F_{t−1}] = μ. Die obere Schranke z_t^max = λ_t F̄ + max_i (b − λ_t f_i)/(N' q_t(i)) ist prädiktabel.

**Satz 1 (anytime-valid, designbasiert).** Für m ∈ ℝ sei K_t(m) = Π_{s≤t}(1 + β_s(m)(m − Z_s)) mit prädiktablen
β_s(m) ∈ [0, c/(z_s^max − m)], c<1. Unter H(m): μ ≥ m ist K_t(m) ein nichtnegatives Supermartingal. Mit
U_t := sup{m : K_s(m) < 1/δ ∀ s ≤ t} gilt P(∃t: μ > U_t) ≤ δ, also auch P(μ > U_τ) ≤ δ für jede Stoppzeit τ.

*Beweis:* E[1 + β(m−Z_t)|F_{t−1}] = 1 + β(m−μ) ≤ 1, Nichtnegativität aus der Wahl von β, dann Ville. □
Eine LCB entsteht symmetrisch.

*Konsequenzen:*

* (i) λ_t, f^{(t)} und q_t dürfen aus allen bisherigen Audits gelernt werden: PPI-Power-Tuning, isotone
  Nachkalibrierung der Selbsteinschätzung auf U und Colander-artiges Lernen der Rangfolge. Die Gültigkeit bleibt
  erhalten. Das **löst die λ-Unterdeckung**.
* (ii) β_s per WSR-Plug-in β_s = min{√(2 log(1/δ)/(σ̂²_{s−1}·t*)), c/(z^max_s − m)}, wie im PAC-Labels-Code.
  Robuster bei unbekannter Stoppzeit ist eine Mischung über ein β-Gitter (Mittel der Kapitalprozesse ist wieder ein
  Supermartingal).

**Verhältnis-Metriken: Fieller-Inversion.** Für Recall setze g_i(r) = M_i − (1−r)D_i ∈ [−(1−r)D_max, r·D_max]. Es gilt

$$\rho^0 \le r\iff \tfrac1{N'}\textstyle\sum_{U'} g_i(r)\ \ge\ c(r):=(1-r)D_R/N'.$$

Auf einem Gitter r_1 < … < r_J wird H(r_j) mit Satz 1 (Terme Z_t(r_j), Niveau m = c(r_j)) getestet.

**Prop. 4.** Mit LCB_t := max{r_j : H(r_l) bis t verworfen für alle l ≤ j} gilt P(∃t: ρ⁰ < LCB_t) ≤ δ.

*Beweis:* Sei r_{j*} der kleinste Gitterpunkt ≥ ρ⁰. Dann ist H(r_{j*}) wahr und wird mit W. ≤ δ je verworfen. Sonst
gilt LCB_t ≤ r_{j*−1} ≤ ρ⁰. Kein Multiplizitätsaufschlag und keine Monotonie nötig. □

Dasselbe gilt für Präzision (g = F − (1−r)P) und gepoolten Dice (g = r(FP+FN) − 2(1−r)TP). Fieller nutzt die
Korrelation von Zähler und Nenner, ohne Delta-Methode.

**Binäre Bildmetriken (Primärmetrik ζ).** Für y_i = b_i ∈ {0,1}:

* bei uniformem q und fester Stichprobe exakt hypergeometrisch: UCB = max{K : P_hyp(X ≤ k | N', K, n) > δ}/N';
* sonst Satz 1 mit Bereich [0,1]/(N'q).

In (S2) sind beide gültig und bei n=100 schmal (0,045–0,16).

**Approximative Variante.** Für Pixelmetriken und große D_max: asymptotische CS bzw. CLT-Fieller (IPW). Stets als
„approximativ“ kennzeichnen; (S1) zeigt 0,83–0,95 statt 0,95.

**Komponenten-Abhängigkeit** ist erledigt: Einheit = Bild, Terme = Bildsummen. Mehrklassen: pro Klasse eigene
y_i^{(c)}, Bonferroni über Klassen und Metriken (δ/J).

**Rückkopplungsregel.** Audit-GT darf **nicht** in die Pseudo-Labels einfließen, die zertifiziert werden: kein
Nachtrainieren mit Audit-Bildern vor Abschluss. Audit-Bilder dürfen ins Training des *Produktivmodells*, ausgenommen
ein reservierter Test-Anteil (§2.5).

### 4.6 Stufe D – Schwellen und Akzeptanz aus dem Audit (G-P, „designbasiertes LTT“)

**Satz 2 (PAC-Akzeptanz für Segmentierung).** Sei u_1 < … < u_J ein Gitter von Review-Schwellen auf r_i. Setze

$$A(u)=\sum_{U'}\big(M_i\,1\{r_i<u\}-\varepsilon D_i\big)-\varepsilon D_R ,$$

nichtfallend in u. Teste H(u_j): A(u_j) > 0 mit Satz 1 (Terme y_i(u) = M_i 1{r_i<u} − εD_i, Niveau εD_R/N'). Setze
û = max{u_j : H(u_l) verworfen ∀ l ≤ j} und reviewe danach alle i ∈ U' mit r_i ≥ û. Dann gilt
P(ρ_final ≥ 1−ε) ≥ 1−δ.

*Beweis:* Sei u* der kleinste Gitterpunkt mit A(u*) > 0, also H(u*) wahr. Ist ρ_final < 1−ε, so ist A(û) > 0
(Lemma 0: Audit-Korrekturen helfen nur), also û ≥ u*, und H(u*) wurde verworfen: W. ≤ δ. □

Das ist PAC-Labels mit Verhältnis-Verlust, Bild-Clustern und Review-Rest.

* **Pixelschwelle statt Review-Schwelle:** Gleiches Argument mit A(t) = Σ(M_i(t) − εD_i). Da für Audit-Bilder die
  Wahrscheinlichkeitskarten vorliegen, ist M_i(t) für alle t berechenbar. Fixed-Sequence von kleinem t (sicher)
  aufwärts. Das ersetzt die F1-Gridsuche auf L durch eine *zertifizierte* Schwellenwahl auf dem Pool.
* **FP-Seite:** zweite Familie mit φ oder Präzision, δ = δ₁ + δ₂, oder Pareto Testing.

### 4.7 Stoppregeln (alle anytime-valid)

1. **Erfolg:** Stopp, sobald alle registrierten Ziele zertifiziert sind, z. B. UCB_t(ζ⁰) ≤ ζ* und LCB_t(ρ⁰) ≥ ρ*
   (bzw. Satz 2 liefert û mit tragbaren Review-Kosten).
2. **Futility:** Stopp, wenn die *obere* Schranke das Ziel verfehlt, etwa UCB_t(ρ⁰) < ρ*. Zweiseitige CS mit δ/2 je
   Seite. Dann Review ausweiten bzw. Modell verbessern.
3. **Budget:** Stopp bei Budgetende und aktuelle CS berichten; sie ist gültig.
4. **Vorab-Machbarkeit** (Idee N5): Vor Beginn die worst-case nötige Auditgröße berechnen. Liegt sie über dem Budget,
   nur ζ zertifizieren und ρ approximativ berichten.

### 4.8 Befunde zur aktuellen Implementierung (falsch bzw. suboptimal)

1. **OOF→Ensemble-Transfer** (`pipeline.py`, `masks_U`/`safe_U` aus `mean_U`): Es gibt keine Garantie. Wegen
   max-of-mean ≤ mean-of-max ist die Abweichung bei dünnen Defekten antikonservativ (S3). Fix: A2/A3 bzw. Satz 2.
2. **In-Sample-Vorhersagen auf U:** Runde ≥1 trainiert auf Pseudo-Labels *aller* U-Bilder; übersehene Defekte sind
   dort Negativ-Label. Runde 0 trainiert auf 70 % von U als Hintergründe mit Negativ-Label. Die U-Scores genau der
   kritischen Defekte werden dadurch gedrückt, während die L-OOF-Scores ehrlich bleiben. Fix: N2.
3. **Komponenten gepoolt** (`defect_scores` → `conformal_threshold`): Defekte im selben Bild sind nicht austauschbar.
   Fix: bildweise CRC (A1).
4. **`certified_recall` bei `t_seed`:** t_seed ist per F1-Gridsuche auf denselben OOF-Daten gewählt. Damit sind weder
   1−(m+1)/(n+1) noch die CP-Schranke gültig; die Formel gilt nur für eine vorab feste Rangregel. Fix: LTT
   (Bonferroni über die 7 Seeds bzw. Fixed-Sequence) oder nicht als „certified“ ausgeben. Die Log-Zeile
   „recall guarantee at t_seed“ ist irreführend.
5. **Ehrlichkeit bei kleinem n:** Zusätzlich die bedingte Abdeckungsverteilung Beta(n+1−ℓ, ℓ) melden, z. B.
   „P(Recall<0,9) ≈ 0,31“.
6. **`ppi.estimate_*`:** t-Quantil + Delta-Methode unterdecken bei seltenen Defekten (S1/S2). Bei 0 Defekten im Audit
   ist audit-only undefiniert; PPI-Varianzen können kollabieren. Fix: Satz 1 + Fieller (Prop. 4), für ζ exakt
   hypergeometrisch. Die heutigen Intervalle höchstens als „approximativ“ kennzeichnen.
7. **λ-Tuning in-sample:** Die Formel Cov/Var ist für die endliche Population richtig (kein (1+n/N), anders als
   PPI++). Das Tuning auf derselben Stichprobe ist das Problem. Fix: λ prädiktabel (Satz 1) oder aus L-OOF bzw. der
   anderen Audit-Hälfte (Cross-Fit).
8. **Uniformes Audit fester Größe** (`cli.py`, `--audit-size 30`): ineffizient für Recall bei seltenen Defekten und
   ohne Stoppregel. Fix: §4.4/§4.5.
9. **Review und Audit getrennt, Zielgröße unkorrigierter Pool:** Die eigentliche Qualität des *finalen* Pools wird
   nicht berichtet. Fix: Lemma 0 + Identität Σ_{U'∖S} y = Σ_{U'} y − Σ_S y.
10. **`no_candidate_rate`** aus ~11 L-Defekten ist extrem verrauscht und skaliert `exp_defects`. Das kostet nur
    Effizienz (PPI korrigiert). Prädiktabel auf Audit-Daten nachkalibrieren.

### 4.9 Pseudocode

```python
# ---------- Stufe 0: Registrierung (vor jedem menschlichen Blick) ----------
freeze(pseudo_labels Ŷ, prob_maps, features f, risk r, sigma σ)            # hash + seed protokollieren
targets = {"zeta": ζ_star, "recall": ρ_star}; delta_split = {"zeta": δ/2, "recall": δ/2}
D_max = preregistered_cap(); gamma = 0.3                                    # Uniform-Floor
# ---------- Stufe A: modellbasiert (optional, als annahmebasiert markiert) ----------
t_crc = image_level_crc(oof_scores_by_image, alpha, D_max)                  # A1, ggf. 0 = "keine Garantie"
safe_masks = hysteresis(max_over_members(p_U), t_crc, ...)                  # A2 Union-Regel
# ---------- Stufe B: Measure-and-Repair ----------
m, s = f["exp_missed"], sigma_ratio(f, r=ρ_star)                            # §4.4
kappa = choose_kappa(m, s, budget, target_halfwidth)
pi = measure_and_repair(m, s, kappa, budget)                                # Prop. 3
R = {i: pi[i] == 1}; human_correct(R); D_R = sum(D[i] for i in R)
Up = U - R; q = (1-gamma) * normalize(pi[Up]) + gamma / len(Up)
# ---------- Stufe C: sequentielles Audit ----------
procs = {name: BettingCS(delta_split[name], grid=...) for name in targets}  # Satz 1 / Prop. 4
seen, t = {}, 0
while True:
    t += 1
    i = sample(Up, q)                                    # mit Zurücklegen, q prädiktabel
    if i not in seen: seen[i] = human_correct_and_count(i)   # liefert M_i, D_i, b_i, F_i, ...
    y = seen[i]
    lam_t, f_t = fit_predictable(seen_before_t)          # PPI-λ, Nachkalibrierung nur aus Vergangenheit
    procs["zeta"].update(Z(y.b, f_t.b, q[i], lam_t))
    procs["recall"].update_fieller(lambda r: Z(y.M - (1-r)*y.D, f_t.M - (1-r)*f_t.D, q[i], lam_t),
                                   level=lambda r: (1-r) * D_R / len(Up))
    q = update_q_predictably(q, seen)                    # optional, Floor beibehalten
    if procs["zeta"].ucb() <= ζ_star and procs["recall"].lcb() >= ρ_star: break      # Erfolg
    if procs["recall"].ucb() < ρ_star or len(seen) >= budget: break                   # Futility/Budget
# ---------- Stufe D (optional): PAC-Akzeptanz ----------
u_hat = fixed_sequence(grid_u, test=lambda u: procs_A[u].rejected())        # Satz 2
human_correct({i in Up if r[i] >= u_hat} - seen)
report(LCB/UCB, u_hat, n_distinct=len(seen), cost=len(R)+len(seen)+n_review_D,
       assumptions=["design", "human=gold", f"D<= {D_max}", "no feedback"])
```

### 4.10 Simulationsdesign zur Validierung

* **Pool-Generator:**
  - N ∈ {500, 2000, 10000}, Prävalenz p ∈ {0,01; 0,03; 0,1; 0,3}, D_i ~ 1+Poisson(λ_D)
  - Sichtbarkeit pro Defekt; Anteil unsichtbarer Misses ν ∈ {0; 0,3; 0,6}
  - Selbsteinschätzung mit Kalibrierfehler (Bias, Rauschen, falsches ν̂)
  - Risiko-AUC für „Bild mit Miss“ ∈ {0,5; 0,7; 0,9}
  - adversariales Szenario: Misses bevorzugt in Bildern mit kleinstem q
* **Semi-synthetisch:** echte Läufe (MT, BSData, KolektorSDD2) mit versteckter GT; `evaluate.audit_simulation`
  erweitern.
* **Verfahren:**
  - heutige (`estimate_ratio`/`estimate_mean` mit λ=1 und getunt; audit-only)
  - Satz 1 mit uniformem bzw. PPS-q (γ ∈ {0,1; 0,3; 1})
  - Fieller-Betting, exakt hypergeometrisch, IPW-CLT, AsympCS
  - Measure-and-Repair vs. „Top-k-Review + separates uniformes Audit“
  - Satz 2; Schwellenregeln (Ensemble-Mittel, Union, voller Cross-Fit, Audit-kalibriert)
* **Optional Stopping:** Stopp, sobald LCB ≥ Ziel. Hier sollen fixe-n-Intervalle sichtbar unterdecken, Satz 1 nicht.
* **Metriken:**
  - Abdeckung zur Stoppzeit (Soll 1−δ; MC-SE bei 2000 Wdh. ≈ 0,5 pp)
  - Rate falscher Zertifikate
  - mittlere LCB/UCB-Breite
  - menschliche Kosten (distinkte Bilder) bis zur Zertifizierung
  - finaler Recall/ζ des korrigierten Pools
  - Anteil „Futility“-Stopps bei verfehlten Zielen
* **Akzeptanzkriterien:**
  - designbasierte Verfahren: Abdeckung ≥ 1−δ − 2 MC-SE in *allen* Szenarien inkl. adversarial
  - approximative Verfahren: Abweichung dokumentieren
  - Effizienzgewinn ggü. uniform berichten

---

## 5. Neue Ideen (über die Literatur hinaus)

**N1 – Measure-and-Repair-Design.** Review und Audit sind in der Literatur getrennte Budgets: entweder
Active-Inference-Messung *oder* Review des Unsichersten (PAC-Labels kombiniert beides nur sequentiell). Prop. 3 gibt
eine geschlossene Form der Inklusionswahrscheinlichkeiten, die Reparaturwert (m_i) und Messwert (σ_i²) gegeneinander
abwägt. Das Take-all-Stratum entsteht endogen; κ ist der „Wechselkurs“ zwischen einem verbliebenen Fehler und einer
Varianzeinheit. Mit Lemma 0 und der Identität „finale Fehler = Populationssumme − korrigierte Fehler“ wird die
Qualität des **korrigierten** Pools unverzerrt zertifiziert, ohne dass Reviewbilder das Audit verzerren.
*Begründung:* Menschliche Zeit ist das knappe Gut; jede Prüfung soll doppelt zählen.

**N2 – Doppelter Cross-Fit über L ∪ U.** Folds über L *und* U. Member k nutzt L_k ∪ U_k weder als Label noch als
Hintergrund noch als Pseudo-Label und sagt beide vorher. Dann sind L-OOF- und U-Vorhersagen vom identischen
Mechanismus (Einzelmember, out-of-sample). Damit:

* (i) L-kalibrierte Schwellen gelten unter L~U-Austauschbarkeit exakt pro Fold (CV+-gepoolt);
* (ii) Audit-Bilder aus U_k kalibrieren Member k direkt;
* (iii) die Bestätigungsverzerrung des Self-Trainings auf den *eigenen* U-Bildern entfällt.

Kosten: kein Ensemble-Mittel auf U, sondern 1–2 Member pro Fold. Ensemble-Diversität bleibt über die Folds erhalten.
*Begründung:* Befunde 1 und 2 in §4.8 sind die größten Garantie-Lecks; N2 schließt beide konstruktiv.

**N3 – Designbasiertes LTT: Schwellenwahl auf dem Audit.** LTT/CRC wählen Parameter auf einem austauschbaren
Kalibrierset; PAC-Labels zertifiziert eine *Review*-Schwelle. Wir wählen Pixelschwelle (t_seed) **und**
Review-Schwelle per Fixed-Sequence-Testing auf den designbasierten e-Prozessen des Audits (Satz 2). Das Zertifikat
gilt für den konkreten Pool, auch wenn L verzerrt ausgewählt wurde. Die F1-Gridsuche auf ~10 L-Bildern wird dann nur
noch als Startpunkt gebraucht.

**N4 – Prädiktabel getuntes Fieller-Betting.** Kombination aus (a) PPI-Kontrollvariate mit *prädiktablem* λ_t und
prädiktabel nachkalibrierter Selbsteinschätzung f^{(t)}, (b) IPW mit adaptivem q_t und (c) Fieller-Inversion für
Verhältnisse auf dem Wettgitter (Prop. 4, ohne Monotonie- oder Multiplizitätsaufschlag). Das löst das beobachtete
Problem „getuntes λ unterdeckt bei n<50“ prinzipiell: Tuning auf Vergangenheitsdaten verletzt die
Martingal-Eigenschaft nicht.

**N5 – Heuhaufen-Schranke als Vorab-Machbarkeitscheck.** Wenn Misses in Bildern ohne jedes Signal liegen können,
muss jedes Verfahren, das ρ ≥ 1−ε zertifiziert, im saubersten Stratum mit (effektiv) uniformer Rate prüfen. Nötig
sind n_c ≈ N_c·ln(1/δ)/(ε·D_tot) Audits (Binomial-Näherung; exakt hypergeometrisch etwas weniger, Zahlen in S4). Das Tool sollte das *vor* dem Audit
ausrechnen und ehrlich sagen, ob Defekt-Recall-Zertifizierung mit dem Budget erreichbar ist. Hebel sind
(i) ein starkes, unabhängiges Signal für unsichtbare Misses, also das Normalitätsmodell als Proposal-Verteilung im
Floor-Anteil (Gültigkeit unberührt, Effizienz steigt mit seiner Trennschärfe), und (ii) kleinere Einheiten (N6).

**N6 – Kacheln als Stichprobeneinheit.** Designbasierte Inferenz braucht keine Bild-Austauschbarkeit. Die
Stichprobeneinheit darf eine Kachel sein, etwa 256×256 mit vorab festgelegter Zuordnungsregel „Komponente gehört zur
Kachel ihres Maximum-Pixels“. Vorteile: (i) D_max pro Einheit sinkt (≈1–2), wodurch die Betting-Schranken für Recall
um Größenordnungen schärfer werden (S1-Problem); (ii) menschliche Prüfzeit pro Einheit sinkt; (iii) mehr Einheiten
bei gleichem Zeitbudget, kleinerer Design-Effekt der Clusterung. Risiko: Kachelränder; Gegenmittel ist eine
Überlappungszone mit eindeutiger Zuordnung.

**N7 – Konforme Ignore-Ringe (morphologisches CRC).** Die Ringbreite w (heute fix 2 px) wird auf Audit-Bildern per
CRC kalibriert. Verlust = Anteil der GT-Randpixel außerhalb dilate(Ŷ, w) (Mossina & Friedrich, Dilatation) bzw. für
den Positiv-Kern erode(Ŷ, w′) ⊆ GT (Erosions-Dual). Garantie: Die erwartete Randabweichung liegt im Ignore-Band. Das
Training sieht dann weniger falsche Negativ-/Positiv-Pixel am Rand, und die Breite passt sich pro Aufgabe an
(task-agnostisch).

---

## 6. Grenzen und offene Punkte

* „Mensch = Goldstandard“ ist eine Annahme. Ein Doppel-Labeling einer kleinen Zufallsauswahl schätzt die
  Reviewer-Fehlerrate und kann als zusätzlicher (designbasierter) Term eingehen.
* D_max muss vorab registriert werden. Wird es im Audit überschritten, ist die nichtasymptotische Garantie für diese
  Metrik formal verletzt; N6 entschärft das.
* Satz 1 mit Zurücklegen verschenkt etwas Effizienz gegenüber Ziehen ohne Zurücklegen. Die gewichtete
  WOR-Martingal-Variante (ALPHA-artig) ist eine mögliche Verbesserung.
* Modellbasierte Garantien (G-M) für *künftige* Produktionsbilder: Prognosen auf U sind durch Self-Training
  in-sample. Für Produktion braucht man frische, zufällige Kalibrierbilder.
* Die Konstanten von CV+ und Details von TBAL/SGR/Xu et al. sind hier nicht am Volltext verifiziert.

---

## 7. Referenzen

Verifiziert = Existenz und Kernaussage in dieser Sitzung per Websuche bzw. offiziellem Code geprüft; sonst
**(unverified)**.

**Auto-Labeling / Selektion**
- Vishwakarma, H., Lin, H., Sala, F., Vinayak, R. K. *Promises and Pitfalls of Threshold-based Auto-labeling.* NeurIPS 2023. arXiv:2211.12620.
- Vishwakarma, H. et al. *Pearls from Pebbles: Improved Confidence Functions for Auto-labeling.* NeurIPS 2024. arXiv:2404.16188.
- Candès, E., Ilyas, A., Zrnic, T. *Probably Approximately Correct Labels.* ICML 2026 (PMLR 306). arXiv:2506.10908. Code: github.com/tijana-zrnic/pac-labels.
- (Autoren unverified) *Model-agnostic Selective Labeling with Provable Statistical Guarantees* (v1: *Selective labeling with false discovery rate control*; „Conformal Labeling“). arXiv:2510.14581, 2025.
- Jin, Y., Candès, E. *Selection by prediction with conformal p-values.* JMLR 2023. arXiv:2210.01408 (ID unverified; Framework per Suche bestätigt).
- (Autoren unverified) *Selection from Hierarchical Data with Conformal e-values.* arXiv:2501.02514.
- Wang, X. et al. *Conformalized Multiple Testing after Data-dependent Selection.* NeurIPS 2024.
- (Autoren unverified) *ACS: An interactive framework for conformal selection.* arXiv:2507.15825.
- Geifman, Y., El-Yaniv, R. *Selective classification for deep neural networks.* NeurIPS 2017. arXiv:1705.08500 (unverified).

**Risikokontrolle / Conformal**
- Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., Schuster, T. *Conformal Risk Control.* ICLR 2024. arXiv:2208.02814. Code: github.com/aangelopoulos/conformal-risk.
- Angelopoulos, A. N., Bates, S., Candès, E., Jordan, M. I., Lei, L. *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control.* Ann. Appl. Stat. 2025. arXiv:2110.01052.
- Bates, S., Angelopoulos, A., Lei, L., Malik, J., Jordan, M. *Distribution-free, risk-controlling prediction sets.* JACM 2021. arXiv:2101.02703 (unverified).
- Laufer-Goldshtein, B., Fisch, A., Barzilay, R., Jaakkola, T. *Efficiently controlling multiple risks with Pareto testing.* ICLR 2023. arXiv:2210.07913 (unverified).
- Mossina, L., Dalmau, J., Andéol, L. *Conformal Semantic Image Segmentation: Post-hoc Quantification of Predictive Uncertainty.* CVPR Workshops 2024. arXiv:2405.05145.
- Mossina, L., Friedrich, C. *Conformal Prediction for Image Segmentation Using Morphological Prediction Sets.* MICCAI 2025. arXiv:2503.05618.
- Mossina, L., Friedrich, C. *Controlling False Positives in Image Segmentation via Conformal Prediction.* arXiv:2511.15406, 2025.
- Brunekreef, J., Marcus, E., Sheombarsing, R., Sonke, J.-J., Teuwen, J. *Kandinsky Conformal Prediction: Efficient Calibration of Image Segmentation Algorithms.* CVPR 2024. arXiv:2311.11837.
- Bairaktari, K., Wu, J., Wu, Z. S. *Kandinsky Conformal Prediction: Beyond Class- and Covariate-Conditional Coverage.* ICML 2025. arXiv:2502.17264.
- Lee, Y., Barber, R. F., Willett, R. *Distribution-free inference with hierarchical data.* arXiv:2306.06342.
- Dunn, R., Wasserman, L., Ramdas, A. *Distribution-free prediction sets for two-layer hierarchical models.* JASA 2023 (unverified).
- Cohen, K. M., Park, S., Simeone, O., Shamai, S. *Cross-Validation Conformal Risk Control.* arXiv:2401.11974, 2024.
- Barber, R. F., Candès, E., Ramdas, A., Tibshirani, R. *Predictive inference with the jackknife+.* Ann. Stat. 2021. arXiv:1905.02928 (unverified).
- Kim, B., Xu, C., Barber, R. F. *Predictive inference is free with the jackknife+-after-bootstrap.* NeurIPS 2020. arXiv:2002.09025 (unverified).
- Tibshirani, R., Barber, R. F., Candès, E., Ramdas, A. *Conformal prediction under covariate shift.* NeurIPS 2019. arXiv:1904.06019 (unverified).
- Ding, T., Angelopoulos, A., Bates, S., Jordan, M., Tibshirani, R. *Class-conditional conformal prediction with many classes.* NeurIPS 2023. arXiv:2306.09335 (unverified).
- Vovk, V. *Conditional validity of inductive conformal predictors.* ACML 2012. arXiv:1209.2673 (unverified).
- (Autoren unverified) *Coverage Guarantees for Pseudo-Calibrated Conformal Prediction under Distribution Shift.* arXiv:2602.14913, 2026 (nur Existenz verifiziert).
- (Autoren unverified) *Leave-One-Out-, Bootstrap- and Cross-Conformal Anomaly Detectors.* arXiv:2402.16388.

**Messung / PPI / aktive Inferenz**
- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., Zrnic, T. *Prediction-powered inference.* Science 2023. arXiv:2301.09633.
- Angelopoulos, A. N., Duchi, J. C., Zrnic, T. *PPI++: Efficient Prediction-Powered Inference.* arXiv:2311.01453. Code: github.com/aangelopoulos/ppi_py.
- Zrnic, T., Candès, E. *Cross-prediction-powered inference.* PNAS 2024. arXiv:2309.16598.
- Zrnic, T., Candès, E. *Active Statistical Inference.* ICML 2024. arXiv:2403.03208. Code: github.com/tijana-zrnic/active-inference.
- (Autoren unverified) *Prediction-Powered Bootstrap.* arXiv:2405.18379.
- (Autoren unverified) *Prediction-Powered Inference with Imputed Covariates and Nonuniform Sampling.* arXiv:2501.18577.
- Brawand, N. P. et al. (unverified) *Active Multiple-Prediction-Powered Inference.* arXiv:2605.08429, 2026.
- Fisch, A. et al. *Stratified Prediction-Powered Inference for Hybrid Language Model Evaluation.* NeurIPS 2024. arXiv:2406.04291 (unverified).
- Gligorić, K., Zrnic, T., Lee, C., Candès, E., Jurafsky, D. *Can Unconfident LLM Annotations Be Used for Confident Conclusions?* arXiv:2408.15204 (unverified).
- Einbinder, B.-S., Ringel, L., Romano, Y. *Semi-Supervised Risk Control via Prediction-Powered Inference.* arXiv:2412.11174 (unverified).
- Kossen, J., Farquhar, S., Gal, Y., Rainforth, T. *Active Testing: Sample-Efficient Model Evaluation.* ICML 2021. arXiv:2103.05331 (unverified).
- Kossen, J. et al. *Active Surrogate Estimators.* NeurIPS 2022. arXiv:2202.06881 (unverified).
- Farquhar, S., Gal, Y., Rainforth, T. *On Statistical Bias in Active Learning.* ICLR 2021. arXiv:2101.11665 (unverified).
- Egami, N., Hinck, M., Stewart, B., Wei, H. *Using imperfect surrogates for downstream inference (DSL).* NeurIPS 2023. arXiv:2306.04746 (unverified).
- Särndal, C.-E., Swensson, B., Wretman, J. *Model Assisted Survey Sampling.* Springer 1992 (unverified).

**Anytime-valid / Audits**
- Waudby-Smith, I., Ramdas, A. *Estimating means of bounded random variables by betting.* JRSS-B 2024. arXiv:2010.09686 (Paper unverified; WSR-Verfahren im PAC-Labels-Code verifiziert).
- Waudby-Smith, I., Ramdas, A. *Confidence sequences for sampling without replacement.* NeurIPS 2020. arXiv:2006.04347 (unverified).
- Howard, S., Ramdas, A., McAuliffe, J., Sekhon, J. *Time-uniform, nonparametric, nonasymptotic confidence sequences.* Ann. Stat. 2021. arXiv:1810.08240 (unverified).
- Waudby-Smith, I., Arbour, D., Sinha, R., Kennedy, E., Ramdas, A. *Time-uniform central limit theory and asymptotic confidence sequences.* arXiv:2103.06476 (unverified).
- Stark, P. B. *ALPHA: Audit that learns from previously hand-audited ballots.* Ann. Appl. Stat. 2023. arXiv:2201.02707 (unverified); *SHANGRLA*, arXiv:1911.10035 (unverified).
- Xu, Z., Karampatziakis, N., Mineiro, P. *Active, anytime-valid risk controlling prediction sets.* NeurIPS 2024. arXiv:2406.10490 (unverified).
- Chugg, B. et al. *Auditing Fairness by Betting.* NeurIPS 2023. arXiv:2305.17570 (unverified).

**Downstream / verrauschte Labels**
- Natarajan, N., Dhillon, I., Ravikumar, P., Tewari, A. *Learning with Noisy Labels.* NeurIPS 2013 (unverified).
- Wei, C., Shen, K., Chen, Y., Ma, T. *Theoretical Analysis of Self-Training with Deep Networks on Unlabeled Data.* ICLR 2021. arXiv:2010.03622 (unverified).
