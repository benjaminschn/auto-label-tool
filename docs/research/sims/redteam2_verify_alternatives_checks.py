"""Adversarial verification of the 'alternatives lens' findings A1-A5 (CertAL-lite vs CertAL-full).

All checks are small, exact where possible, CPU < 3 min. Nothing here changes other files.

  A1  review stratum R inspected verify-first (sensitivity s < 1) and excluded from the audit frame U':
      (a) the worked example (N=2000, 180 bad, top-700 review, n=400 gold audit of U'),
      (b) the low-prevalence pools of lite_zeta (k=300), false-certificate rate when n is power-planned,
      (c) fix (i): one SRSWOR from the WHOLE pool after review (R included) -> validity and cost.
  A2  zero-failure lower bound for any design that certifies on zero findings (Poisson inclusion).
  A3  explained-variance ceiling of a hidden-miss predictor and PPS gain from a hardness map, in
      regimes harsher than c1_value.py (more hidden misses, stronger concentration, perfect map).
  A4  (a) PPR without-replacement anytime bound: stopping-time quantiles vs fixed-n n_50 / n_90,
      (b) how fragile an exactly planned fixed n is when the residual K' is guessed wrong.
  A5  zero-discrepancy UCB of reviewer-missed bad images from a gold subsample of size n_2.

Run:  python docs/research/sims/redteam2_verify_alternatives_checks.py
"""
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
from scipy.special import gammaln  # noqa: E402
from scipy.stats import hypergeom, norm  # noqa: E402

RNG = np.random.default_rng(20261019)
DELTA = 0.05
_UCB = {}


def ucb_count(x, n, Np, delta=DELTA):
    """Exact one-sided (1-delta) UCB of the number of bad units (SRSWOR, n of Np, x found)."""
    key = (x, n, Np, delta)
    if key not in _UCB:
        Ks = np.arange(x, Np + 1)
        _UCB[key] = int(Ks[hypergeom.cdf(x, Np, Ks, n) > delta].max())
    return _UCB[key]


def power(Kp, Np, n, Kstar, delta=DELTA):
    xs = np.arange(0, min(n, Kp) + 1)
    pm = hypergeom.pmf(xs, Np, Kp, n)
    ok = np.array([ucb_count(int(x), n, Np, delta) - x <= Kstar for x in xs])
    return float((pm * ok).sum())


def n_plan(Kp, Np, Kstar, pw=0.9, grid=None):
    grid = grid if grid is not None else list(range(20, 400, 5)) + list(range(400, 1801, 10))
    for n in grid:
        if n <= Np and power(Kp, Np, n, Kstar) >= pw:
            return n
    return None


# ------------------------------------------------------------------------------------------- A1
def a1():
    print("A1 (a) worked example: N=2000, R=700 reviewed verify-first, 150 bad in R, 30 bad in U' (N'=1300)")
    N, NR, KR, Np, Kp, n, Kstar = 2000, 700, 150, 1300, 30, 400, 40
    print(f"    exact UCB(K') for x=9, n=400, N'=1300: {ucb_count(9, n, Np)} -> UCB - found = {ucb_count(9, n, Np) - 9}"
          f" (K* = {Kstar})")
    for s in (1.0, 0.9, 0.7):
        reps, cert, false = 4000, 0, 0
        for _ in range(reps):
            r_res = RNG.binomial(KR, 1 - s)                       # left unfixed in R
            x = hypergeom.rvs(Np, Kp, n, random_state=RNG)        # gold audit of U'
            if ucb_count(int(x), n, Np) - x <= Kstar:
                cert += 1
                false += (r_res + Kp - x) > Kstar
        print(f"    review sensitivity s={s}: certifies 'zeta<=0.02' in {cert / reps:.3f}; false certificates "
              f"{false / reps:.3f} of all runs")

    print("A1 (b) low-prevalence pools as in lite_zeta (N=2000, 10% defective, p_bad=0.25, v=0.3, mu_bad=2.5)")
    rows = []
    for _ in range(30):
        defect = RNG.random(N) < 0.10
        bad = defect & (RNG.random(N) < 0.25)
        inv = bad & (RNG.random(N) < 0.3)
        mu = np.where(bad & ~inv, 2.5, np.where(defect & ~bad, 1.5, 0.0))
        r = mu + RNG.normal(size=N)
        order = np.argsort(-r)
        rows.append((bad[order[:300]].sum(), bad[order[300:]].sum()))
    rows = np.array(rows)
    print(f"    k=300: bad in R mean {rows[:, 0].mean():.1f} (range {rows[:, 0].min()}-{rows[:, 0].max()}), "
          f"bad in U' mean {rows[:, 1].mean():.1f}")
    KR, Kp, Np = int(round(rows[:, 0].mean())), int(round(rows[:, 1].mean())), 1700
    n90 = n_plan(Kp, Np, Kstar)
    for s in (1.0, 0.7, 0.5):
        reps, cert, false = 4000, 0, 0
        for _ in range(reps):
            r_res = RNG.binomial(KR, 1 - s)
            x = hypergeom.rvs(Np, Kp, n90, random_state=RNG)
            if ucb_count(int(x), n90, Np) - x <= Kstar:
                cert += 1
                false += (r_res + Kp - x) > Kstar
        print(f"    K_R={KR}, K'={Kp}, n planned (90% power, R ignored) = {n90}; s={s}: certifies {cert / reps:.3f}, "
              f"false {false / reps:.3f}; E[residual in R] = {KR * (1 - s):.1f} = {KR * (1 - s) / Kstar:.0%} of K*")

    print("A1 (c) fix (i): one SRSWOR from the WHOLE pool after review (R included), gold audit")
    for s in (1.0, 0.7):
        Ktot = int(round(KR * (1 - s))) + Kp
        nf = n_plan(Ktot, N, Kstar)
        worst = 0.0                                   # P(certify AND K - found > K*), worst case over K
        for K in range(Kstar + 1, Kstar + 80):
            xs = np.arange(0, min(nf, K) + 1)
            pm = hypergeom.pmf(xs, N, K, nf)
            bad_cert = np.array([(ucb_count(int(x), nf, N) - x <= Kstar) and (K - x > Kstar) for x in xs])
            worst = max(worst, float((pm * bad_cert).sum()))
        print(f"    s={s}: residual after review {Ktot} in N=2000 -> n_90 = {nf} (vs. U'-only plan {n90} "
              f"that assumes R clean); worst-case false-certificate prob. with n={nf}: {worst:.3f} (<= 0.05)")


# ------------------------------------------------------------------------------------------- A2
def a2():
    Np, Kstar = 1850, 40
    pi0 = 1 - DELTA ** (1 / (Kstar + 1))
    n0 = next(n for n in range(1, Np) if ucb_count(0, n, Np) <= Kstar)
    print(f"A2 zero-failure: Poisson bound N'(1-delta^(1/(K*+1))) = {Np * pi0:.1f}; with the K* free units "
          f"(N'-K*)pi0 = {(Np - Kstar) * pi0:.1f}; exact SRSWOR smallest n with UCB(0) <= K*: {n0}")
    # any two-level design: K*+1 units at pi_lo < pi0 make the zero-hit probability > delta
    for pl in (0.5 * pi0, 0.9 * pi0):
        print(f"    a unit set of size K*+1 at pi={pl:.4f} (< pi0={pi0:.4f}) is missed entirely with prob "
              f"{(1 - pl) ** (Kstar + 1):.3f} > delta -> such a design cannot certify on zero findings")


# ------------------------------------------------------------------------------------------- A3
def a3():
    print("A3 ceiling R^2 = Var(lam)/(E lam + Var lam) for Poisson hidden misses (N=2000), and PPS gain")
    N = 2000
    for H in (15, 60, 150):
        for s in (0.01, 0.05, 0.2):
            conc = np.zeros(N, bool)
            conc[: int(s * N)] = True
            h = np.where(conc, 1.0, 0.02)
            lam = H * h / h.sum()
            r2 = lam.var() / (lam.mean() + lam.var())
            # Monte-Carlo: candidate misses as in c1_value.py, perfect hardness map (fidelity 1, oracle N_hat)
            cv_a, cv_aw, pps_a, pps_aw, l1 = [], [], [], [], {0.5: [], 1.0: [], 2.0: []}
            for _ in range(40):
                nc = RNG.poisson(0.15, N)
                a = np.array([RNG.beta(2, 3, k).sum() for k in nc])
                cm = np.array([(RNG.random(k) < RNG.beta(2, 3, k)).sum() for k in nc])
                y = cm + RNG.poisson(lam)
                V0 = y.var()

                def cv(f):
                    c = np.cov(y, f)[0, 1] / f.var()
                    return (y - c * f).var() / V0
                cv_a.append(cv(a))
                X = np.column_stack([np.ones(N), a, lam])
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                cv_aw.append((y - X @ beta).var() / V0)
                for c in l1:                                   # lambda = 1 predictor, N_hat = c * truth
                    l1[c].append((y - a - c * lam).var() / V0)
                T = y.sum()

                def vht(p):
                    p = p / p.sum()
                    return np.sum(y ** 2 / p) - T ** 2
                vu = vht(np.ones(N))
                va = a - a ** 2 / np.maximum(nc, 1)
                pps_a.append(vht(np.sqrt(va + a ** 2 + H / N + (H / N) ** 2)) / vu)
                pps_aw.append(vht(np.sqrt(va + a ** 2 + lam + lam ** 2 + 2 * a * lam)) / vu)
            print(f"    hidden total {H:3d}, concentrated in {s:.0%}: ceiling R^2 of hidden part {r2:.3f}; "
                  f"residual var CV(a) {np.mean(cv_a):.3f} -> GREG(a, perfect map) {np.mean(cv_aw):.3f}; "
                  f"PPS var a-only {np.mean(pps_a):.3f} -> with perfect map {np.mean(pps_aw):.3f} "
                  f"({1 - np.mean(pps_aw) / np.mean(pps_a):+.0%}); lambda=1 with N_hat x0.5/1/2: "
                  + "/".join(f"{np.mean(v):.3f}" for v in l1.values()))
    print("A3 marginal audit cost of one residual bad image (n_90 slope, N'=1850)")
    for Kstar in (40, 20):
        ks = [0, 5, 10, 15] if Kstar == 40 else [0, 3, 6, 9]
        ns = [n_plan(k, 1850, Kstar) for k in ks]
        slope = np.polyfit(ks, ns, 1)[0]
        print(f"    K*={Kstar}: n_90 at K'={ks} -> {ns}; slope {slope:.1f} audits per residual bad image")


# ------------------------------------------------------------------------------------------- A4
def a4():
    N, n_max, step, reps = 2000, 800, 5, 400
    Ks = np.arange(N + 1)
    print("A4 (a) PPR WOR anytime bound (prior Beta-Binomial(1,100), looks every 5) vs fixed n, K=10 of 2000")
    for a_, b_ in ((1.0, 100.0), (1.0, 1.0)):
        logprior = (gammaln(N + 1) - gammaln(Ks + 1) - gammaln(N - Ks + 1) + gammaln(Ks + a_)
                    + gammaln(N - Ks + b_) - gammaln(N + a_ + b_) - (gammaln(a_) + gammaln(b_) - gammaln(a_ + b_)))
        pop = np.zeros(N, int)
        pop[:10] = 1
        stops = []
        for _ in range(reps):
            cum = np.cumsum(RNG.permutation(pop)[:n_max])
            st = np.inf
            for n in range(step, n_max + 1, step):
                k = cum[n - 1]
                ok = (Ks >= k) & (N - Ks >= n - k)
                logL = np.full(N + 1, -np.inf)
                Kk = Ks[ok]
                logL[ok] = gammaln(Kk + 1) - gammaln(Kk - k + 1) + gammaln(N - Kk + 1) - gammaln(N - Kk - (n - k) + 1)
                lm = logprior + logL
                mx = lm[ok].max()
                logM = mx + np.log(np.exp(lm[ok] - mx).sum()) - logL
                if np.all(logM[41:] >= np.log(1 / DELTA)):
                    st = n
                    break
            stops.append(st)
        stops = np.array(stops)
        print(f"    prior Beta({a_:.0f},{b_:.0f}): certified by 800 {np.isfinite(stops).mean():.3f}; stopping n "
              f"q50 {np.quantile(stops, 0.5):.0f}, q90 {np.quantile(stops, 0.9):.0f}")
    fx = [next(n for n in range(20, 2001, 5) if power(10, N, n, 40) >= p) for p in (0.5, 0.9)]
    print(f"    fixed-n exact (oracle K=10): n_50 = {fx[0]}, n_90 = {fx[1]}  "
          f"[note: test 'K <= 40' without crediting, i.e. ucb_count(x) <= 40, see below]")
    # same estimand as (b)/PPR: certify K <= 40 (no crediting); recompute fixed n for that rule
    def pw_nocredit(K, n):
        xs = np.arange(0, min(n, K) + 1)
        return float((hypergeom.pmf(xs, N, K, n) * np.array([ucb_count(int(x), n, N) <= 40 for x in xs])).sum())
    fx2 = [next(n for n in range(20, 2001, 5) if pw_nocredit(10, n) >= p) for p in (0.5, 0.9)]
    print(f"    fixed-n exact, no crediting (as fixedn_vs_seq.py): n_50 = {fx2[0]}, n_90 = {fx2[1]}")
    # WOR betting (confseq-style mean adjustment), mixture of constant bets, H0: K >= 41
    betas = np.linspace(0, 1, 41)[1:]
    pop = np.zeros(N, int)
    pop[:10] = 1
    for K_true, lab in ((10, "K=10"), (41, "K=41 (boundary, any certificate false)")):
        pop = np.zeros(N, int)
        pop[:K_true] = 1
        st = []
        for _ in range(1000):
            x = RNG.permutation(pop)[:n_max]
            t = np.arange(1, n_max + 1)
            S_prev = np.concatenate([[0], np.cumsum(x)[:-1]])
            mu = (41 - S_prev) / (N - (t - 1))                       # null conditional mean at K0 = 41
            bmax = 0.99 / np.maximum(1 - mu, 1e-9)
            logk = np.cumsum(np.log1p((betas[None, :] * bmax[:, None]) * (mu[:, None] - x[:, None])), axis=0)
            mxk = logk.max(1)
            mix = mxk + np.log(np.exp(logk - mxk[:, None]).mean(1))
            hit = np.nonzero(mix >= np.log(1 / DELTA))[0]
            st.append(hit[0] + 1 if len(hit) else np.inf)
        st = np.array(st)
        print(f"    WOR betting mixture, {lab}: certifies by 800 in {np.isfinite(st).mean():.3f}; "
              f"q50 {np.quantile(st, 0.5):.0f}, q90 {np.quantile(st, 0.9):.0f}")
    # group-sequential exact hypergeometric, looks (150, 250, 350, 500, 700), delta/5 each
    looks = (150, 250, 350, 500, 700)
    for K_true in (10, 41):
        pop = np.zeros(N, int)
        pop[:K_true] = 1
        st = []
        for _ in range(2000):
            cum = np.cumsum(RNG.permutation(pop)[:looks[-1]])
            c = np.inf
            for n in looks:
                if ucb_count(int(cum[n - 1]), n, N, DELTA / len(looks)) <= 40:
                    c = n
                    break
            st.append(c)
        st = np.array(st)
        print(f"    group-sequential exact (5 looks, delta/5), K={K_true}: certifies {np.isfinite(st).mean():.3f}; "
              f"q50 {np.quantile(st, 0.5):.0f}, q90 {np.quantile(st, 0.9):.0f}")
    print("A4 (b) fixed n planned for a guessed K' (N'=1850, K*=40): power at the true K'")
    for guess in (5, 10):
        n = n_plan(guess, 1850, 40)
        print(f"    planned for K'={guess} -> n={n}: power at true K' = "
              + ", ".join(f"{t}: {power(t, 1850, n, 40):.2f}" for t in (guess, 2 * guess, 3 * guess)))


# ------------------------------------------------------------------------------------------- A5
def a5():
    N = 2000
    print("A5 zero-discrepancy UCB (delta/2 = 0.025) of reviewer-missed bad images from a gold subsample")
    for n2 in (200, 400, 600, 1000):
        u = ucb_count(0, n2, N, 0.025)
        print(f"    n_2 = {n2:4d}: UCB = {u} images (zeta {u / N:.4f}); N ln(2/delta)/n_2 = {N * np.log(40) / n2:.1f}")


if __name__ == "__main__":
    a1()
    a2()
    a3()
    a4()
    a5()
