"""Adversarial verification (math lens) of the second red-team findings F1-F7 on 04 / 06 / 00.

Only checks that the earlier scripts do not already cover:

  A (F2)  Satz 2 with a *predictable* (not post-hoc) learned review score r: Bayesian group-level
          miss-rate learner, r^(t) uses only draws < t. Budget stop and stop-at-certification.
          Compared with frozen r and with the two-phase scheme.
  B (F3)  Which synthetic-leak shapes the GT-free transport test can see: A-only logit leak,
          both-detector logit leak, proportional ("visibility") leak.
  C (F4)  Per-step drift of the per-draw bet cap of sim_recall_cert on the adversarial pool (analytic),
          and how often sim2.py's 'zmax_pred' is not an upper bound on the benign S1 pools.
  D (F5)  ppi.estimate_ratio: coverage with a good vs. a heavy-tailed bad predictor (SRS), and
          bias of the unweighted estimator under PPS.
  E (F1)  Growth-rate upper bound (Jensen) for any capped bet: draws needed with D_max = 6, with a
          'physically safe' tile bound, and with the area bound proposed in the F1 fix.
  F (F7e) Prop. 3 closed form returns pi_i = 0 for sigma_i = 0.

Run:  python3 docs/research/sims/redteam2_verify_math_checks.py      (~1-2 min, CPU)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
import redteam_capture_recapture as cr  # noqa: E402
from redteam_audit_checks import prop3_closed  # noqa: E402
from sim_recall_cert import make_pool, DMAX  # noqa: E402
from autolabel import ppi  # noqa: E402

RNG = np.random.default_rng(20261019)
DELTA = 0.05


# ------------------------------------------------------------------------------------------- A
def mixture_step(logk, x, m, betas):
    return logk + np.log1p(betas[None, :] * (m - x[:, None]))


def satz2_predictable_learner(reps=2000, n=600, u=0.4, eps=0.1, a=50.0, b=50.0):
    """U' = 2000 units. Group G = units 0..399: 100 single-defect units, all missed (miss rate .25).
    Rest: 400 single-defect units, 10 missed. D_tot = 500 -> claim 'rho_final >= 0.9' needs final
    misses <= 50. Review rule: review all units with r_i >= u. r = 0.05 outside G; inside G the
    learner uses the posterior mean of the G miss rate, prior Beta(a, b) (mean 0.5), updated ONLY
    with past audits (predictable). Single hypothesis H(u): A(u) > 0, terms y_t = M 1{r^(t) < u} - eps D."""
    N = 2000
    G = np.zeros(N, bool)
    G[:400] = True
    D = np.zeros(N)
    M = np.zeros(N)
    D[:100] = 1
    M[:100] = 1
    D[400:800] = 1
    M[400:410] = 1
    nb = 20
    m = eps                                             # x = y + eps in [0, 1], H: E x >= eps
    betas = np.linspace(0, 0.99 / (1 - m), nb + 1)[1:]
    out = {}
    for mode in ("predictable learner", "frozen r = prior", "frozen r = truth (0.25)"):
        idx = RNG.integers(0, N, (reps, n))
        seen = np.zeros((reps, N), bool)
        nG = np.zeros(reps)
        kG = np.zeros(reps)
        found_other = np.zeros(reps)
        logk = np.zeros((reps, nb))
        rej = np.zeros(reps, bool)
        stop_false = np.zeros(reps, bool)
        stop_cert = np.zeros(reps, bool)
        ar = np.arange(reps)
        for t in range(n):
            i = idx[:, t]
            if mode == "predictable learner":
                rG = (a + kG) / (a + b + nG)
            elif mode == "frozen r = prior":
                rG = np.full(reps, a / (a + b))
            else:
                rG = np.full(reps, 0.25)
            r_i = np.where(G[i], rG, 0.05)
            y = M[i] * (r_i < u) - eps * D[i]
            logk = mixture_step(logk, y + eps, m, betas)
            new = ~seen[ar, i]
            seen[ar, i] = True
            nG += new & G[i]
            kG += new & G[i] & (M[i] > 0)
            found_other += new & ~G[i] & (M[i] > 0)
            mx = logk.max(1)
            now = (mx + np.log(np.exp(logk - mx[:, None]).mean(1)) >= np.log(1 / DELTA)) & ~rej
            if now.any():                                # stop-at-certification: review with current r
                rG_now = (a + kG) / (a + b + nG) if mode == "predictable learner" else rG
                fm = (10 - found_other) + (100 - kG) * (rG_now < u)
                stop_cert |= now
                stop_false |= now & (fm > eps * 500)
            rej |= now
        rG_end = (a + kG) / (a + b + nG) if mode == "predictable learner" else rG
        final_m = (10 - found_other) + (100 - kG) * (rG_end < u)
        bud_false = rej & (final_m > eps * 500)
        out[mode] = (rej.mean(), bud_false.mean(), stop_cert.mean(), stop_false.mean(),
                     np.mean(rG_end < u))
        print(f"A  Satz 2, {mode:<24s}: certifies by n={n} {rej.mean():.3f}; FALSE certificates with "
              f"budget stop {bud_false.mean():.3f}, with stop-at-certification {stop_false.mean():.3f} "
              f"(nominal <= {DELTA}); G left unreviewed at the end in {np.mean(rG_end < u):.2f}")
    # two-phase: learn r_G on 300 draws, freeze, 300 fresh draws from U'' = U' minus phase-1 units
    false2 = 0
    cert2 = 0
    for _ in range(reps):
        i1 = RNG.integers(0, N, 300)
        s1 = np.unique(i1)
        rG = (a + (M[s1] * G[s1]).sum()) / (a + b + G[s1].sum())
        rest = np.setdiff1d(np.arange(N), s1)
        i2 = rest[RNG.integers(0, len(rest), 300)]
        r_i = np.where(G[i2], rG, 0.05)
        y = M[i2] * (r_i < u) - eps * D[i2]
        lvl = eps * D[s1].sum() / len(rest)              # phase-1 units: known stratum, M = 0, D known
        x = (y + eps)
        mm = lvl + eps
        bet = np.linspace(0, 0.99 / (1 - mm), nb + 1)[1:]
        k = np.exp(np.cumsum(np.log1p(bet[None, :] * (mm - x[:, None])), 0)).mean(1)
        c = (k >= 1 / DELTA).any()
        s2 = np.union1d(s1, np.unique(i2))
        unaud = np.setdiff1d(np.arange(N), s2)
        fm = (M[unaud] * (np.where(G[unaud], rG, 0.05) < u)).sum()
        cert2 += c
        false2 += c and fm > eps * 500
    print(f"A  Satz 2, two-phase (learn 300, freeze, 300 fresh from U''): certifies {cert2 / reps:.3f}; "
          f"false certificates {false2 / reps:.3f}")
    return out


# ------------------------------------------------------------------------------------------- B
def simulate_shift(n, rng, dA=0.0, dB=0.0, vis=1.0):
    x = rng.choice(3, n, p=cr.BIN_P)
    pa = cr.expit(cr.logit(cr.PA[x]) + dA)
    pb = cr.expit(cr.logit(cr.PB[x]) + dB)
    v = rng.random(n) < vis
    return x, ((rng.random(n) < pa) & v).astype(int), ((rng.random(n) < pb) & v).astype(int)


def transport_leaks(reps=2000, n_real=300, n_syn=600):
    cases = [("A-only leak: synthetic logit_A +1", dict(dA=1.0), {}),
             ("both-detector leak: synthetic logit_A,B +1", dict(dA=1.0, dB=1.0), {}),
             ("proportional leak: 20% of real invisible to all", {}, dict(vis=0.8))]
    for name, syn_kw, real_kw in cases:
        rej, rel, rule = [], [], []
        for _ in range(reps):
            xs, as_, bs = simulate_shift(n_syn, RNG, **syn_kw)
            syn = cr.table(xs, as_, bs)
            x, a, b = simulate_shift(n_real, RNG, **real_kw)
            real = np.hstack([cr.table(x, a, b)[:, :3], np.zeros((3, 1))])
            ht, hg, hyb, _ = cr.estimators(real, syn)
            r = cr.transport_test(real, syn)
            rej.append(r)
            rel.append(np.array([ht, hg]) / n_real - 1)
            rule.append((hg if r else ht) / n_real - 1)
        rel = np.array(rel)
        print(f"B  {name:<48s}: test rejects {np.mean(rej):.2f} | rel. bias HT {rel[:, 0].mean():+.3f} "
              f"HG {rel[:, 1].mean():+.3f} | 06 rule {np.mean(rule):+.3f}")


# ------------------------------------------------------------------------------------------- C
def bet_cap_drift():
    N, gamma, r, dmax, n = 2000, 0.3, 0.8, 6, 300
    q_det = 0.7 / 160 + gamma / N
    q_floor = gamma / N
    p_det, p_miss = 160 * q_det, 40 * q_floor
    z_det, z_miss = -(1 - r) / (N * q_det), r / (N * q_floor)
    mu = p_det * z_det + p_miss * z_miss
    var = p_det * z_det ** 2 + p_miss * z_miss ** 2 - mu ** 2
    for n in (300, 1000):
        b_wsr = np.sqrt(2 * np.log(1 / DELTA) / (n * (var + 0.25 / n)))
        b_det = min(b_wsr, 0.5 / (r * dmax / (N * q_det)))
        b_miss = min(b_wsr, 0.5 / (r * dmax / (N * q_floor)))
        drift = p_det * b_det * (0 - z_det) + p_miss * b_miss * (0 - z_miss)
        b_pred = min(b_wsr, 0.5 / (r * dmax / (N * q_floor)))
        print(f"C  per-draw cap, n={n}: mu={mu:+.2e} (H true), E[beta(c-Z)] per step {drift:+.4f} "
              f"(> 0 breaks the supermartingale); predictable cap: {b_pred * (0 - mu):+.1e}")
    # sim2 zmax_pred: fraction of benign pools where some unit has Z > zmax_pred for some r
    grid = np.linspace(0, 1, 201)
    rr = grid[None, :]
    viol_pools, n_pools = 0, 0
    for prev in (0.03, 0.10, 0.30):
        for _ in range(40):
            D, M, fH, fM, fD, s2 = make_pool(N, prev, RNG)
            sig = np.sqrt(s2 + 0.05 * fD + 1e-3)
            q = 0.7 * sig / sig.sum() + 0.3 / N
            fMh = fD - fH
            f_all = (fMh[:, None] - (1 - rr) * fD[:, None]).sum(0) / N
            y = M[:, None] - (1 - rr) * D[:, None]
            f = fMh[:, None] - (1 - rr) * fD[:, None]
            Z = f_all[None, :] + (y - f) / (N * q[:, None])
            zp = f_all + (rr[0] * DMAX + np.abs(fMh).max()) / (N * q.min())
            viol_pools += (Z > zp[None, :] + 1e-12).any()
            n_pools += 1
    print(f"C  sim2.py zmax_pred is violated by some drawable unit in {viol_pools}/{n_pools} benign pools")


# ------------------------------------------------------------------------------------------- D
def estimate_ratio_checks(reps=2000, N=2000):
    for n in (50, 150):
        cov = {"good f": 0, "bad f (heavy-tailed)": 0, "no f (lam=0)": 0}
        und = {k: 0 for k in cov}
        for _ in range(reps):
            D = np.where(RNG.random(N) < 0.05, 1 + RNG.poisson(0.6, N), 0)
            H = RNG.binomial(D, 0.8)
            rho = H.sum() / D.sum()
            s = RNG.choice(N, n, replace=False)
            rest = np.setdiff1d(np.arange(N), s)
            noise = RNG.lognormal(0, 1.5, N) * (RNG.random(N) < 0.1)
            preds = {"good f": (H + 0.1 * RNG.random(N), D + 0.1 * RNG.random(N), 1.0),
                     "bad f (heavy-tailed)": (0.8 * noise, noise, 1.0),
                     "no f (lam=0)": (np.zeros(N), np.zeros(N) + 1e-9, 0.0)}
            for k, (fa, fb, lam) in preds.items():
                e, lo, hi = ppi.estimate_ratio(H[s], D[s], fa[s], fb[s], fa[rest], fb[rest],
                                               alpha=2 * DELTA, lam=lam)
                if lo != lo:
                    und[k] += 1
                    lo = 0.0
                cov[k] += lo <= rho
        print(f"D  estimate_ratio SRS n={n}, one-sided 95% LCB coverage: " + " | ".join(
            f"{k} {cov[k] / reps:.3f}" for k in cov) + f" (undefined runs counted as covered: "
            + ", ".join(f"{und[k] / reps:.2f}" for k in und) + ")")
    # PPS bias of the unweighted estimator: draw prop. to a score that is high where misses are
    bias = []
    for _ in range(500):
        D = np.where(RNG.random(N) < 0.1, 1 + RNG.poisson(0.6, N), 0)
        H = RNG.binomial(D, 0.8)
        w = 0.3 / N + 0.7 * (D - H + 0.05) / (D - H + 0.05).sum()
        s = RNG.choice(N, 150, replace=False, p=w)
        rest = np.setdiff1d(np.arange(N), s)
        fa, fb = 0.8 * D.astype(float), D.astype(float)     # even with a perfect-mean predictor
        e, _, _ = ppi.estimate_ratio(H[s], D[s], fa[s], fb[s], fa[rest], fb[rest], lam=1.0)
        bias.append(e - H.sum() / D.sum())
    print(f"D  estimate_ratio under PPS (draws prop. to misses), unweighted: mean bias "
          f"{np.mean(bias):+.3f} (true recall ~0.80)")


# ------------------------------------------------------------------------------------------- E
def growth_bounds():
    """Jensen: for any bet beta <= c/(z_max - m) and H false with mean gap mu < m, the expected log-wealth
    increment is <= beta*(m - mu) <= c*(m - mu)/(z_max - m). Uniform draws, m = 0."""
    lg = np.log(1 / DELTA)
    r = 0.9
    # count recall: tile prevalence 0.1, mean 1.6 defects per defective tile, true recall 0.95
    mu = 0.1 * (0.05 * 1.6 - (1 - r) * 1.6)
    for name, b in (("D_max = 6 (04 S1)", 6), ("D_max = 2 (tiles, 00 S4)", 2),
                    ("D_max = (256/5)^2 (physically safe, merge_radius 2)", (256 // 5) ** 2)):
        g = 0.95 * (-mu) / (r * b)
        print(f"E  count recall, {name:<52s}: growth <= {g:.2e}/draw -> >= {lg / g:,.0f} draws")
    # area recall: mean defect area 200 px in defective tiles, area recall 0.95, bound = tile area
    mu_a = 0.1 * (0.05 * 200 - (1 - r) * 200)
    for name, b in (("area bound = 256^2 px (F1 fix)", 256 ** 2), ("area bound = 512^2 px", 512 ** 2)):
        g = 0.95 * (-mu_a) / (r * b)
        print(f"E  area recall, {name:<53s}: growth <= {g:.2e}/draw -> >= {lg / g:,.0f} draws")
    for n, N in ((300, 2000), (1000, 2000)):
        print(f"E  no-bound impossibility: any valid LCB of count recall exceeds eps with prob "
              f"<= delta*N/(N-n) = {DELTA * N / (N - n):.3f} on EVERY population (N={N}, n={n})")


# ------------------------------------------------------------------------------------------- F
def prop3_zero():
    m = np.array([0.0, 0.1, 0.2, 0.05])
    s = np.array([0.0, 0.5, 0.3, 0.2])
    p = prop3_closed(m, s, kappa=1.0, B=1.5)
    print(f"F  Prop. 3 closed form with sigma = {s.tolist()}: pi = {np.round(p, 3).tolist()} "
          f"(unit with sigma = 0 is never audited without a floor)")


if __name__ == "__main__":
    satz2_predictable_learner()
    transport_leaks()
    bet_cap_drift()
    estimate_ratio_checks()
    growth_bounds()
    prop3_zero()
