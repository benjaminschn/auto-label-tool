"""Second red-team pass (mathematical correctness) on the design-based audit of 04 / S4 of 00.

  (1) Non-predictable bet cap. sims/sim_recall_cert.py (source of the 'Betting-Fieller' column of 04 S1)
      caps the bet with zmax evaluated at the *current* draw i_t, i.e. beta_t depends on i_t and is not
      F_{t-1}-measurable. Satz 1 itself prescribes the predictable cap max_i(...). Adversarial pool from
      04 §4.10 (misses in units with the smallest q): type-I error of both variants.
  (2) D_max is a population assumption, not a design property: one unsampled unit that exceeds the
      pre-registered D_max makes the certificate false with probability ~ P(unit never drawn).
  (3) Anytime-valid replacement for the 'exact hypergeometric bound' under monitoring: posterior-prior
      ratio (PPR) martingale for sampling without replacement (Waudby-Smith & Ramdas 2020).
  (4) Haystack bound with packing: the worst case allowed by D_max needs ceil(k/D_max) units, not k.
  (5) Re-computation of the exact numbers quoted in 04 S4 / §0.8 (188 / 57 / 18, n = 640).

Run:  python3 docs/research/sims/redteam2_betting_validity.py      (~1-2 min)
"""
import os
import sys

import numpy as np
from scipy.special import gammaln
from scipy.stats import hypergeom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_recall_cert import ucb_reject  # noqa: E402  (the bet used for 04 S1)

RNG = np.random.default_rng(11)
DELTA = 0.05


# --------------------------------------------------------------------------------------------- (1)
def nonpredictable_cap():
    """N=2000 images; 160 detected single defects in high-sigma images, 40 missed defects in images
    without any signal (sigma ~ 0, q at the floor), 1800 clean. Recall rho = 0.8 exactly.
    H(r): rho <= r is TRUE for r >= 0.8; any rejection of H(0.80) or H(0.82) is a false certificate."""
    N, gamma, dmax = 2000, 0.3, 6
    D = np.zeros(N)
    M = np.zeros(N)
    D[:200] = 1
    M[160:200] = 1                       # missed, no signal
    sig = np.zeros(N)
    sig[:160] = 1.0                      # S3 is confident where A already found the defect
    q = (1 - gamma) * sig / sig.sum() + gamma / N
    rho = 1 - M.sum() / D.sum()
    reps = 2000
    print(f"(1) adversarial pool: true recall {rho:.3f}; q at floor for the 40 missed defects")
    for n in (300, 1000):
        for r in (0.80, 0.82):
            idx = RNG.choice(N, size=(n, reps), p=q)
            g = M[idx] - (1 - r) * D[idx]
            Z = g / (N * q[idx])                                   # lambda = 0, F_bar = 0, level c = 0
            zmax_draw = (r * dmax) / (N * q[idx])                  # as in sim_recall_cert.betting_lcb
            zmax_pred = np.full_like(Z, (r * dmax) / (N * q.min()))  # Satz 1: max over i, predictable
            c = np.zeros(reps)
            rej_draw = ucb_reject(Z, zmax_draw, c, DELTA, n_plan=n).mean()
            rej_pred = ucb_reject(Z, zmax_pred, c, DELTA, n_plan=n).mean()
            print(f"    n={n:5d}, H(r={r:.2f}) true: false rejection with per-draw cap {rej_draw:.3f} | "
                  f"with predictable cap {rej_pred:.3f}   (nominal <= {DELTA})")


# --------------------------------------------------------------------------------------------- (2)
def mix_wealth_reject(g_draws, grid, dmax_assumed, nb=12):
    """Uniform WR draws, Z = g (no IPW needed), H(r): mean g(r) >= 0, beta-grid mixture with the
    predictable cap 0.95/(r*D_max). Wealth factors are clipped at 0 when D_max is violated."""
    T, R = g_draws.shape[0], g_draws.shape[1]
    zmax = np.maximum(grid * dmax_assumed, 1e-9)
    betas = (0.95 / zmax)[None, :] * np.logspace(-2.5, 0, nb)[:, None]          # (nb, G)
    logk = np.zeros((R, nb, len(grid)))
    rej = np.zeros((R, len(grid)), bool)
    for t in range(T):
        fac = 1 + betas[None] * (0.0 - g_draws[t][:, None, :])                 # (R, nb, G)
        logk += np.log(np.maximum(fac, 1e-300))                               # clip: wealth -> ~0
        mx = logk.max(1)
        rej |= mx + np.log(np.exp(logk - mx[:, None, :]).mean(1)) >= np.log(1 / DELTA)
    return rej


def dmax_violation():
    """2000 images, 400 single-defect images with 4 misses (rho_base 0.99) plus ONE image with 100
    missed micro-defects (pitting) -> true rho = 1 - 104/500 = 0.792. Pre-registered D_max = 6.
    Claim to certify: rho >= 0.80 (FALSE)."""
    N, dmax, reps = 2000, 6, 300
    D = np.zeros(N)
    M = np.zeros(N)
    D[:400] = 1
    M[:4] = 1
    D[400], M[400] = 100, 100
    rho = 1 - M.sum() / D.sum()
    grid = np.array([0.5, 0.6, 0.7, 0.75, 0.8])
    print(f"(2) D_max violated by one unit: true recall {rho:.3f}; target 'rho >= 0.80' is FALSE")
    for n in (1500, 3000):
        idx = RNG.integers(0, N, size=(n, reps))
        g = M[idx][..., None] - (1 - grid) * D[idx][..., None]                 # (n, reps, G)
        rej = mix_wealth_reject(g, grid, dmax)
        cert = rej.all(1)
        heavy_seen = (idx == 400).any(0)
        print(f"    n={n} WR draws: certificate 'rho >= 0.80' issued in {cert.mean():.3f} of runs "
              f"(nominal <= {DELTA}); heavy unit never drawn in {1 - heavy_seen.mean():.3f}; "
              f"false certs among runs that never drew it {cert[~heavy_seen].mean():.3f}")


# --------------------------------------------------------------------------------------------- (3)
def ppr_anytime():
    N, reps, n_max, step = 2000, 1000, 800, 5
    Ks = np.arange(N + 1)
    a, b = 1.0, 100.0                                   # beta-binomial prior on K (choice affects power only)
    logprior = (gammaln(N + 1) - gammaln(Ks + 1) - gammaln(N - Ks + 1)
                + gammaln(Ks + a) + gammaln(N - Ks + b) - gammaln(N + a + b)
                - (gammaln(a) + gammaln(b) - gammaln(a + b)))
    target_K = 40                                       # certificate: zeta <= 0.02  <=>  K <= 40
    for K_true, label in ((41, "true zeta 0.0205 (certificate false)"), (10, "true zeta 0.005")):
        pop = np.zeros(N, int)
        pop[:K_true] = 1
        cert, stops = 0, []
        for _ in range(reps):
            cum = np.cumsum(RNG.permutation(pop)[:n_max])
            for n in range(step, n_max + 1, step):
                k = cum[n - 1]
                ok = (Ks >= k) & (N - Ks >= n - k)
                logL = np.full(N + 1, -np.inf)
                Kk = Ks[ok]
                logL[ok] = (gammaln(Kk + 1) - gammaln(Kk - k + 1) + gammaln(N - Kk + 1)
                            - gammaln(N - Kk - (n - k) + 1))
                lm = logprior + logL
                mx = lm[ok].max()
                logmarg = mx + np.log(np.exp(lm[ok] - mx).sum())
                logM = logmarg - logL                    # PPR martingale for each K (inf where L=0)
                if np.all(logM[target_K + 1:] >= np.log(1 / DELTA)):
                    cert += 1
                    stops.append(n)
                    break
        print(f"(3) PPR anytime WOR bound, monitored every {step} draws up to {n_max}, {label}: "
              f"certifies zeta <= 0.02 in {cert / reps:.3f}"
              + (f" (median stop n={np.median(stops):.0f})" if stops else ""))


# --------------------------------------------------------------------------------------------- (4)/(5)
def haystack_and_numbers():
    k = 10
    for dm in (1, 2, 6):
        units = int(np.ceil(k / dm))
        print(f"(4) haystack, k={k} misses, worst case packed into ceil(k/D_max)={units} units "
              f"(D_max={dm}): required inspected share of the no-signal stratum = "
              f"{1 - DELTA ** (1 / units):.3f}")
    N = 2000
    for n in (30, 100, 300):
        Ks = np.arange(0, N + 1)
        ucb = Ks[hypergeom.cdf(0, N, Ks, n) > DELTA].max()
        print(f"(5) 04 S4: N={N}, n={n}, 0 bad found -> exact 95% UCB of #bad = {ucb}")
    for n in range(500, 800):
        if hypergeom.cdf(0, N, 8, n) <= DELTA:
            print(f"(5) 04 §0.8: smallest n with exact UCB <= 7 (0 found) = {n}; binomial formula "
                  f"N ln(1/delta)/8 = {N * np.log(1 / DELTA) / 8:.0f}")
            break


if __name__ == "__main__":
    nonpredictable_cap()
    dmax_violation()
    ppr_anytime()
    haystack_and_numbers()
