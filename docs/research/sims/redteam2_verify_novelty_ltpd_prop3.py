"""Novelty-lens verification (red team 2) for findings M3/C2 on 00 §5 and 04 §4.4-4.5.

  (i)  M3/C2: Is CertAL's fixed-n zeta certificate (exact hypergeometric UCB <= zeta*) just a
       Dodge-Romig LTPD single-sampling plan with consumer's risk beta = delta? Checks the
       equivalence exactly and reports the ATI-optimal rectifying plan as the natural baseline.
  (ii) C2: How much does the Prop. 3 closed form gain over *known, tuned* two-stage designs
       (top-k review + uniform audit; top-k review + Neyman/active-inference audit; pure Neyman)?
       This measures the size of the novelty residue, not its correctness (red team (a) did that).

Run:  python docs/research/sims/redteam2_verify_novelty_ltpd_prop3.py      (CPU, ~10 s)
"""
import numpy as np
from scipy.stats import hypergeom

DELTA = 0.05
RNG = np.random.default_rng(11)


# ------------------------------------------------------------------------------------------ (i)
def hyper_ucb_count(k, n, N):
    Ks = np.arange(k, N + 1)
    return Ks[hypergeom.cdf(k, N, Ks, n) > DELTA].max()


def ltpd_c(n, N, K_ltpd, beta=DELTA):
    """Largest acceptance number c with P(accept | K_ltpd) <= beta (-1 if none)."""
    cs = np.arange(0, n + 1)
    ok = hypergeom.cdf(cs, N, K_ltpd, n) <= beta
    return cs[ok].max() if ok.any() else -1


def part_i():
    N, zstar = 2000, 0.02
    K_bad = int(np.floor(zstar * N)) + 1          # smallest count for which "zeta <= zstar" is false
    mism = 0
    for n in range(50, 1001, 50):
        c = ltpd_c(n, N, K_bad)
        for k in range(0, 31):
            cert = hyper_ucb_count(k, n, N) / N <= zstar
            if cert != (k <= c):
                mism += 1
    print(f"(i) N={N}, zeta*={zstar}: 'hypergeometric UCB <= zeta*' vs 'LTPD plan accepts (c=c(n), "
          f"LTPD count {K_bad}, beta={DELTA})' disagree in {mism} of {20 * 31} (n,k) cells")
    for K_avg in (5, 10, 20):
        best = None
        for n in range(20, 1201, 5):
            c = ltpd_c(n, N, K_bad)
            if c < 0:
                continue
            pa = hypergeom.cdf(c, N, K_avg, n)
            ati = n + (1 - pa) * (N - n)
            aoq = (K_avg / N) * pa * (N - n) / N
            if best is None or ati < best[0]:
                best = (ati, n, c, pa, aoq)
        ati, n, c, pa, aoq = best
        print(f"    process average zeta={K_avg / N:.4f}: ATI-optimal Dodge-Romig LTPD plan n={n}, c={c}, "
              f"P(accept)={pa:.3f}, ATI={ati:.0f} of {N} units, AOQ={aoq:.4f}")


# ----------------------------------------------------------------------------------------- (ii)
def capped_prop(w, budget):
    """pi_i = min(1, c*w_i) with sum pi = budget (water-filling)."""
    w = np.maximum(w, 1e-12)
    lo, hi = 0.0, 1e12
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if np.minimum(1, mid * w).sum() < budget else (lo, mid)
    return np.minimum(1, hi * w)


def prop3_closed(m, s, kappa, B):
    def pis(nu):
        p = np.where(m + kappa * s ** 2 >= nu, 1.0, s * np.sqrt(kappa / np.maximum(nu - m, 1e-300)))
        return np.minimum(p, 1.0)
    lo, hi = m.min(), m.max() + kappa * s.max() ** 2 + 1e6
    for _ in range(300):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pis(mid).sum() > B else (lo, mid)
    return pis(hi)


def objective(p, m, s, kappa):
    return np.sum((1 - p) * m) + kappa * np.sum((1 / p - 1) * s ** 2)


def two_stage(m, s, kappa, B, order, audit):
    best = np.inf
    idx = np.argsort(-order)
    for k in range(0, int(B) - 5, max(1, int(B) // 100)):
        p = np.zeros_like(m)
        rest = idx[k:]
        if audit == "uniform":
            p[rest] = (B - k) / len(rest)
        else:
            p[rest] = capped_prop(s[rest], B - k)
        p[idx[:k]] = 1.0
        best = min(best, objective(p, m, s, kappa))
    return best


def part_ii():
    N = 2000
    rows = []
    for scen in ("rare misses, informative risk", "rare misses + unseen mass (floor)",
                 "m and sigma decorrelated"):
        for B in (100, 300):
            for kappa in (0.03, 0.3, 3.0):
                ex = {"top-k(m)+uniform": [], "top-k(m)+Neyman": [], "top-k(m+k*s2)+Neyman": [], "Neyman only": []}
                for _ in range(6):
                    p_bad = RNG.beta(0.3, 8.0, N)                 # P(unit has a missed defect)
                    m = p_bad * (1 + RNG.poisson(0.4, N))         # expected misses
                    s2 = p_bad * (1 - p_bad) + (0.02 if "unseen" in scen else 0.002)
                    if "decorrelated" in scen:                    # variance driven by other features
                        s2 = RNG.permutation(s2) * RNG.lognormal(0, 1, N)
                    s = np.sqrt(s2)
                    if "unseen" in scen:
                        m = m + 0.01
                    j_star = objective(prop3_closed(m, s, kappa, B), m, s, kappa)
                    ex["top-k(m)+uniform"].append(two_stage(m, s, kappa, B, m, "uniform") / j_star - 1)
                    ex["top-k(m)+Neyman"].append(two_stage(m, s, kappa, B, m, "neyman") / j_star - 1)
                    ex["top-k(m+k*s2)+Neyman"].append(two_stage(m, s, kappa, B, m + kappa * s2, "neyman") / j_star - 1)
                    ex["Neyman only"].append(objective(capped_prop(s, B), m, s, kappa) / j_star - 1)
                rows.append((scen, B, kappa, {k: np.mean(v) for k, v in ex.items()}))
    print("(ii) Prop. 3 objective excess of known designs over the closed form pi* (mean over 6 pools, "
          "N=2000; tuned k for two-stage designs):")
    for scen, B, kappa, ex in rows:
        print(f"    {scen:40s} B={B:3d} kappa={kappa:5.2f}: " +
              " | ".join(f"{k} {100 * v:+6.1f}%" for k, v in ex.items()))


if __name__ == "__main__":
    part_i()
    part_ii()
