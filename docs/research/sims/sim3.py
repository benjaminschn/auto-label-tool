"""S2: upper confidence bound (UCB) for zeta = share of images with a missed defect.

Columns: audit-only t (approximate), PPI lambda=1 (approximate, autolabel.ppi), exact hypergeometric
(valid for a fixed n registered in advance and uniform SRS without replacement; not anytime-valid,
07 F6) and PPI + WSR betting.

v2 (07_red_team.md, F16 / change 6): the betting UCB returned the largest NON-rejected grid point, which
is anti-conservative by up to one grid step (the true mean can lie between that point and the next one).
It now returns the grid point ABOVE the largest non-rejected one (or the upper end of the range).
Caveat (untested, not fixed here): the WSR bet treats the SRS-without-replacement audit as i.i.d.
draws; strictly valid only for sampling with replacement or with a without-replacement martingale
(Waudby-Smith & Ramdas 2020).

Run:  python3 docs/research/sims/sim3.py      (~2.5 min on a CPU)
"""
import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sim_recall_cert import make_pool
from scipy.stats import hypergeom, norm
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", ".."))
from autolabel import ppi

def hyper_ucb(k, n, N, delta):
    # smallest K s.t. P(X <= k | K) <= delta  -> UCB = that K - 1 ... search largest K with P(X<=k|K) > delta
    Ks = np.arange(k, N + 1)
    cdf = hypergeom.cdf(k, N, Ks, n)
    ok = Ks[cdf > delta]
    return ok.max() / N

def wsr_ucb_diff(Z, lo, hi, delta, grid):
    # one-sided UCB for mean of Z in [lo,hi] (iid, without-replacement treated as with; conservative-ish)
    x = (Z - lo) / (hi - lo); n = len(x)
    t = np.arange(1, n + 1)
    mu = (0.5 + np.cumsum(x)) / (1 + t)
    s2 = (0.25 + np.cumsum((x - mu) ** 2)) / (1 + t)
    s2p = np.append(0.25, s2[:-1])
    lam = np.sqrt(2 * np.log(1 / delta) / (n * s2p))
    m = (grid[:, None] - lo) / (hi - lo)
    lamc = np.minimum(lam[None, :], 0.75 / np.maximum(1 - m, 1e-9))
    logK = np.cumsum(np.log1p(lamc * (m - x[None, :])), axis=1)  # bets that mean < m
    rej = (logK >= np.log(1 / delta)).any(1)
    nr = np.where(~rej)[0]
    if len(nr) == 0:          # every grid point rejected: probability <= delta if the mean is >= grid[0]
        return grid[0]
    j = nr.max()
    # the true mean can lie anywhere in [grid[j], grid[j+1]) -> report the grid point above (07 F16)
    return grid[j + 1] if j + 1 < len(grid) else hi

rng = np.random.default_rng(3); N, delta, reps = 2000, 0.05, 300
grid = np.linspace(-0.2, 1, 1201)
for prev in (0.03, 0.10, 0.30):
    for n in (30, 100):
        out = {k: [] for k in ("hyper", "ppi_t", "ppi_wsr", "classical_t")}; tr = []
        for _ in range(reps):
            D, M, fH, fM, fD, s2 = make_pool(N, prev, rng)
            y = (M > 0).astype(float)
            # self-assessed P(missed defect): 1 - exp(-expected misses) as proxy
            f = 1 - np.exp(-(fD - fH))
            zeta = y.mean(); tr.append(zeta)
            s = rng.choice(N, n, replace=False); rest = np.setdiff1d(np.arange(N), s)
            k = int(y[s].sum())
            uh = hyper_ucb(k, n, N, delta)
            out["hyper"].append((uh >= zeta, uh))
            e, lo, hi = ppi.estimate_mean(y[s], f[s], f[rest], alpha=2 * delta)
            out["ppi_t"].append((hi >= zeta, hi))
            e, lo, hi = ppi.classical_mean(y[s], alpha=2 * delta, n_total=N)
            out["classical_t"].append((hi >= zeta, hi))
            Z = f.mean() + (y[s] - f[s])      # PPI difference terms, in [f.mean()-1, f.mean()+1]
            u = wsr_ucb_diff(Z, f.mean() - 1, f.mean() + 1, delta, grid)
            out["ppi_wsr"].append((u >= zeta, u))
        print(f"prev={prev:.2f} n={n:3d} zeta~{np.mean(tr):.4f} | " + " | ".join(
            f"{k}: cov={np.mean([a for a,_ in v]):.3f} UCB={np.mean([b for _,b in v]):.4f}" for k, v in out.items()), flush=True)
