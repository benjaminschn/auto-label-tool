"""Exact planning table: audit size n (SRSWOR from U', exact hypergeometric UCB, corrections credited)
needed for P(certify zeta_final <= K*/N) >= 0.9, as a function of the residual number K' of bad images in U'."""
import numpy as np
from scipy.stats import hypergeom
D = 0.05
def ucb_table(n, Np, xmax):
    Ks = np.arange(0, Np + 1)
    out = np.empty(xmax + 1, int)
    for x in range(xmax + 1):
        c = hypergeom.cdf(x, Np, Ks, n)
        out[x] = Ks[(c > D) & (Ks >= x)].max()
    return out
def n_needed(Kp, Np, Kstar, pw=0.9, ns=None):
    for n in ns:
        xs = np.arange(0, min(n, Kp) + 1)
        u = ucb_table(n, Np, int(xs.max()))
        ok = (u[xs] - xs) <= Kstar
        if (hypergeom.pmf(xs, Np, Kp, n) * ok).sum() >= pw:
            return n
    return None
ns = list(range(10, 200, 5)) + list(range(200, 1801, 10))
for N, Np, Kstar in ((2000, 1850, 40), (2000, 1850, 20), (10000, 9500, 200), (10000, 9500, 100)):
    row = []
    for frac in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75):
        Kp = int(round(frac * Kstar))
        row.append(f"K'={Kp}:{n_needed(Kp, Np, Kstar, ns=ns if N == 2000 else [int(x * 5) for x in ns])}")
    print(f"N={N}, N'={Np}, target zeta*={Kstar / N:.3f}: n_90 by residual bad -> " + "  ".join(row), flush=True)
