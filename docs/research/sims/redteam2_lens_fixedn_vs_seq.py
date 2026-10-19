"""Fixed-n exact hypergeometric power vs. the sequential numbers of redteam (b) (betting) and redteam2 (3) (PPR).
N=2000, K bad, certify K <= 40 (zeta <= 0.02), no crediting of corrections (as in (b))."""
import numpy as np
from scipy.stats import hypergeom
N, D = 2000, 0.05
def ucb(k, n):
    Ks = np.arange(k, N + 1)
    return Ks[hypergeom.cdf(k, N, Ks, n) > D].max()
for K in (10, 20, 30):
    out = []
    for target_pw in (0.5, 0.9):
        for n in range(20, 2001, 5):
            xs = np.arange(0, min(n, K) + 1)
            pw = (hypergeom.pmf(xs, N, K, n) * np.array([ucb(x, n) <= 40 for x in xs])).sum()
            if pw >= target_pw:
                out.append((target_pw, n)); break
    print(f"K={K} (zeta={K/N:.4f}): fixed-n exact SRSWOR, smallest n with P(certify zeta<=0.02) >= 0.5: {out[0][1]}, >= 0.9: {out[1][1]}")
