"""Feasibility numbers: haystack vs prevalence, vacuity of absolute zeta, Bonferroni over classes x metrics."""
import numpy as np
from scipy.stats import hypergeom
DELTA=0.05
def n_zero_fail(N, K, delta):
    """smallest n (WOR) such that P(0 hits | K bad among N) <= delta"""
    for n in range(1, N+1):
        if hypergeom.pmf(0, N, K, n) <= delta: return n
    return N
print("A) recall >= 0.9 certificate, worst case: k+1 = floor(0.1*D_tot)+1 misses hidden in signal-free units")
N=2000
for p in (0.005,0.01,0.03,0.1,0.3):
    Dtot = int(round(N*p*1.6))
    k = int(np.floor(0.1*Dtot))+1
    n = n_zero_fail(N - int(N*p), k, DELTA)  # clean-predicted stratum ~ N(1-p)
    print(f"  p={p:5.3f}: D_tot~{Dtot:4d}, false-claim misses={k:3d} -> inspect >= {n:4d} of {N-int(N*p)} clean units ({n/(N-int(N*p)):.0%})")
print("B) absolute zeta certificate zeta<=z*: zero-failure n (WOR, N=2000) and what it means relative to prevalence")
for zs in (0.02,0.01,0.005,0.002):
    K = int(np.floor(zs*N))+1
    n = n_zero_fail(N, K, DELTA)
    print(f"  zeta*={zs:5.3f}: n>={n:4d} ({n/N:.0%});  implied image-recall >= 1-z*/p: p=0.3 -> {1-zs/0.3:.2f}, p=0.1 -> {1-zs/0.1:.2f}, p=0.03 -> {1-zs/0.03:.2f}, p=0.01 -> {max(0,1-zs/0.01):.2f}")
print("C) Bonferroni over J = classes x metrics, zero-failure n for zeta_c <= 0.02 per class (N=2000)")
for J in (1,2,6,10,20):
    K=41
    print(f"  J={J:2d}: n>={n_zero_fail(N,K,DELTA/J)} ; ln(J/delta)/ln(1/delta) = {np.log(J/DELTA)/np.log(1/DELTA):.2f}")
print("D) reviewer sensitivity s: zero-failure n to certify zeta<=0.02 if s known and bound divided by s")
for s in (1.0,0.9,0.8,0.7,0.5):
    # need observed-rate bound s*z* -> K_obs = floor(s*0.02*N)+1
    K = int(np.floor(s*0.02*N))+1
    print(f"  s={s:.1f}: n>={n_zero_fail(N,K,DELTA)}")
