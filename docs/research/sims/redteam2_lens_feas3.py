"""Human != gold: does a sensitivity correction with seeded (injected) defects restore validity?
True zeta = 0.03 > target 0.025 (every certificate is false). Audit n units WOR from N=2000; reviewer
finds a real miss with prob s_real; reviewer sensitivity is estimated from m seeded defects found with prob
s_seed (seeded defects are easier when s_seed > s_real). Certificate: UCB_hyp(obs)/LCB_CP(s) <= target,
delta split 0.025/0.025. Also: two-phase design, expert (gold) re-inspects a random subsample of r audit
units; bound = hypergeometric UCB on observed + UCB on reviewer-missed share from phase 2 (Bonferroni)."""
import numpy as np
from scipy.stats import hypergeom, beta
RNG=np.random.default_rng(1); N=2000; K=60; TGT=0.025; REPS=2000
cache={}
def ucb_cnt(k,n,N,d):
    key=(k,n,N,d)
    if key not in cache:
        Ks=np.arange(k,N+1); cache[key]=Ks[hypergeom.cdf(k,N,Ks,n)>d].max()
    return cache[key]
def cp_lcb(x,m,d): return 0.0 if x==0 else beta.ppf(d,x,m-x+1)
def cp_ucb(x,m,d): return 1.0 if x==m else beta.ppf(1-d,x+1,m-x)
print("A) seeded-defect sensitivity correction")
for s_real,s_seed in ((0.7,0.7),(0.6,0.85),(0.5,0.9)):
    for n,m in ((1000,100),(1000,300)):
        cert=0
        for _ in range(REPS):
            pop=np.zeros(N,int); pop[:K]=1
            obs=pop[RNG.choice(N,n,replace=False)]*(RNG.random(n)<s_real)
            sh=cp_lcb(int((RNG.random(m)<s_seed).sum()),m,0.025)
            if sh>0 and ucb_cnt(int(obs.sum()),n,N,0.025)/N/sh<=TGT: cert+=1
        print(f"  s_real {s_real} s_seed {s_seed} n={n} seeds={m}: false-cert rate {cert/REPS:.3f} (nominal <= 0.05)")
print("B) two-phase: expert re-inspects r random audit units (gold), Bonferroni 0.025/0.025")
for s_real in (0.7,0.5):
    for n,r in ((1000,100),(1000,300),(400,400)):
        cert=0; 
        for _ in range(REPS):
            pop=np.zeros(N,int); pop[:K]=1
            S=RNG.choice(N,n,replace=False); found=pop[S]*(RNG.random(n)<s_real)
            sub=RNG.choice(n,r,replace=False)
            missed_by_rev = (pop[S][sub]-found[sub])  # expert finds what reviewer missed
            u1=ucb_cnt(int(found.sum()),n,N,0.025)/N
            # share of units (pop level) that are bad AND reviewer-missed: estimate from phase-2 as a WOR
            # sample of size r from N (two-phase -> phase-2 units are a SRS of U)
            u2=ucb_cnt(int(missed_by_rev.sum()),r,N,0.025)/N
            if u1+u2<=TGT: cert+=1
        print(f"  s_real {s_real} n={n} expert r={r}: false-cert rate {cert/REPS:.3f}")
# power of the two-phase design when the claim is TRUE (zeta=0.005)
print("C) power when true zeta = 0.005 (claim true), s_real 0.7")
for n,r in ((400,100),(1000,100),(1000,300),(600,600)):
    cert=0
    for _ in range(500):
        pop=np.zeros(N,int); pop[:10]=1
        S=RNG.choice(N,n,replace=False); found=pop[S]*(RNG.random(n)<0.7)
        sub=RNG.choice(n,r,replace=False)
        u=ucb_cnt(int(found.sum()),n,N,0.025)/N+ucb_cnt(int((pop[S][sub]-found[sub]).sum()),r,N,0.025)/N
        cert+= u<=TGT
    print(f"  n={n} reviewer + r={r} expert: certifies {cert/500:.3f}")
