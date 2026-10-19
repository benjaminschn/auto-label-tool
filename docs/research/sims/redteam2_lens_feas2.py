"""Does risk-proportional (PPS) sampling with IPW / PPI help to certify a SMALL binary rate zeta
(e.g. zeta <= 0.02 when true zeta = 0.005) compared to uniform sampling? Betting CS (beta-grid mixture),
with-replacement draws, N'=2000. Predictor: binormal score, AUC 0.7 / 0.9, calibrated f = P(bad|s).
Also: false-certificate rate when true zeta = 0.025 (> target 0.02)."""
import numpy as np
from scipy.stats import norm
RNG=np.random.default_rng(3); DELTA=0.05
N=2000; TARGET=0.02; NMAX=1500; REPS=400
def run(Kbad, auc, gamma, lam, reps=REPS):
    d = np.sqrt(2)*norm.ppf(auc) if auc>0.5 else 0.0
    y=np.zeros(N); y[:Kbad]=1
    stops=[]; cert=0
    for _ in range(reps):
        s = RNG.normal(0,1,N) + d*y
        pri = Kbad/N
        f = pri*norm.pdf(s-d)/(pri*norm.pdf(s-d)+(1-pri)*norm.pdf(s)) if d>0 else np.full(N,pri)
        q = (1-gamma)*f/f.sum() + gamma/N
        Fbar = f.mean()
        idx = RNG.choice(N, NMAX, p=q)
        Z = lam*Fbar + (y[idx]-lam*f[idx])/(N*q[idx])
        zmax = (lam*Fbar + ((1-lam*f)/(N*q))).max()   # predictable (fixed design)
        m=TARGET
        betas = np.linspace(0,0.99,21)[1:]/(zmax-m)
        logk = np.cumsum(np.log1p(betas[None,:]*(m-Z[:,None])),axis=0)
        mx=logk.max(1); lk = mx+np.log(np.exp(logk-mx[:,None]).mean(1))
        hit=np.nonzero(lk>=np.log(1/DELTA))[0]
        if len(hit):
            cert+=1; stops.append(len(np.unique(idx[:hit[0]+1])))
    return cert/reps, (np.median(stops) if stops else np.nan)
for Kbad,label in ((10,"true zeta 0.005 (cert correct)"),(50,"true zeta 0.025 (cert FALSE)")):
    print(label)
    print(f"  uniform, IPW=plain mean : cert {run(Kbad,0.5,1.0,0.0)}")
    for auc in (0.7,0.9):
        for gamma in (0.3,0.1):
            for lam in (0.0,1.0):
                c,m=run(Kbad,auc,gamma,lam)
                print(f"  PPS AUC {auc} floor {gamma} lambda {lam}: certifies {c:.3f}, median distinct units at stop {m}")
