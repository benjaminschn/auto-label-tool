import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import redteam2_verify_feasibility_checks as v
def first_rejection_lin(Z, zmax, m, nb=20):
    cap = 0.99 / (zmax - m)
    betas = np.linspace(0, cap, nb + 1)[1:]
    logk = np.cumsum(np.log1p(betas[None, :] * (m - Z[:, None])), axis=0)
    mx = logk.max(1)
    lm = mx + np.log(np.exp(logk - mx[:, None]).mean(1))
    hit = np.nonzero(lm >= np.log(1 / v.DELTA))[0]
    return hit[0] if len(hit) else None
v.first_rejection = first_rejection_lin
v.p1_zeta_efficiency(reps=150)
