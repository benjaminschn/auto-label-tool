"""Upper bounds on what a per-image 'hidden miss' predictor (06 §5: m_i from N_hat and injection hardness)
can buy in the design-based audit, population-level (no sampling noise in lambda/beta).

Pool N=2000. Misses = candidate misses (detected by some detector, unselected; q_r calibrated)
                   + hidden misses (no candidate), Poisson with intensity lam_i = theta * h_i.
Hardness h_i: hidden-miss intensity is concentrated in a share s of images (h=1 there, eps elsewhere);
the injection map w_i observes h_i with rank-correlation controlled by 'fidelity'.
Reported (relative to audit-only SRS, i.e. Var(y)):
  CV(a)       : control variate from candidates only (Sigma q_r), lambda optimal
  CV(a+N_hat w): 06 predictor with N_hat = c * truth, single lambda (c in 0.5, 1, 1.5), and lambda=1
  GREG(a,w)   : two-coefficient regression (N_hat irrelevant)
  PPS gain    : variance factor of optimal pi ~ sqrt(E[y^2|x]) vs uniform, using x=(a,w)
"""
import numpy as np
rng = np.random.default_rng(1)
N = 2000
def run(s, fidelity, hidden_total=15, reps=200):
    acc = {k: [] for k in ("cv_a", "cv_1_c0.5", "cv_1_c1", "cv_1_c1.5", "lam1_c1", "lam1_c0.5", "lam1_c2", "greg", "pps_a", "pps_aw")}
    for _ in range(reps):
        nc = rng.poisson(0.15, N)
        a = np.array([rng.beta(2, 3, k).sum() for k in nc])
        cand_miss = np.array([(rng.random(k) < rng.beta(2, 3, k)).sum() for k in nc])  # q calibrated in expectation
        conc = rng.random(N) < s
        h = np.where(conc, 1.0, 0.02)
        lam = hidden_total * h / h.sum()
        hid = rng.poisson(lam)
        # injection map: noisy view of h (fidelity=1 perfect, 0 independent)
        z = fidelity * (conc.astype(float) - s) / np.sqrt(s * (1 - s)) + np.sqrt(1 - fidelity**2) * rng.normal(size=N)
        w = 1 / (1 + np.exp(-2 * z)); w = w / w.mean()
        y = cand_miss + hid
        V0 = y.var()
        def cv(f):
            c = np.cov(y, f)[0, 1] / f.var(); return (y - c * f).var() / V0
        acc["cv_a"].append(cv(a))
        for c in (0.5, 1, 1.5):
            f = a + c * hidden_total * w / w.sum()
            acc[f"cv_1_c{c}"].append(cv(f))
        for c in (0.5, 1, 2):
            f = a + c * hidden_total * w / w.sum()
            if f"lam1_c{c}" in acc: acc[f"lam1_c{c}"].append((y - f).var() / V0)
        X = np.column_stack([np.ones(N), a, w]); beta = np.linalg.lstsq(X, y, rcond=None)[0]
        acc["greg"].append((y - X @ beta).var() / V0)
        # PPS: optimal pi_i ~ sqrt(E[y^2 | x]); approximate E[y^2|x] by oracle conditional second moment
        m_a = a; v_a = (a - a**2 / np.maximum(nc, 1))  # rough Var of candidate misses
        Ey2_a = v_a + m_a**2 + hidden_total / N + (hidden_total / N) ** 2          # hidden unknown -> flat
        lam_hat = hidden_total * w / w.sum()
        Ey2_aw = v_a + m_a**2 + lam_hat + lam_hat**2 + 2 * m_a * lam_hat
        # HT variance for total with n draws WR: (1/n) [sum y^2/p - T^2]; ratio to uniform with p ~ sqrt(Ey2)
        T = y.sum()
        def vht(p):
            p = p / p.sum(); return (np.sum(y**2 / p) - T**2)
        vu = vht(np.ones(N))
        acc["pps_a"].append(vht(np.sqrt(Ey2_a)) / vu)
        acc["pps_aw"].append(vht(np.sqrt(Ey2_aw)) / vu)
    return {k: np.mean(v) for k, v in acc.items()}
for s in (0.05, 0.2, 1.0):
    for fid in (0.9, 0.5):
        if s == 1.0 and fid == 0.5: continue
        r = run(s, fid if s < 1 else 0.0)
        print(f"hidden misses concentrated in s={s:.2f} of images, injection-map fidelity={fid if s<1 else 'n/a'}: "
              + " | ".join(f"{k} {v:.3f}" for k, v in r.items()))
