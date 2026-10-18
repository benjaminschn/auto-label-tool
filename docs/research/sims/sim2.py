import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sim_recall_cert import make_pool, DMAX
from scipy.stats import norm

def draws_pps(N, q, n_budget, rng):
    seen, d = set(), []
    while len(seen) < n_budget:
        i = rng.choice(N, p=q); d.append(i); seen.add(i)
    return np.array(d)

def terms(D, M, fH, fD, q, idx, grid):
    N = len(D); r = grid[None, :]
    fMh = fD - fH
    f_all = (fMh[:, None] - (1 - r) * fD[:, None]).sum(0) / N
    y = M[idx][:, None] - (1 - r) * D[idx][:, None]
    f = fMh[idx][:, None] - (1 - r) * fD[idx][:, None]
    Z = f_all[None, :] + (y - f) / (N * q[idx][:, None])
    zmax = f_all[None, :] + (r * DMAX - f) / (N * q[idx][:, None])
    zmax_pred = f_all[None, :] + (r * DMAX + (1 - r) * fD.max() * 0 + np.abs(fMh).max()) / (N * q.min())  # predictable worst case
    return Z, np.broadcast_to(zmax_pred, Z.shape)

def mix_reject(Z, zmax, c, delta, nb=24):
    T, R = Z.shape
    caps = 0.95 / np.maximum(zmax[0] - c, 1e-12)            # (R,)
    betas = caps[None, :] * np.logspace(-3, 0, nb)[:, None]  # (nb,R)
    logK = np.zeros((nb, R)); rej = np.zeros(R, bool)
    for t in range(T):
        logK += np.log1p(betas * (c - Z[t])[None, :])
        K = np.exp(logK).mean(0)
        rej |= K >= 1 / delta
    return rej

def lcb_from_rej(rej, grid):
    nr = np.where(~rej)[0]
    if len(nr) == 0: return 1.0
    j = nr[0]; return grid[j - 1] if j > 0 else 0.0

def clt_lcb(Z, grid, delta):
    m = Z.mean(0); s = Z.std(0, ddof=1) / np.sqrt(len(Z))
    rej = m + norm.isf(delta) * s < 0
    return lcb_from_rej(rej, grid)

rng = np.random.default_rng(7); N, delta, reps = 2000, 0.05, 200
grid = np.linspace(0, 1, 201)
for prev in (0.03, 0.10, 0.30):
    for n in (30, 100):
        res = {k: [] for k in ("bet_g0.3", "bet_unif", "clt_g0.3", "clt_unif")}; rhos = []
        for _ in range(reps):
            D, M, fH, fM, fD, s2 = make_pool(N, prev, rng)
            rho = 1 - M.sum() / D.sum(); rhos.append(rho)
            sig = np.sqrt(s2 + 0.05 * fD + 1e-3)
            for name, gam in (("g0.3", 0.3), ("unif", 1.0)):
                q = (1 - gam) * sig / sig.sum() + gam / N
                idx = draws_pps(N, q, n, rng)
                Z, zmax = terms(D, M, fH, fD, q, idx, grid)
                b = lcb_from_rej(mix_reject(Z, zmax, np.zeros(len(grid)), delta), grid)
                res["bet_" + name].append((b <= rho, b))
                c = clt_lcb(Z, grid, delta)
                res["clt_" + name].append((c <= rho, c))
        print(f"prev={prev:.2f} n={n:3d} rho~{np.mean(rhos):.3f} | " + " | ".join(
            f"{k}: cov={np.mean([a for a,_ in v]):.3f} LCB={np.mean([b for _,b in v]):.3f}" for k, v in res.items()), flush=True)
