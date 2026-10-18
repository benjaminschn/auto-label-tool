"""Quick coverage check: current PPI ratio CI (delta method + t) vs. design-based betting/Fieller LCB.

Semi-synthetic pool of N images with rare defects; per-image self-assessment from candidate regions.
Target: pool-level defect recall rho = 1 - sum(M)/sum(D) (no review, no corrections) -> one-sided 95% LCB.
"""
import sys
import numpy as np

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", ".."))
from autolabel import ppi  # noqa: E402

DMAX = 6


def make_pool(N, prev, rng, inv=0.3):
    D = np.where(rng.random(N) < prev, 1 + rng.poisson(0.6, N), 0)
    D = np.minimum(D, DMAX)
    M = np.zeros(N, int)
    fM = np.zeros(N)
    fH = np.zeros(N)
    s2 = np.zeros(N)
    for i in range(N):
        vis = rng.beta(4, 1.2, D[i])                    # visibility of each defect
        det = rng.random(D[i]) < vis                    # detected by the pipeline
        M[i] = int((~det).sum())
        # candidates: undetected real defects visible as candidate with prob 1-inv
        p_un = rng.beta(2, 3, int(((~det) & (rng.random(D[i]) > inv)).sum()))
        p_false = rng.beta(1, 8, rng.poisson(0.3))      # false candidates
        p_sel = rng.beta(6, 2, int(det.sum()))
        fM[i] = p_un.sum() + p_false.sum()
        fH[i] = p_sel.sum()
        r = 0.9
        s2[i] = r**2 * (np.sum(p_un * (1 - p_un)) + np.sum(p_false * (1 - p_false))) \
            + (1 - r)**2 * np.sum(p_sel * (1 - p_sel))
    nu = inv * 0.3                                       # assumed no-candidate share (misspecified on purpose)
    fD = (fH + fM) / (1 - nu)
    return D, M, fH, fM, fD, s2


def ucb_reject(Z, zmax, c, delta, n_plan):
    """One-sided betting test of H0: mean(Z) >= c. Z: (T, R) array, zmax: (T,) or (T,R) upper bounds, c: (R,).
    Returns boolean (R,) - True if H0 rejected at some time (=> mean < c)."""
    T, R = Z.shape
    logK = np.zeros(R)
    rej = np.zeros(R, bool)
    mu, ss, cnt = np.zeros(R), np.zeros(R), 0
    for t in range(T):
        var = (ss + 0.25) / (cnt + 1)                   # predictable variance estimate
        beta = np.sqrt(2 * np.log(1 / delta) / (n_plan * var))
        cap = 0.5 / np.maximum(zmax[t] - c, 1e-12)
        beta = np.minimum(beta, cap)
        logK += np.log1p(beta * (c - Z[t]))
        rej |= logK >= np.log(1 / delta)
        cnt += 1
        d = Z[t] - mu
        mu += d / cnt
        ss += d * (Z[t] - mu)
    return rej


def betting_lcb(D, M, fH, fM, fD, s2, n_budget, delta, rng, gamma=0.3, grid=np.linspace(0, 1, 201)):
    N = len(D)
    sig = np.sqrt(s2 + 0.05 * fD + 1e-3)
    q = (1 - gamma) * sig / sig.sum() + gamma / N
    seen, draws = set(), []
    while len(seen) < n_budget:
        i = rng.choice(N, p=q)
        draws.append(i)
        seen.add(i)
    idx = np.array(draws)
    fMhat = fD - fH  # predicted misses consistent with fD
    # y_i(r) = M_i - (1-r) D_i ; prediction f_i(r) = fMhat_i - (1-r) fD_i ; normalised by N
    r = grid[None, :]
    f_all = (fMhat[:, None] - (1 - r) * fD[:, None]).sum(0) / N
    y = M[idx][:, None] - (1 - r) * D[idx][:, None]
    f = fMhat[idx][:, None] - (1 - r) * fD[idx][:, None]
    Z = f_all[None, :] + (y - f) / (N * q[idx][:, None])
    zmax = f_all[None, :] + (r * DMAX - f) / (N * q[idx][:, None])
    c = np.zeros(len(grid))   # H(r): rho <= r  <=>  mean_i(M - (1-r) D) >= 0
    rej = ucb_reject(Z, zmax, c, delta, n_plan=len(idx))
    # LCB = largest grid point such that all points below are rejected
    nonrej = np.where(~rej)[0]
    if len(nonrej) == 0:
        return 1.0, len(idx)
    j = nonrej[0]
    return (grid[j - 1] if j > 0 else 0.0), len(idx)


def main():
    rng = np.random.default_rng(1)
    N, delta, reps = 2000, 0.05, 300
    for prev in (0.03, 0.10, 0.30):
        for n in (30, 100):
            cov = {"ppi_l1": [], "classical": [], "betting": []}
            lcb = {"ppi_l1": [], "classical": [], "betting": []}
            nan_cl = 0
            rhos = []
            for _ in range(reps):
                D, M, fH, fM, fD, s2 = make_pool(N, prev, rng)
                rho = 1 - M.sum() / D.sum()
                rhos.append(rho)
                s = rng.choice(N, n, replace=False)
                rest = np.setdiff1d(np.arange(N), s)
                H = D - M
                e, lo, hi = ppi.estimate_ratio(H[s], D[s], fH[s], fD[s], fH[rest], fD[rest], alpha=2 * delta)
                lo = max(0.0, lo) if lo == lo else 0.0
                cov["ppi_l1"].append(lo <= rho)
                lcb["ppi_l1"].append(lo)
                e, lo, hi = ppi.classical_ratio(H[s], D[s], alpha=2 * delta, n_total=N)
                if lo != lo:
                    nan_cl += 1
                    lo = 0.0
                lo = max(0.0, lo)
                cov["classical"].append(lo <= rho)
                lcb["classical"].append(lo)
                b, _ = betting_lcb(D, M, fH, fM, fD, s2, n, delta, rng)
                cov["betting"].append(b <= rho)
                lcb["betting"].append(b)
            print(f"prev={prev:.2f} n={n:3d} true_rho~{np.mean(rhos):.3f} | " + " | ".join(
                f"{k}: cov={np.mean(cov[k]):.3f} meanLCB={np.mean(lcb[k]):.3f}" for k in cov)
                + f" | classical NaN={nan_cl / reps:.2f}", flush=True)


if __name__ == "__main__":
    main()
