"""Quick coverage check: current PPI ratio CI (delta method + t) vs. design-based betting/Fieller LCB.

Semi-synthetic pool of N images with rare defects; per-image self-assessment from candidate regions.
Target: pool-level defect recall rho = 1 - sum(M)/sum(D) (no review, no corrections) -> one-sided 95% LCB.

v2 (07_red_team.md, F4 / change 6): the bet cap uses a PREDICTABLE upper bound
    z^max_t = lambda_t * Fbar^(t) + max_{i in U'} (b - lambda_t f_i^(t)) / (N' q_t(i)),
computed from F_{t-1} (here q, f, lambda=1 are frozen before the first draw, so z^max is a constant per
grid point). v1 evaluated the bound at the CURRENT draw i_t, which makes beta_t depend on i_t and breaks
the supermartingale property (type-I error 0.63-0.99 on an adversarial pool). Also new in v2:
  * runtime check Z_t <= z^max_t; a violation invalidates the certificate (LCB := 0), it is never skipped;
  * the bet size is tuned with the planned budget n (fixed in advance), not with the realised number of
    draws (which depends on future draws and is therefore not predictable);
  * an adversarial scenario (all misses in units with q at the floor gamma/N) compares both caps;
  * the floor q_i >= gamma/N is checked with autolabel.certify.check_floor (raises if violated).
A reduced version of the adversarial scenario runs as a unit test: tests/test_sims_adversarial.py.
The hard bound b = r * D_MAX is a model assumption (07 F1); the pools below satisfy it by construction.

Run:  python3 docs/research/sims/sim_recall_cert.py     (~2-3 min on a CPU)
"""
import sys
import numpy as np

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", ".."))
from autolabel import ppi  # noqa: E402
from autolabel.certify import check_floor  # noqa: E402

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


def predictable_zmax(fMhat, fD, f_all, q, grid):
    """Predictable bet bound per grid point r (07 F4): lambda=1, Fbar = f_all, b = r * D_MAX,
    z^max(r) = f_all(r) + max_i (b(r) - f_i(r)) / (N q_i), maximum over ALL units of the pool.
    q and f are frozen before the first draw, so this is F_0-measurable."""
    N = len(q)
    r = grid[None, :]
    f_pool = fMhat[:, None] - (1 - r) * fD[:, None]                     # (N, R)
    return f_all + ((r * DMAX - f_pool) / (N * q[:, None])).max(0)      # (R,)


def betting_lcb(D, M, fH, fM, fD, s2, n_budget, delta, rng, gamma=0.3, grid=np.linspace(0, 1, 201),
                cap="predictable", q=None):
    """Betting-Fieller LCB for rho from PPS draws with replacement (Hansen-Hurwitz / IPW terms).

    cap='predictable' (valid, v2) or 'per_draw' (v1, INVALID: bound evaluated at the current draw; kept
    only to reproduce 07 F4). Returns (lcb, number_of_draws, bound_violated)."""
    N = len(D)
    if q is None:
        sig = np.sqrt(s2 + 0.05 * fD + 1e-3)
        q = (1 - gamma) * sig / sig.sum() + gamma / N
    check_floor(q, gamma)     # S3 validity condition (ii), 07 section 7 no. 7: q_i >= gamma / N
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
    if cap == "predictable":
        zmax = np.broadcast_to(predictable_zmax(fMhat, fD, f_all, q, grid), Z.shape)
    elif cap == "per_draw":   # v1, not F_{t-1}-measurable (07 F4) - for the comparison only
        zmax = f_all[None, :] + (r * DMAX - f) / (N * q[idx][:, None])
    else:
        raise ValueError(cap)
    # Runtime check Z_t <= z^max_t: a violation means the registered bound b = r*D_MAX is wrong for a
    # drawn unit -> the certificate is invalid (no LCB), it is not skipped.
    violated = bool(np.any(Z > zmax + 1e-9))
    if violated:
        return 0.0, len(idx), True
    c = np.zeros(len(grid))   # H(r): rho <= r  <=>  mean_i(M - (1-r) D) >= 0
    rej = ucb_reject(Z, zmax, c, delta, n_plan=n_budget)   # n_budget is fixed in advance (predictable)
    # Fixed-sequence scan from below: LCB = largest grid point such that all points up to it are rejected
    nonrej = np.where(~rej)[0]
    if len(nonrej) == 0:
        return 1.0, len(idx), False
    j = nonrej[0]
    return (grid[j - 1] if j > 0 else 0.0), len(idx), False


def make_adversarial_pool(N=2000, gamma=0.3):
    """07 F4 / 04 §4.10: 160 detected single defects in images where the self-assessment is confident
    (high sigma), 40 missed single defects in images WITHOUT any signal (q at the floor), 1800 clean
    images. True recall 0.8; the self-assessment predicts no misses at all."""
    D = np.zeros(N)
    M = np.zeros(N)
    D[:200] = 1
    M[160:200] = 1
    fH = np.zeros(N)
    fH[:160] = 1.0
    fD = fH.copy()                       # predicts zero misses
    fM = np.zeros(N)
    s2 = np.zeros(N)
    sig = np.zeros(N)
    sig[:160] = 1.0
    q = (1 - gamma) * sig / sig.sum() + gamma / N
    return D, M, fH, fM, fD, s2, q


def adversarial(rng, delta=0.05, reps=100):
    D, M, fH, fM, fD, s2, q = make_adversarial_pool()
    rho = 1 - M.sum() / D.sum()
    print(f"Adversarial pool (07 F4): true rho = {rho:.3f}, misses at floor q = {q.min():.2e}, "
          f"{reps} reps; LCB > rho is a false certificate", flush=True)
    for n in (300, 1000):
        for cap in ("per_draw", "predictable"):
            res = [betting_lcb(D, M, fH, fM, fD, s2, n, delta, rng, q=q, cap=cap) for _ in range(reps)]
            lcb = np.array([a for a, _, _ in res])
            viol = np.mean([v for _, _, v in res])
            print(f"  n={n:5d} distinct units, cap={cap:11s}: coverage P(LCB <= rho) = "
                  f"{np.mean(lcb <= rho):.3f}, mean LCB = {lcb.mean():.3f}, "
                  f"mean draws = {np.mean([t for _, t, _ in res]):.0f}, bound violations = {viol:.3f}",
                  flush=True)


def main():
    rng = np.random.default_rng(1)
    N, delta, reps = 2000, 0.05, 300
    print(f"S1 (04 §3): N={N}, one-sided {1 - delta:.0%} LCB, {reps} reps; betting = PPS gamma=0.3, "
          f"predictable cap (v2)", flush=True)
    for prev in (0.03, 0.10, 0.30):
        for n in (30, 100):
            cov = {"ppi_l1": [], "classical": [], "betting": []}
            lcb = {"ppi_l1": [], "classical": [], "betting": []}
            nan_cl = 0
            viol = 0
            rhos = []
            for _ in range(reps):
                D, M, fH, fM, fD, s2 = make_pool(N, prev, rng)
                rho = 1 - M.sum() / D.sum()
                rhos.append(rho)
                s = rng.choice(N, n, replace=False)
                rest = np.setdiff1d(np.arange(N), s)
                H = D - M
                e, lo, hi = ppi.estimate_ratio(H[s], D[s], fH[s], fD[s], fH[rest], fD[rest], alpha=2 * delta)
                lo = max(0.0, lo)          # undefined -> trivial interval [0, 1] (ppi v2)
                cov["ppi_l1"].append(lo <= rho)
                lcb["ppi_l1"].append(lo)
                e, lo, hi = ppi.classical_ratio(H[s], D[s], alpha=2 * delta, n_total=N)
                if e != e:                 # no defect in the audit: trivial interval, counted as LCB = 0
                    nan_cl += 1
                lo = max(0.0, lo)
                cov["classical"].append(lo <= rho)
                lcb["classical"].append(lo)
                b, _, v = betting_lcb(D, M, fH, fM, fD, s2, n, delta, rng)
                viol += v
                cov["betting"].append(b <= rho)
                lcb["betting"].append(b)
            print(f"prev={prev:.2f} n={n:3d} true_rho~{np.mean(rhos):.3f} | " + " | ".join(
                f"{k}: cov={np.mean(cov[k]):.3f} meanLCB={np.mean(lcb[k]):.3f}" for k in cov)
                + f" | classical undefined={nan_cl / reps:.2f} | betting bound violations={viol}", flush=True)


if __name__ == "__main__":
    main()
    adversarial(np.random.default_rng(2))
