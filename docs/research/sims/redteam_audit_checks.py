"""Red-team checks of the design-based audit results in 04 that the CertAL proposal (00) relies on.

  (a) Prop. 3: closed-form inclusion probabilities vs. a numerical solver.
  (b) Exact hypergeometric UCB for zeta under optional stopping (00 lists it inside an "anytime
      stoppable" audit) vs. a time-uniform betting bound.
  (c) Tiles vs. images: betting-Fieller recall LCB at the *same inspected area* (D_max effect only,
      no predictor), plus the haystack bound, which does not depend on unit size.
  (d) Satz 2 when the review score r is re-learned from the same audit (allowed for q, f, lambda
      in Satz 1, but not for r): false certification.
  (e) Human != gold: verify-first reviewer who overlooks 30 % of missed defects.

Run:  python docs/research/sims/redteam_audit_checks.py       (~1-2 min on a laptop CPU)
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import hypergeom

RNG = np.random.default_rng(7)
DELTA = 0.05


# ----------------------------------------------------------------------------------------- (a)
def prop3_closed(m, s, kappa, B):
    def pis(nu):
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(m + kappa * s ** 2 >= nu, 1.0, s * np.sqrt(kappa / np.maximum(nu - m, 1e-300)))
        return np.minimum(p, 1.0)
    lo, hi = m.min(), m.max() + kappa * s.max() ** 2 + 1e6
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pis(mid).sum() > B else (lo, mid)
    return pis((lo + hi) / 2)


def prop3_check():
    gaps, worst_dpi = [], 0.0
    for _ in range(20):
        n = 60
        m = RNG.gamma(0.5, 0.3, n)
        s = RNG.gamma(1.0, 0.5, n) + 0.02
        kappa, B = RNG.uniform(0.2, 5), RNG.uniform(5, 30)
        obj = lambda p: np.sum((1 - p) * m) + kappa * np.sum((1 / p - 1) * s ** 2)
        pc = prop3_closed(m, s, kappa, B)
        res = minimize(obj, np.full(n, B / n), method="SLSQP", bounds=[(1e-4, 1)] * n,
                       constraints=[{"type": "eq", "fun": lambda p: p.sum() - B}],
                       options={"maxiter": 500, "ftol": 1e-12})
        gaps.append((obj(pc) - res.fun) / abs(res.fun))
        worst_dpi = max(worst_dpi, np.abs(pc - res.x).max())
    print(f"(a) Prop. 3: closed form vs SLSQP over 20 random instances: rel. objective gap in "
          f"[{min(gaps):+.1e}, {max(gaps):+.1e}] (negative = closed form better), max |pi diff| {worst_dpi:.3f}")


# ----------------------------------------------------------------------------------------- (b)
_UCB_CACHE = {}


def hyper_ucb_count(k, n, N):
    """Exact one-sided (1-DELTA) UCB of the number of bad units, fixed-n SRS without replacement."""
    key = (k, n, N)
    if key not in _UCB_CACHE:
        Ks = np.arange(k, N + 1)
        _UCB_CACHE[key] = Ks[hypergeom.cdf(k, N, Ks, n) > DELTA].max()
    return _UCB_CACHE[key]


def betting_reject_mean_ge(x, m, nb=20):
    """Time-uniform test of H: E x >= m for x in {0,1} (with replacement). Returns first rejection index."""
    betas = np.linspace(0.0, 0.99 / max(1 - m, 1e-9), nb + 1)[1:]
    logk = np.cumsum(np.log1p(betas[None, :] * (m - x[:, None])), axis=0)
    k = np.exp(logk).mean(1)
    hit = np.nonzero(k >= 1 / DELTA)[0]
    return hit[0] if len(hit) else None


def optional_stopping():
    N, target, reps = 2000, 0.02, 2000
    looks = np.arange(20, 801, 20)
    for K, label in ((41, "true zeta 0.0205 > target 0.02 (any certificate is false)"),
                     (10, "true zeta 0.005 (certificate is correct)")):
        pop = np.zeros(N, int)
        pop[:K] = 1
        cert_fixed, cert_monitor, stop_h, stop_b, cert_bet = 0, 0, [], [], 0
        for _ in range(reps):
            perm = RNG.permutation(pop)             # sampling without replacement
            cum = np.cumsum(perm)
            if hyper_ucb_count(cum[599], 600, N) / N <= target:
                cert_fixed += 1
            for n in looks:
                if hyper_ucb_count(cum[n - 1], n, N) / N <= target:
                    cert_monitor += 1
                    stop_h.append(n)
                    break
            xr = pop[RNG.integers(0, N, 800)]       # with replacement for the betting CS
            j = betting_reject_mean_ge(xr, target)
            if j is not None:
                cert_bet += 1
                stop_b.append(j + 1)
        print(f"(b) {label}: fixed n=600 hypergeom certifies {cert_fixed / reps:.3f}; hypergeom "
              f"monitored every 20 draws certifies {cert_monitor / reps:.3f}"
              + (f" (median stop n={np.median(stop_h):.0f})" if stop_h else "")
              + f"; betting CS certifies {cert_bet / reps:.3f}"
              + (f" (median stop n={np.median(stop_b):.0f})" if stop_b else ""))


# ----------------------------------------------------------------------------------------- (c)
def make_pool(N_img=2000, T=16, prev=0.1, p_miss=0.1, cluster=0.5, rng=RNG):
    D_img = np.where(rng.random(N_img) < prev, 1 + rng.poisson(0.6, N_img), 0)
    D_img = np.minimum(D_img, 6)
    Dt = np.zeros((N_img, T), int)
    Mt = np.zeros((N_img, T), int)
    for i in np.nonzero(D_img)[0]:
        home = rng.integers(T)
        for _ in range(D_img[i]):                       # defects cluster around a home tile
            t = home if rng.random() < cluster else rng.integers(T)
            Dt[i, t] += 1
            Mt[i, t] += rng.random() < p_miss
    return Dt, Mt


def fieller_lcb(Mu, Du, n, dmax, rng, grid=np.linspace(0, 1, 101), nb=24, reps=200):
    """Uniform with-replacement draws of n units; LCB of pooled recall via betting-Fieller (Prop. 4)."""
    N = len(Mu)
    r = grid[None, None, :]
    zmax = np.maximum(grid * dmax, 1e-9)                # g = M-(1-r)D <= r*M <= r*dmax
    betas = (0.95 / zmax)[None, :] * np.logspace(-3, 0, nb)[:, None]      # (nb, G)
    idx = rng.integers(0, N, (reps, n))
    logk = np.zeros((reps, nb, len(grid)))
    rej = np.zeros((reps, len(grid)), bool)
    for t in range(n):
        g = Mu[idx[:, t]][:, None, None] - (1 - r) * Du[idx[:, t]][:, None, None]   # (reps,1,G)
        logk += np.log1p(betas[None] * (0.0 - g))
        mx = logk.max(1)
        rej |= mx + np.log(np.exp(logk - mx[:, None, :]).mean(1)) >= np.log(1 / DELTA)
    lcb = []
    for row in rej:
        nr = np.nonzero(~row)[0]
        j = nr[0] if len(nr) else len(grid)
        lcb.append(grid[j - 1] if j > 0 else 0.0)
    return np.mean(lcb), np.mean(np.array(lcb) <= 1 - Mu.sum() / Du.sum())


def tiles_vs_images():
    for cluster, label in ((0.0, "defects spread uniformly over tiles"), (0.5, "defects clustered (50% in a home tile)")):
        Dt, Mt = make_pool(cluster=cluster)
        rho = 1 - Mt.sum() / Dt.sum()
        Di, Mi = Dt.sum(1), Mt.sum(1)
        Dtile, Mtile = Dt.ravel(), Mt.ravel()
        print(f"(c) {label}: 2000 images x 16 tiles, {Dt.sum()} defects, true recall {rho:.3f}; "
              f"max defects per image {Di.max()}, per tile {Dtile.max()} "
              f"(defect tiles with >2 defects: {(Dtile > 2).sum() / max((Dtile > 0).sum(), 1):.3f})")
        for frac in (0.1, 0.3):
            n_img = int(frac * 2000)
            l_img, c_img = fieller_lcb(Mi, Di, n_img, dmax=6, rng=RNG)
            l_til, c_til = fieller_lcb(Mtile, Dtile, n_img * 16, dmax=Dtile.max(), rng=RNG, reps=100)
            print(f"    inspected area {frac:.0%}: mean LCB images (D_max=6) {l_img:.3f} [cov {c_img:.2f}] | "
                  f"tiles (D_max={Dtile.max()}) {l_til:.3f} [cov {c_til:.2f}]")
    k = 10
    print(f"    haystack bound (k={k} misses hidden in units without signal, delta={DELTA}): required "
          f"inspected share = 1-delta^(1/k) = {1 - DELTA ** (1 / k):.3f} for images AND for tiles")


# ----------------------------------------------------------------------------------------- (d)
def satz2_learned_r():
    """Population U': 2000 units, 400 with one defect, 80 of them missed (rho = 0.8). Target eps = 0.1
    (rho_final >= 0.9). Hypothesis tested at the 'review nothing more' threshold u = u_top."""
    N, eps, reps, n = 2000, 0.1, 1000, 400
    D = np.zeros(N)
    D[:400] = 1
    M = np.zeros(N)
    M[:80] = 1
    out = {}
    for mode in ("r frozen before audit", "r re-learned from audit"):
        cert, still_bad = 0, []
        for _ in range(reps):
            idx = RNG.integers(0, N, n)
            seen = np.unique(idx)
            if mode == "r frozen before audit":
                y = M[idx] - eps * D[idx]                       # nothing reviewed at u_top
                A_true = M.sum() - eps * D.sum()
            else:                                              # audited misses get r = +inf -> 'reviewed'
                known_bad = np.zeros(N, bool)
                known_bad[seen] = M[seen] > 0
                y = M[idx] * (~known_bad[idx]) - eps * D[idx]
                A_true = (M * ~known_bad).sum() - eps * D.sum()
            # test H: mean(y) > 0 (=A>0) ; y in [-eps, 1]; reject => certify rho_final >= 1-eps
            x = (y + eps) / (1 + eps)
            m0 = eps / (1 + eps)
            j = betting_reject_mean_ge(x, m0)
            if j is not None:
                cert += 1
                still_bad.append(A_true > 0)
        out[mode] = (cert / reps, np.mean(still_bad) if still_bad else float("nan"))
        print(f"(d) Satz 2, {mode}: certifies in {out[mode][0]:.2f} of runs; among certified runs the "
              f"claim rho_final >= 0.9 is false in {out[mode][1]:.2f}")


# ----------------------------------------------------------------------------------------- (e)
def human_not_gold():
    N, K, reps, tgt = 2000, 60, 2000, 0.025      # true zeta = 0.03 > target: every certificate is false
    for sens, label in ((1.0, "gold reviewer"), (0.7, "verify-first reviewer, sensitivity 0.7")):
        for n in (400, 1000):
            cert = 0
            for _ in range(reps):
                pop = np.zeros(N, int)
                pop[:K] = 1
                s = RNG.choice(N, n, replace=False)
                obs = pop[s] * (RNG.random(n) < sens)
                if hyper_ucb_count(int(obs.sum()), n, N) / N <= tgt:
                    cert += 1
            print(f"(e) {label}, n={n}: certifies zeta <= {tgt} (FALSE, true zeta 0.03) in "
                  f"{cert / reps:.3f} of runs (nominal <= {DELTA})")


if __name__ == "__main__":
    prop3_check()
    optional_stopping()
    tiles_vs_images()
    satz2_learned_r()
    human_not_gold()
