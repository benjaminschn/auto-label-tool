"""CertAL-lite vs CertAL-full for the image-level miss rate zeta (human cost = distinct inspected images).

Pool: N images, prevalence of defective images p_def; a defective image is 'bad' (>= 1 missed defect after
auto-labeling) with prob p_bad. Risk score r (higher = review first):
  clean           r ~ N(0, 1)
  defective, ok   r ~ N(mu_ok, 1)
  bad, visible    r ~ N(mu_bad, 1)
  bad, invisible  r ~ N(0, 1)       (share v of bad images; no signal at all)
Oracle calibration m(r) = P(bad | r) is given to the FULL design (favours FULL).

Designs (all: review top-k by r first = take-all stratum, corrections in review and audit are credited):
  LITE-SRS : SRSWOR of fixed n from U' = U \\ top-k; exact hypergeometric UCB of #bad in U'; certify if
             UCB - found <= K*.  n planned for 90 % power (exact, averaged over pools).
  LITE-GS  : same, but group-sequential with 4 looks, delta/4 each (valid by Bonferroni).
  FULL     : WR draws q = (1-gamma) sigma/sum sigma + gamma/N' (sigma = sqrt(m(1-m))), PPI term with
             f = m (oracle), lambda = 1, betting mixture CS (predictable cap), stop when certified.
  FULL-U   : uniform WR draws, no predictor (betting CS only) -> isolates the cost of anytime validity.
Cost = k + distinct audited images. Report: cost at which >= 90 % of runs certify (and median).
"""
import numpy as np
from scipy.stats import hypergeom, norm

RNG = np.random.default_rng(20261019)
DELTA = 0.05


def make_pool(N, p_def, p_bad, v, mu_ok, mu_bad, rng):
    defect = rng.random(N) < p_def
    bad = defect & (rng.random(N) < p_bad)
    inv = bad & (rng.random(N) < v)
    mu = np.where(bad & ~inv, mu_bad, np.where(defect & ~bad, mu_ok, 0.0))
    r = mu + rng.normal(size=N)
    # oracle P(bad | r) under the generative model
    pc = (1 - p_def) * norm.pdf(r)
    pok = p_def * (1 - p_bad) * norm.pdf(r - mu_ok)
    pbv = p_def * p_bad * (1 - v) * norm.pdf(r - mu_bad)
    pbi = p_def * p_bad * v * norm.pdf(r)
    m = (pbv + pbi) / (pc + pok + pbv + pbi)
    return bad.astype(int), r, m


_UCB = {}


def ucb_count(k, n, Np, delta):
    key = (k, n, Np, delta)
    if key not in _UCB:
        Ks = np.arange(k, Np + 1)
        ok = Ks[hypergeom.cdf(k, Np, Ks, n) > delta]
        _UCB[key] = ok.max()
    return _UCB[key]


def lite_srs_power(Kp, Np, n, Kstar, delta=DELTA):
    """Exact P(certify) for SRSWOR n from N' with Kp bad: UCB(X) - X <= K*."""
    xs = np.arange(0, min(n, Kp) + 1)
    pm = hypergeom.pmf(xs, Np, Kp, n)
    ok = np.array([ucb_count(int(x), n, Np, delta) - x <= Kstar for x in xs])
    return float((pm * ok).sum())


def lite_gs(bad_up, looks, Kstar, reps, rng):
    Np = len(bad_up)
    d = DELTA / len(looks)
    costs = []
    for _ in range(reps):
        cum = np.cumsum(rng.permutation(bad_up)[:looks[-1]])
        c = np.inf
        for n in looks:
            x = int(cum[n - 1])
            if ucb_count(x, n, Np, d) - x <= Kstar:
                c = n
                break
        costs.append(c)
    return np.array(costs)


def full_betting(bad_up, m_up, Kstar, reps, rng, gamma=0.3, lam=1.0, uniform=False, T=2500, nb=16):
    """Sequential betting UCB for K' = sum bad over U' (Satz 1, PPI + IPW, predictable cap)."""
    Np = len(bad_up)
    if uniform:
        q = np.full(Np, 1.0 / Np)
        f = np.zeros(Np)
        lam = 0.0
    else:
        s = np.sqrt(m_up * (1 - m_up)) + 1e-12
        q = (1 - gamma) * s / s.sum() + gamma / Np
        f = m_up
    Fbar = f.mean()
    zmax = lam * Fbar + np.max((1 - lam * f) / (Np * q))
    grid = np.linspace(0, min(1.0, (Kstar + 80) / Np), 121)       # candidate values of mu = K'/N'
    betas = np.logspace(-3, 0, nb)[:, None] * (0.95 / np.maximum(zmax - grid, 1e-9))[None, :]
    idx = rng.choice(Np, size=(reps, T), p=q)
    y = bad_up[idx]
    Z = lam * Fbar + (y - lam * f[idx]) / (Np * q[idx])
    logk = np.zeros((reps, nb, len(grid)))
    rej = np.zeros((reps, len(grid)), bool)
    seen = np.zeros((reps, Np), bool)
    found = np.zeros(reps)
    ndist = np.zeros(reps)
    cost = np.full(reps, np.inf)
    done = np.zeros(reps, bool)
    for t in range(T):
        it = idx[:, t]
        new = ~seen[np.arange(reps), it]
        seen[np.arange(reps), it] = True
        ndist += new
        found += new * y[:, t]
        logk += np.log1p(betas[None] * (grid[None, None, :] - Z[:, t][:, None, None]))
        mx = logk.max(1)
        rej |= (mx + np.log(np.exp(logk - mx[:, None, :]).mean(1))) >= np.log(1 / DELTA)
        # UCB of mu: smallest grid value such that all larger values are rejected
        nr = np.where(~rej, np.arange(len(grid))[None, :], -1).max(1)
        ucb_mu = grid[np.minimum(nr + 1, len(grid) - 1)]
        okc = (Np * ucb_mu - found <= Kstar) & ~done
        cost[okc] = ndist[okc]
        done |= okc
        if done.all():
            break
    return cost


def q90(c):
    return np.quantile(c, 0.9) if np.isfinite(np.quantile(c, 0.9)) else np.inf


def scenario(name, N=2000, p_def=0.10, p_bad=0.25, v=0.3, mu_ok=1.5, mu_bad=2.5, Kstar=40,
             ks=(0, 50, 100, 150, 200, 300), n_pools=6, reps=200):
    print(f"\n=== {name}: N={N}, p_def={p_def}, p_bad={p_bad}, invisible share v={v}, mu_bad={mu_bad}, "
          f"target zeta* = {Kstar / N:.3f} (K*={Kstar})")
    pools = [make_pool(N, p_def, p_bad, v, mu_ok, mu_bad, RNG) for _ in range(n_pools)]
    print("    mean #bad = %.1f, AUC(r for bad) = %.3f" % (
        np.mean([b.sum() for b, _, _ in pools]),
        np.mean([(np.argsort(np.argsort(r))[b == 1].mean() - (b.sum() - 1) / 2) / (len(b) - b.sum())
                 for b, r, _ in pools])))
    best = {}
    for k in ks:
        rowp, gs, fu, fuu = [], [], [], []
        Kps = []
        for b, r, m in pools:
            order = np.argsort(-r)
            up = order[k:]
            Kp, Np = int(b[up].sum()), len(up)
            Kps.append(Kp)
            rowp.append((Kp, Np))
        # LITE-SRS: smallest n with mean power >= 0.9
        n_srs = np.inf
        for n in list(range(20, 400, 10)) + list(range(400, 1801, 25)):
            pw = np.mean([lite_srs_power(Kp, Np, n, Kstar) for Kp, Np in rowp])
            if pw >= 0.9:
                n_srs = n
                break
        for b, r, m in pools:
            order = np.argsort(-r)
            up = order[k:]
            gs.append(lite_gs(b[up], looks=(150, 300, 450, 600, 800), Kstar=Kstar, reps=reps // 2, rng=RNG))
            fu.append(full_betting(b[up], m[up], Kstar, reps // 4, RNG))
            fuu.append(full_betting(b[up], m[up], Kstar, reps // 4, RNG, uniform=True))
        gs, fu, fuu = (np.concatenate(x) for x in (gs, fu, fuu))
        res = {"LITE-SRS (fixed n, 90% power)": k + n_srs,
               "LITE-GS q90": k + q90(gs), "LITE-GS median": k + np.median(gs),
               "FULL q90": k + q90(fu), "FULL median": k + np.median(fu),
               "FULL-U q90": k + q90(fuu), "FULL-U median": k + np.median(fuu)}
        print(f"  k={k:4d} (residual bad in U' mean {np.mean(Kps):5.1f}): " + " | ".join(
            f"{a} {v_:.0f}" for a, v_ in res.items()), flush=True)
        for a, v_ in res.items():
            best[a] = min(best.get(a, np.inf), v_)
    print("  BEST over k: " + " | ".join(f"{a} {v_:.0f}" for a, v_ in best.items()))


if __name__ == "__main__":
    scenario("good risk score, zeta*=0.02")
    scenario("weak risk score (v=0.5, mu_bad=1.5), zeta*=0.02", v=0.5, mu_bad=1.5)
    scenario("good risk score, zeta*=0.015", Kstar=30)
