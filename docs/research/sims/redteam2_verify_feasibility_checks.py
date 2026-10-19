"""Adversarial verification (feasibility lens) of red-team-2 findings A1-A10 on 00 / 04 / 05.

Only re-checks numbers the findings rely on; earlier scripts are reused where possible.

  P1 (A3)  zeta certification cost: uniform WR betting vs risk-proportional (PPS) WR betting with IPW,
           lambda = 0 and lambda = 1, AUC 0.7 / 0.9, gamma 0.3 / 0.1, true zeta far from AND near the
           target (the finding only tested 0.005). Reference: exact hypergeometric fixed-n (WOR).
  P2 (A1)  (i) sensitivity correction with seeded defects (s_real vs s_seed);
           (ii) two-phase design (verify-first reviewer + gold expert re-check of r units) vs a
           gold-only audit: expert units needed for the same power.
  P3 (A5)  delivered-pool false-certificate rate over R_max retrain rounds, per-round delta vs delta/R.
  P4 (A4)  haystack shares (exact hypergeometric), incl. the packed worst case (D_max), zeta* numbers.
  P5 (A6)  Phase-1 compute arithmetic with the real dataset sizes (remote-CPU speeds from HANDOFF).

Run:  python docs/research/sims/redteam2_verify_feasibility_checks.py      (CPU, ~2 min)
"""
import math

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import hypergeom, norm

RNG = np.random.default_rng(2026)
DELTA = 0.05


def hyper_ucb_count(k, n, N, delta=DELTA, _cache={}):
    key = (k, n, N, delta)
    if key not in _cache:
        Ks = np.arange(k, N + 1)
        _cache[key] = Ks[hypergeom.cdf(k, N, Ks, n) > delta].max()
    return _cache[key]


# ------------------------------------------------------------------------------------------- P1
def first_rejection(Z, zmax, m, grid="lin", nb=20):
    """Mixture-over-beta test of H: E Z >= m (Satz 1, predictable cap). Returns first index or None.
    grid 'lin': beta in linspace(0, cap] (as redteam_audit_checks); 'log': cap * logspace(-3, 0)."""
    cap = 0.99 / (zmax - m)
    betas = np.linspace(0, cap, nb + 1)[1:] if grid == "lin" else cap * np.logspace(-3, 0, 30)
    logk = np.cumsum(np.log1p(betas[None, :] * (m - Z[:, None])), axis=0)
    mx = logk.max(1)
    lm = mx + np.log(np.exp(logk - mx[:, None]).mean(1))
    hit = np.nonzero(lm >= np.log(1 / DELTA))[0]
    return hit[0] if len(hit) else None


def p1_zeta_efficiency(reps=150, T=4000, grid="lin"):
    N, target = 2000, 0.02
    print(f"P1 (A3) zeta certification, N'={N}, target {target}, WR draws up to {T}, {reps} reps. "
          f"beta grid {grid}. cert/med = share certified / median DISTINCT units at certification")
    for zeta in (0.005, 0.010, 0.015):
        K = int(round(zeta * N))
        # exact fixed-n WOR reference: smallest n with power >= 0.5 / 0.9
        nref = {}
        for n in range(20, N + 1, 10):
            kc = -1
            for k in range(0, n + 1):
                if hyper_ucb_count(k, n, N) / N <= target:
                    kc = k
                else:
                    break
            pw = hypergeom.cdf(kc, N, K, n) if kc >= 0 else 0.0
            for lev in (0.5, 0.9):
                if lev not in nref and pw >= lev:
                    nref[lev] = n
        designs = [("uniform", None, None, 0)]
        for auc in (0.7, 0.9):
            for gam in (0.3, 0.1):
                designs.append((f"PPS AUC{auc} g{gam}", auc, gam, 0))
        designs.append(("PPS AUC0.9 g0.3 PPI l=1", 0.9, 0.3, 1))
        out = []
        for name, auc, gam, lam in designs:
            cert, med = 0, []
            for _ in range(reps):
                y = np.zeros(N)
                y[:K] = 1
                if auc is None:
                    idx = RNG.integers(0, N, T)
                    Z, zmax = y[idx], 1.0
                else:
                    d = math.sqrt(2) * norm.ppf(auc)
                    s = RNG.normal(d * y, 1.0)
                    lr = np.exp(d * s - d * d / 2)                       # binormal likelihood ratio
                    f = zeta * lr / (zeta * lr + 1 - zeta)               # oracle-calibrated P(bad | s)
                    q = (1 - gam) * f / f.sum() + gam / N
                    idx = RNG.choice(N, T, p=q / q.sum())
                    if lam == 0:
                        Z, zmax = y[idx] / (N * q[idx]), 1.0 / (N * q.min())
                    else:
                        Fb = f.mean()
                        Z = Fb + (y[idx] - f[idx]) / (N * q[idx])
                        zmax = Fb + ((1 - f) / (N * q)).max()
                j = first_rejection(Z, zmax, target, grid)
                if j is not None:
                    cert += 1
                    med.append(len(np.unique(idx[: j + 1])))
            out.append(f"{name} {cert / reps:.2f}/{np.median(med):.0f}" if med else f"{name} 0.00/-")
        print(f"  true zeta {zeta:.3f}: exact fixed-n WOR needs n={nref.get(0.5)} (power 0.5), "
              f"n={nref.get(0.9)} (power 0.9)")
        print("    " + " | ".join(out))


# ------------------------------------------------------------------------------------------- P2
def p2_human_not_gold(reps=2000):
    N, target = 2000, 0.025
    print("P2 (A1) (i) sensitivity correction UCB_hyp(obs; d/2) / LCB_CP(s_seed; d/2), true zeta 0.03 "
          f"> target {target} (every certificate is false), {reps} reps")
    K = 60
    for s_real, s_seed in ((0.7, 0.7), (0.6, 0.85), (0.5, 0.9)):
        row = []
        for n in (400, 1000):
            for n_seed in (100, 300):
                cert = 0
                for _ in range(reps):
                    hits = RNG.hypergeometric(K, N - K, n)
                    obs = RNG.binomial(hits, s_real)
                    xs = RNG.binomial(n_seed, s_seed)
                    lcb_s = beta_dist.ppf(DELTA / 2, xs, n_seed - xs + 1) if xs > 0 else 0.0
                    ucb = hyper_ucb_count(obs, n, N, DELTA / 2) / N
                    if lcb_s > 0 and ucb / lcb_s <= target:
                        cert += 1
                row.append(f"n={n},seeds={n_seed}: {cert / reps:.3f}")
        print(f"    s_real {s_real} / s_seed {s_seed}: " + " | ".join(row))

    print("P2 (A1) (ii) true zeta 0.005 (10 bad units), reviewer sensitivity 0.7, target "
          f"{target}: power of two-phase (n verify-first, expert re-checks r of them) vs gold-only")
    K = 10
    for n in (600, 1000):
        row = []
        for r in (100, 200, 300, 400, 600):
            if r > n:
                continue
            cert = 0
            for _ in range(reps):
                pop = np.zeros(N, int)
                pop[:K] = 1
                det = RNG.random(N) < 0.7                     # latent: reviewer would find this bad unit
                S = RNG.choice(N, n, replace=False)
                x_found = int((pop[S] & det[S]).sum())
                Rr = S[:r]                                    # random subsample of the audited units
                x_missed = int((pop[Rr] & ~det[Rr]).sum())
                ucb = (hyper_ucb_count(x_found, n, N, DELTA / 2) + hyper_ucb_count(x_missed, r, N, DELTA / 2)) / N
                cert += ucb <= target
            row.append(f"r={r}: {cert / reps:.3f}")
        print(f"    two-phase n={n}: " + " | ".join(row))
    row = []
    for n in (120, 150, 200, 250, 300, 350, 400, 500):
        kc = max([k for k in range(0, 30) if hyper_ucb_count(k, n, N) / N <= target], default=-1)
        row.append(f"n={n}: {hypergeom.cdf(kc, N, K, n) if kc >= 0 else 0:.3f}")
    print("    gold-only fixed-n exact (delta 0.05): " + " | ".join(row))
    for zs in (0.02, 0.025):
        n0 = next(n for n in range(10, N) if hyper_ucb_count(0, n, N) / N <= zs)
        print(f"    gold-only zero-failure n for zeta* = {zs}: {n0}")


# ------------------------------------------------------------------------------------------- P3
def p3_rounds():
    N, n, Kb, target = 2000, 600, 41, 0.02
    print(f"P3 (A5) retrain loop, every version has true zeta {Kb / N} > target {target}, fresh fixed "
          f"n={n} exact audit per round, stop at first certificate")
    for d_round_name, d_fun in (("delta per round", lambda R: DELTA), ("delta/R_max", lambda R: DELTA / R)):
        row = []
        for R in (1, 2, 3, 5):
            dr = d_fun(R)
            kc = max([k for k in range(0, 40) if hyper_ucb_count(k, n, N, dr) / N <= target], default=-1)
            p = hypergeom.cdf(kc, N, Kb, n) if kc >= 0 else 0.0
            row.append(f"R_max={R}: {1 - (1 - p) ** R:.3f}")
        print(f"    {d_round_name}: P(delivered pool falsely certified) " + " | ".join(row))


# ------------------------------------------------------------------------------------------- P4
def p4_haystack():
    N = 2000
    print("P4 (A4) recall >= 0.9, misses hidden in signal-free clean units, exact WOR zero-failure, "
          "share of clean units to inspect (1 miss/unit | packed D_max=2 | packed D_max=6)")
    for p in (0.005, 0.01, 0.03, 0.10, 0.30):
        n_def = int(round(p * N))
        Nc = N - n_def
        Dtot = int(round(1.6 * n_def))
        k = int(math.floor(0.1 * Dtot)) + 1
        shares = []
        for dmax in (1, 2, 6):
            units = math.ceil(k / dmax)
            n_req = next(n for n in range(1, Nc + 1) if hypergeom.pmf(0, Nc, units, n) <= DELTA)
            shares.append(f"{n_req / Nc:.2f}")
        imp = 1 - min(40, n_def) / n_def
        print(f"    p={p:.3f}: D_tot={Dtot}, k={k}: " + " | ".join(shares)
              + f";  zeta<=0.02 implies image-level recall >= {max(imp, 0):.2f}")
    for zs in (0.02, 0.002):
        n0 = next(n for n in range(10, N) if hyper_ucb_count(0, n, N) / N <= zs)
        print(f"    zeta* = {zs}: zero-failure n = {n0} ({n0 / N:.0%} of N)")


# ------------------------------------------------------------------------------------------- P5
def p5_compute():
    s_it, it, s_inf = 0.46, 600, 0.08                 # HANDOFF: remote 4-core CPU
    sizes = {"mt": 1344, "bsdata": 1104, "deepcrack": 537, "cfd": 118}
    rest_mt = 75 - (8 * it * s_it + 8 * s_inf * 1329) / 60   # remainder of the 75-min MT run

    def run_min(ds, rounds=2, anomaly=True):
        n = sizes[ds]
        U, T = n - 15 - 0.2 * n, 0.2 * n
        members = 4 * rounds
        train = members * it * s_it / 60
        inf = members * s_inf * (U + T) / 60
        rest = rest_mt * (U / 1060) * (1.0 if anomaly else 0.3)
        return train + inf + rest

    plan = []
    plan += [run_min(d) for d in sizes for _ in range(3)]                              # A
    plan += [run_min(d, rounds=1, anomaly=False) for d in ("mt", "bsdata", "cfd") for _ in range(2)]  # B
    plan += [run_min("mt") for _ in range(3)]                                          # C
    plan += [run_min("mt")] * 3 + [run_min("bsdata")]                                  # D
    oracle = [1500 * s_it / 60 + 3] * 2                                                # E (approx.)
    tot = sum(plan) + sum(oracle)
    print(f"P5 (A6) HANDOFF Phase-1 plan: {len(plan) + len(oracle)} runs; remainder of MT run "
          f"{rest_mt:.0f} min; total at remote-CPU speed {tot / 60:.1f} h "
          f"(27 x 75 min = {27 * 75 / 60:.1f} h); speed-up the M4 needs for 24 h: {tot / 60 / 24:.2f}x")
    print(f"    full CertAL run 2.5-4.5 h vs 75 min = {150 / 75:.1f}x-{270 / 75:.1f}x")


# ------------------------------------------------------------------------------------------- P6
def p6_one_sided_target():
    """A10: the north-star example (zeta + defect recall only) is one-sided. An all-foreground mask
    under the tool's own matching rule (autolabel.metrics.image_stats, tol=2, merge_radius=2)."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    from autolabel.metrics import image_stats
    gt = np.zeros((256, 256), bool)
    gt[10:13, 20:120] = True                     # thin scratch
    gt[200:205, 200:205] = True                  # small spot
    s = image_stats(np.ones_like(gt), gt)
    clean = image_stats(np.ones_like(gt), np.zeros_like(gt))
    print(f"P6 (A10) all-foreground mask: GT components hit {s['gt_comp_hit']:.0f}/{s['gt_comp']:.0f} "
          f"(M=0 -> zeta=0, recall=1), component precision on the defective image "
          f"{s['pred_comp_hit']:.0f}/{s['pred_comp']:.0f}, Dice {s['dice']:.3f}; on a clean image "
          f"{clean['pred_comp']:.0f} false component(s)")


if __name__ == "__main__":
    p6_one_sided_target()
    p4_haystack()
    p5_compute()
    p3_rounds()
    p2_human_not_gold()
    p1_zeta_efficiency(grid="lin")
    p1_zeta_efficiency(grid="log")
