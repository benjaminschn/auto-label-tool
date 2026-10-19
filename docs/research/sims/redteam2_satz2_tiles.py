"""Second red-team pass: (1) Satz 2 of 04 - which ways of using the audit for the review rule are valid;
(2) Lemma 0 of 04 at tile level - two explicit counterexamples (fragmentation of F, tolerance hit of M).

(1) Pool U': 2000 units, 400 with one defect. Case 'false': 80 missed (rho = 0.80), target eps = 0.1
    (claim rho_final >= 0.9). Case 'true': 30 missed (rho = 0.925, claim true even without corrections).
    Hypothesis tested at u_top ('review nothing more'), WR draws, mixture betting test.
    a) r re-learned from the audit, terms evaluated with the post-audit r     (previous sim (d))
    b) two-phase: r learned on phase 1 (200 draws), frozen; phase 2 (400 fresh draws) only from
       U'' = U' minus phase-1 units, which join the known stratum with their D           (valid)
    c) r frozen, pre-correction terms, audit corrections credited via the identity
       sum_{U' \\ S} M = sum_{U'} M - sum_S M with an anytime UCB for sum_{U'} M            (valid)
    d) Satz 2 as stated (r frozen, Lemma-0 conservative, corrections not credited)       (valid)
(2) uses autolabel.metrics.label_components (merge_radius = 2) and the tolerance rule tol = 2.

Run:  python3 docs/research/sims/redteam2_satz2_tiles.py     (~30 s)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from redteam_audit_checks import betting_reject_mean_ge  # noqa: E402
from autolabel.metrics import dilate, label_components  # noqa: E402

RNG = np.random.default_rng(5)
DELTA, EPS = 0.05, 0.1


def anytime_ucb_count(xdraws, N, grid):
    """Anytime UCB path for sum of x over N units (x in [0,1], uniform WR draws), beta-grid mixture.
    Returns array (T,) of UCBs; grid convention: smallest grid value not yet ruled out (conservative)."""
    nb = 20
    T = len(xdraws)
    # bet that the mean is < m: factor 1 + beta (m - x) >= 0 for x <= 1 needs beta <= 1/(1 - m)
    betas = np.linspace(0, 0.99, nb + 1)[1:][:, None] / (1 - grid[None, :])
    logk = np.zeros((nb, len(grid)))
    rejected = np.zeros(len(grid), bool)
    out = np.empty(T)
    for t in range(T):
        logk += np.log1p(betas * (grid[None, :] - xdraws[t]))        # factor >= 0.01
        mx = logk.max(0)
        rejected |= mx + np.log(np.exp(logk - mx).mean(0)) >= np.log(1 / DELTA)
        ok = np.nonzero(~rejected)[0]
        out[t] = N * grid[ok.max() + 1 if ok.max() + 1 < len(grid) else ok.max()]
    return out


def satz2_variants():
    N, reps = 2000, 1000
    grid = np.linspace(0.0, 0.2, 401)
    for n_miss, label in ((80, "claim FALSE (rho 0.80)"), (30, "claim TRUE (rho 0.925)")):
        D = np.zeros(N)
        D[:400] = 1
        M = np.zeros(N)
        M[:n_miss] = 1
        Dtot = D.sum()
        res = {k: [] for k in "abcd"}
        for _ in range(reps):
            # a) re-learned r (reproduces sim (d) of redteam_audit_checks)
            idx = RNG.integers(0, N, 400)
            seen = np.unique(idx)
            kb = np.zeros(N, bool)
            kb[seen] = M[seen] > 0
            y = M[idx] * (~kb[idx]) - EPS * D[idx]
            j = betting_reject_mean_ge((y + EPS), EPS)
            final_bad = (M * ~kb).sum() > EPS * Dtot
            res["a"].append((j is not None, final_bad))
            # d) Satz 2 as stated: frozen r, pre-correction terms
            y = M[idx] - EPS * D[idx]
            j = betting_reject_mean_ge(y + EPS, EPS)
            res["d"].append((j is not None, final_bad))
            # b) two-phase
            i1 = RNG.integers(0, N, 200)
            s1 = np.unique(i1)
            rest = np.setdiff1d(np.arange(N), s1)
            Dk = D[s1].sum()
            i2 = rest[RNG.integers(0, len(rest), 400)]
            seen2 = np.union1d(s1, np.unique(i2))
            y2 = M[i2] - EPS * D[i2]
            m_level = EPS * Dk / len(rest)
            j = betting_reject_mean_ge(y2 + EPS, m_level + EPS)
            final_bad_b = (M[rest].sum()) > EPS * Dtot      # nothing reviewed in U'' at u_top
            res["b"].append((j is not None, final_bad_b))
            # c) identity: certify at t if UCB_t(sum M) - (misses found so far) <= eps * Dtot
            ucb = anytime_ucb_count(M[idx], N, grid)
            found = np.zeros(len(idx))
            seen_set, f = set(), 0
            for t, i in enumerate(idx):
                if i not in seen_set:
                    seen_set.add(i)
                    f += M[i]
                found[t] = f
            ok = np.nonzero(ucb - found <= EPS * Dtot)[0]
            cert = len(ok) > 0
            stop = ok[0] if cert else len(idx) - 1
            seen_c = np.unique(idx[:stop + 1])
            final_bad_c = (M.sum() - M[seen_c].sum()) > EPS * Dtot
            res["c"].append((cert, final_bad_c))
        print(f"(1) {label}:")
        names = {"a": "r re-learned from same audit      ", "b": "two-phase (learn r, freeze, fresh)",
                 "c": "frozen r + correction identity   ", "d": "Satz 2 as stated (Lemma 0 only)  "}
        for k in "abcd":
            arr = np.array(res[k])
            false_cert = np.mean(arr[:, 0] & arr[:, 1])
            print(f"    {names[k]}: certifies {arr[:, 0].mean():.3f} | false certificates "
                  f"{false_cert:.3f} (nominal <= {DELTA})")


def lemma0_tiles():
    tol, mr = 2, 2
    # (i) F fragmentation: FP 'U' shape homed in T' (cols >= 32); its bridge lies in T (cols < 32)
    pred = np.zeros((40, 64), bool)
    pred[5, 28:61] = True
    pred[25, 28:61] = True
    pred[5:26, 28:30] = True
    gt = np.zeros_like(pred)

    def f_in_tile(p, c0):
        lab, n = label_components(p, mr)
        cnt = 0
        for k in range(1, n + 1):
            ys, xs = np.nonzero(lab == k)
            home = xs.max()                               # home rule: tile of the right-most pixel
            gt_near = dilate(gt, tol)[ys, xs].any()
            cnt += (not gt_near) and (home >= c0)
        return cnt
    before = f_in_tile(pred, 32)
    pc = pred.copy()
    pc[:, :32] = False                                    # gold correction of tile T only
    after = f_in_tile(pc, 32)
    print(f"(2i) Lemma 0 at tile level, false components homed in uncorrected tile T': before {before}, "
          f"after correcting neighbour tile T: {after}")
    # (ii) M via tolerance: GT component in T' at cols 33-40, the only prediction is at col 31 in T
    gt2 = np.zeros((40, 64), bool)
    gt2[10:13, 33:41] = True
    p2 = np.zeros_like(gt2)
    p2[10:13, 31] = True

    def m_tprime(p):
        glab, ng = label_components(gt2, mr)
        hit = [dilate(p, tol)[glab == k].any() for k in range(1, ng + 1)]
        return int(ng - sum(hit))
    p2c = p2.copy()
    p2c[:, :32] = p2[:, :32] & gt2[:, :32]               # T corrected to exact GT inside T
    print(f"(2ii) missed GT components homed in T': before {m_tprime(p2)}, after correcting T: "
          f"{m_tprime(p2c)}  (tolerance hit removed by an exact correction of T)")


if __name__ == "__main__":
    satz2_variants()
    lemma0_tiles()
