"""Second red-team pass on 06 (completeness estimation): what the GT-free transport test identifies.

Key algebra (J = 2, per covariate bin x): the real data identify only n_obs(x) and the conditional
shares P(h | x, h != 00). N_HT = sum_x n_obs(x) / pi_syn(x) uses the real data ONLY through n_obs(x),
so its consistency hinges on T0: P_real(00 | x) = P_syn(00 | x), which is not identified. The
transport test (06 §4) tests equality of the conditional shares, which N_HT does not use.

  E  T0 holds, conditional shares differ (complementary detectors, real P(11) = 0.4 pA pB, P(00) as in
     the synthetic data): the test rejects, HT is unbiased, HG/Hybrid are biased. 06 §4 says: on
     rejection fall back to Hybrid or Huggins -> the rule makes things worse.
  F  T holds exactly, but defects cluster in images that share an image-level random effect
     (reflection/texture) acting on both detectors: size of the chi^2 transport test (nominal 0.05).
  G  power of the transport test for the shared-hardness scenario (B, sd 1.5) at realistic numbers of
     captured real defects.

Reuses table/estimators/transport_test from redteam_capture_recapture.py.
Run:  python3 docs/research/sims/redteam2_transport_identifiability.py     (~1 min)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import redteam_capture_recapture as cr  # noqa: E402

RNG = np.random.default_rng(99)
REPS = 2000


def draw_cells(x, P, rng):
    """P: (3 bins, 4 cells) in order 10, 01, 11, 00. Returns a, b."""
    u = rng.random(len(x))
    cum = np.cumsum(P[x], 1)
    cell = (u[:, None] > cum).sum(1)
    a = np.isin(cell, (0, 2)).astype(int)
    b = np.isin(cell, (1, 2)).astype(int)
    return a, b


def scenario_E(n_real=300, n_syn=600, kappa=0.4):
    pA, pB = cr.PA, cr.PB
    P = np.stack([pA * (1 - pB) + (1 - kappa) * pA * pB / 2,
                  (1 - pA) * pB + (1 - kappa) * pA * pB / 2,
                  kappa * pA * pB,
                  (1 - pA) * (1 - pB)], 1)
    rej, rel, rule = [], [], []
    for _ in range(REPS):
        xs, as_, bs = cr.simulate(n_syn, RNG)
        syn = cr.table(xs, as_, bs)
        x = RNG.choice(3, n_real, p=cr.BIN_P)
        a, b = draw_cells(x, P, RNG)
        real = np.hstack([cr.table(x, a, b)[:, :3], np.zeros((3, 1))])
        ht, hg, hyb, _ = cr.estimators(real, syn)
        r = cr.transport_test(real, syn)
        rej.append(r)
        rel.append(np.array([ht, hg, hyb]) / n_real - 1)
        rule.append((hg if r else ht) / n_real - 1)        # 06 §4: on rejection switch to Huggins
    rel = np.array(rel)
    print(f"E  T0 holds, shares differ (kappa={kappa}): test rejects {np.mean(rej):.2f} | rel. bias "
          f"HT {rel[:, 0].mean():+.3f} HG {rel[:, 1].mean():+.3f} Hyb {rel[:, 2].mean():+.3f} | "
          f"06-rule (HT, switch to HG on rejection) {np.mean(rule):+.3f}")


def clustered(n_img, per_img_mean, sd_img, rng, fixed_k=None):
    """Defects clustered in images; image effect z_i ~ N(0, sd_img) lowers both logits."""
    k = np.full(n_img, fixed_k) if fixed_k else 1 + rng.poisson(per_img_mean - 1, n_img)
    z = np.repeat(rng.normal(0, sd_img, n_img), k)
    n = k.sum()
    x = rng.choice(3, n, p=cr.BIN_P)
    pa = cr.expit(cr.logit(cr.PA[x]) - z)
    pb = cr.expit(cr.logit(cr.PB[x]) - z)
    return x, (rng.random(n) < pa).astype(int), (rng.random(n) < pb).astype(int)


def scenario_F():
    for sd in (0.0, 1.0, 2.0):
        for per_img in (1, 4):
            rej = []
            for _ in range(REPS):
                xs, as_, bs = clustered(200, None, sd, RNG, fixed_k=3)       # 600 injections, 3 per image
                syn = cr.table(xs, as_, bs)
                x, a, b = clustered(int(300 / per_img), per_img, sd, RNG, fixed_k=None if per_img > 1 else 1)
                real = np.hstack([cr.table(x, a, b)[:, :3], np.zeros((3, 1))])
                rej.append(cr.transport_test(real, syn))
            print(f"F  T holds marginally; image effect sd {sd:.1f}, real defects/image ~{per_img}, "
                  f"3 injections/image: chi^2 test size {np.mean(rej):.3f} (nominal 0.05)")


def scenario_G():
    for n_real in (40, 100, 300):
        rej = []
        for _ in range(REPS):
            xs, as_, bs = cr.simulate(600, RNG)
            syn = cr.table(xs, as_, bs)
            x, a, b = cr.simulate(n_real, RNG, latent=1.5)
            real = np.hstack([cr.table(x, a, b)[:, :3], np.zeros((3, 1))])
            rej.append(cr.transport_test(real, syn))
        print(f"G  shared hardness sd 1.5 (HT bias ~ -0.08, HG ~ -0.17), {n_real} real defects in U: "
              f"test power {np.mean(rej):.2f}")


if __name__ == "__main__":
    scenario_E()
    scenario_E(kappa=0.0)
    scenario_F()
    scenario_G()
