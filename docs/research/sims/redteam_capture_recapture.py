"""Red-team check of 06 (completeness estimation): what the GT-free transport test can and cannot see.

Two detectors A (labeler) and B (control detector), covariate x = size bin (3 bins). Synthetic
(injected) defects have known capture histories incl. h = 00. Real defects in U are observed only
when h != 00. Estimators (per bin, summed):
  HT     : N = sum_b n_obs_b / pi_syn_b                    (06 §3.1, assumption T)
  HG     : Lincoln-Petersen / Huggins under CI per bin      (06 §3.2, assumption CI)
  Hybrid : odds ratio taken from injection, levels from real (06 §3.3)
Transport test (06 §4): chi^2 homogeneity test of real vs. synthetic cell shares among h != 00,
per bin, summed (df = 2 per bin). Best case: q_r is perfect unless stated otherwise.

Scenarios
  0  null: real == synthetic, CI holds                       (sanity)
  A  30 % of real defects invisible to *both* detectors       (novel type / too faint)
  B  shared latent hardness for real defects only             (correlated blind spots)
  C  T and CI hold, but 300 false B-only regions get q = 0.1  (stacker miscalibrated where it has no data)
  D  T holds in true x, but real x is measured from detector output (B-only catches look one bin larger
     half of the time), synthetic x is known exactly        (covariate measured differently)

Run:  python docs/research/sims/redteam_capture_recapture.py      (~10 s)
"""
import numpy as np
from scipy.stats import chi2

RNG = np.random.default_rng(2026)
BIN_P = np.array([0.4, 0.4, 0.2])          # small / medium / large
PA = np.array([0.50, 0.80, 0.95])          # synthetic capture prob. of A per bin
PB = np.array([0.30, 0.50, 0.70])          # synthetic capture prob. of B per bin
N_REAL, N_SYN, REPS, DELTA = 300, 600, 2000, 0.05
CELLS = [(1, 0), (0, 1), (1, 1)]           # observable cells


def logit(p):
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def simulate(n, rng, invisible=0.0, latent=0.0):
    x = rng.choice(3, n, p=BIN_P)
    z = rng.normal(size=n)
    pa = expit(logit(PA[x]) - latent * z)
    pb = expit(logit(PB[x]) - latent * z)
    vis = rng.random(n) >= invisible
    a = (rng.random(n) < pa) & vis
    b = (rng.random(n) < pb) & vis
    return x, a.astype(int), b.astype(int)


def table(x, a, b, w=None):
    """counts[bin, cell] for cells 10, 01, 11, 00 (weights optional)."""
    w = np.ones(len(x)) if w is None else w
    t = np.zeros((3, 4))
    for k, (ha, hb) in enumerate(CELLS + [(0, 0)]):
        m = (a == ha) & (b == hb)
        np.add.at(t[:, k], x[m], w[m])
    return t


def estimators(real, syn):
    n10, n01, n11 = real[:, 0], real[:, 1], real[:, 2]
    nobs = n10 + n01 + n11
    pi_syn = 1 - syn[:, 3] / syn.sum(1)
    ht = (nobs / pi_syn).sum()
    hg = (nobs + n10 * n01 / np.maximum(n11, 0.5)).sum()          # Lincoln-Petersen per bin
    s10, s01, s11, s00 = (syn[:, k] + 0.5 for k in range(4))
    orr = s11 * s00 / (s10 * s01)
    hyb = (nobs + orr * n10 * n01 / np.maximum(n11, 0.5)).sum()
    nA = (n10 + n11).sum()
    return ht, hg, hyb, nA


def transport_test(real, syn):
    stat, df = 0.0, 0
    for b in range(3):
        obs = np.vstack([real[b, :3], syn[b, :3]])
        if (obs.sum(1) == 0).any():
            continue
        exp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
        ok = exp.sum(0) > 0
        stat += (((obs - exp) ** 2)[:, ok] / exp[:, ok]).sum()
        df += ok.sum() - 1
    return chi2.sf(stat, max(df, 1)) < DELTA


def run(name, invisible=0.0, latent=0.0, false_b=0, q_false=0.0, xshift=False):
    rej, rel, agree, ra_hat, ra_true = [], [], [], [], []
    for _ in range(REPS):
        xs, as_, bs = simulate(N_SYN, RNG)
        syn = table(xs, as_, bs)
        x, a, b = simulate(N_REAL, RNG, invisible, latent)
        N = len(x)
        nA_true = a.sum()
        xm = x.copy()
        if xshift:   # B-only detections: measured size inflated by one bin half of the time
            m = (a == 0) & (b == 1) & (RNG.random(N) < 0.5)
            xm[m] = np.minimum(xm[m] + 1, 2)
        w = np.ones(N)
        if false_b:
            xf = RNG.choice(3, false_b, p=[0.6, 0.3, 0.1])
            xm = np.concatenate([xm, xf])
            a = np.concatenate([a, np.zeros(false_b, int)])
            b = np.concatenate([b, np.ones(false_b, int)])
            w = np.concatenate([w, np.full(false_b, q_false)])
        real = table(xm, a, b, w)[:, :3]
        real = np.hstack([real, np.zeros((3, 1))])
        ht, hg, hyb, nA = estimators(real, syn)
        est = np.array([ht, hg, hyb])
        rej.append(transport_test(real, syn))
        rel.append(est / N - 1)
        agree.append(est.max() / est.min() < 1.15)
        ra_hat.append(nA / ht)
        ra_true.append(nA_true / N)
    rel = np.array(rel)
    print(f"{name:<44s} test rejects {np.mean(rej):5.2f} | rel. bias N: HT {rel[:, 0].mean():+.2f} "
          f"HG {rel[:, 1].mean():+.2f} Hyb {rel[:, 2].mean():+.2f} | 'triangulation agrees' "
          f"{np.mean(agree):.2f} | recall_A est {np.mean(ra_hat):.2f} vs true {np.mean(ra_true):.2f}")


if __name__ == "__main__":
    print(f"N_real={N_REAL}, injections={N_SYN}, reps={REPS}, test level {DELTA}")
    run("0 null (T and CI hold)")
    run("A 30% of real defects invisible to both", invisible=0.3)
    run("B shared latent hardness (sd 1.5 logit)", latent=1.5)
    run("B' shared latent hardness (sd 3 logit)", latent=3.0)
    run("C 300 false B-only regions with q=0.1", false_b=300, q_false=0.1)
    run("D x of B-only catches measured too large", xshift=True)
