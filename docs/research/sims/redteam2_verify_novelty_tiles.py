"""Novelty-lens verification (red team 2) for finding C5 (00 §4 S4 / 04 N6: tiles make the betting
recall bounds 'um Groessenordnungen schaerfer').

Red team (c) compared LCBs at equal inspected area. Here the complementary question: how much
inspected AREA (share of image area that a human inspects) is needed until the betting-Fieller test
(Prop. 4, uniform with-replacement draws, beta-grid mixture, no predictor) rejects H(r): rho <= r,
for images (D_max = 6) vs 256-px tiles (16 per image, D_max registered = empirical max or 3).
'Orders of magnitude' would mean an area ratio >= 10.

Run:  python docs/research/sims/redteam2_verify_novelty_tiles.py      (CPU, ~1 min)
"""
import numpy as np

DELTA = 0.05
RNG = np.random.default_rng(23)


def make_pool(N_img=2000, T=16, prev=0.1, p_miss=0.1, cluster=0.5, rng=RNG):
    D_img = np.minimum(np.where(rng.random(N_img) < prev, 1 + rng.poisson(0.6, N_img), 0), 6)
    Dt = np.zeros((N_img, T), int)
    Mt = np.zeros((N_img, T), int)
    for i in np.nonzero(D_img)[0]:
        home = rng.integers(T)
        for _ in range(D_img[i]):
            t = home if rng.random() < cluster else rng.integers(T)
            Dt[i, t] += 1
            Mt[i, t] += rng.random() < p_miss
    return Dt, Mt


def area_to_reject(M, D, r, dmax, reps, max_share=1.0, nb=24):
    """Inspected share of units (distinct) at the first rejection of H(r): mean g >= 0, g = M-(1-r)D."""
    N = len(M)
    zmax = r * dmax
    betas = (0.95 / zmax) * np.logspace(-3, 0, nb)
    T = int(-N * np.log(1 - min(max_share, 0.999)))     # draws with replacement for that distinct share
    out = []
    for _ in range(reps):
        idx = RNG.integers(0, N, T)
        g = M[idx] - (1 - r) * D[idx]
        logk = np.cumsum(np.log1p(-betas[None, :] * g[:, None]), axis=0)
        mx = logk.max(1)
        cap = mx + np.log(np.exp(logk - mx[:, None]).mean(1))
        hit = np.nonzero(cap >= np.log(1 / DELTA))[0]
        if len(hit):
            t = hit[0] + 1
            seen = np.zeros(N, bool)
            seen[idx[:t]] = True
            out.append(seen.mean())
        else:
            out.append(np.inf)
    return np.array(out)


def main():
    for cluster in (0.0, 0.5):
        Dt, Mt = make_pool(cluster=cluster)
        rho = 1 - Mt.sum() / Dt.sum()
        Di, Mi = Dt.sum(1), Mt.sum(1)
        Dtl, Mtl = Dt.ravel(), Mt.ravel()
        print(f"cluster={cluster}: {Dt.sum()} defects, true recall {rho:.3f}, max/img {Di.max()}, max/tile {Dtl.max()}")
        for r in (0.6, 0.7, 0.8):
            a_img = area_to_reject(Mi, Di, r, 6, reps=200)
            res = [f"images(D_max=6) {np.median(a_img):.3f}"]
            for dm in sorted({int(Dtl.max()), 3}):
                a_t = area_to_reject(Mtl, Dtl, r, dm, reps=60)
                ratio = np.median(a_img) / np.median(a_t)
                res.append(f"tiles(D_max={dm}) {np.median(a_t):.3f} -> area ratio {ratio:.1f}x")
            print(f"    H(rho<={r}) rejected: median inspected area share  " + " | ".join(res))


if __name__ == "__main__":
    main()
