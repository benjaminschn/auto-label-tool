"""Does a bounded image-level recall metric make tiles (C5) unnecessary?

Reuses make_pool / fieller_lcb from docs/research/sims/redteam_audit_checks.py (imported read-only, no .pyc).
Estimands (uniform WR draws, betting-Fieller LCB, delta = 0.05, same inspected area):
  pooled recall, image unit (D_max = 6)         rho     = 1 - sum M / sum D
  pooled recall, tile unit (D_max = max/tile)   rho     (same estimand, smaller range)
  image-weighted recall (range 1, no D_max)     rho_img = 1 - sum_i (M_i/D_i) / #defective images
  defective-image recall (binary)               rho_b   = 1 - #(images with >=1 miss) / #defective images
"""
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np  # noqa: E402
import redteam_audit_checks as rc  # noqa: E402

rng = np.random.default_rng(42)
for cluster, label in ((0.0, "spread"), (0.5, "clustered")):
    for p_miss in (0.1, 0.25):
        Dt, Mt = rc.make_pool(cluster=cluster, p_miss=p_miss, rng=rng)
        Di, Mi = Dt.sum(1), Mt.sum(1)
        dfi = Di > 0
        rho = 1 - Mi.sum() / Di.sum()
        fnr = np.where(dfi, Mi / np.maximum(Di, 1), 0.0)
        rho_img = 1 - fnr.sum() / dfi.sum()
        bi = (Mi > 0).astype(float)
        rho_b = 1 - bi.sum() / dfi.sum()
        print(f"{label}, p_miss={p_miss}: rho={rho:.3f} rho_img={rho_img:.3f} rho_b={rho_b:.3f}; "
              f"D_max image={Di.max()} tile={Dt.max()}")
        for frac in (0.1, 0.3):
            n = int(frac * 2000)
            l1, c1 = rc.fieller_lcb(Mi, Di, n, dmax=6, rng=rng)
            l2, c2 = rc.fieller_lcb(Mt.ravel(), Dt.ravel(), n * 16, dmax=Dt.max(), rng=rng, reps=100)
            l3, _ = rc.fieller_lcb(fnr, dfi.astype(float), n, dmax=1, rng=rng)
            l4, _ = rc.fieller_lcb(bi, dfi.astype(float), n, dmax=1, rng=rng)
            print(f"   area {frac:.0%}: gap true-LCB  pooled/images {rho - l1:.3f} | pooled/tiles {rho - l2:.3f} | "
                  f"image-weighted/images {rho_img - l3:.3f} | defective-image/images {rho_b - l4:.3f}   "
                  f"(LCBs {l1:.3f} {l2:.3f} {l3:.3f} {l4:.3f})", flush=True)
