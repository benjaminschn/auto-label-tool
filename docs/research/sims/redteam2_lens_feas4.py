"""Tiles vs images at EQUAL HUMAN TIME instead of equal area. Cost per unit = c0 + c_area*area_share.
Image: c0 + c_img; tile (1/16 area): c0 + c_img/16. Reuses make_pool / fieller_lcb of the earlier red-team."""
import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import redteam_audit_checks as R
rng = np.random.default_rng(17)
Dt, Mt = R.make_pool(cluster=0.5, rng=rng)
Di, Mi = Dt.sum(1), Mt.sum(1); Dtile, Mtile = Dt.ravel(), Mt.ravel()
print("true recall", 1 - Mt.sum()/Dt.sum(), "Dmax tile", Dtile.max())
c_img = 30.0
budget_s = 0.10 * 2000 * (3.0 + c_img)        # human seconds that buy 10% of images at c0 = 3 s
for c0 in (0.0, 1.0, 3.0, 6.0):
    n_img = int(budget_s / (c0 + c_img)); n_tile = int(budget_s / (c0 + c_img / 16))
    li, _ = R.fieller_lcb(Mi, Di, n_img, dmax=6, rng=rng, reps=100)
    lt, _ = R.fieller_lcb(Mtile, Dtile, n_tile, dmax=Dtile.max(), rng=rng, reps=60)
    print(f"c0={c0:3.0f}s: images n={n_img:4d} ({n_img/2000:.1%} area) LCB {li:.3f} | tiles n={n_tile:5d} "
          f"({n_tile/32000:.1%} area) LCB {lt:.3f}")
