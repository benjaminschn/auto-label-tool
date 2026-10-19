import numpy as np
import pytest

from autolabel import conformal as C
from autolabel import metrics as M
from autolabel import ppi
from autolabel.augment import build_defect_bank, paste_defect, procedural_defect, random_crop
from autolabel.calibration import IsotonicCalibrator
from autolabel.postprocess import hysteresis


def _line_mask(h=64, w=64, y=30):
    m = np.zeros((h, w), bool)
    m[y, 10:50] = True
    return m


def test_metrics_perfect_and_tolerance():
    gt = _line_mask()
    s = M.image_stats(gt, gt)
    assert s["tp"] == gt.sum() and s["fp"] == 0 and s["fn"] == 0
    shifted = np.roll(gt, 1, axis=0)  # 1 px off: exact F1 = 0, tolerant F1 = 1
    agg = M.aggregate([M.image_stats(shifted, gt, tol=2)])
    assert agg["px_f1"] == 0.0
    assert agg["tol_f1"] == pytest.approx(1.0)
    assert agg["defect_recall"] == 1.0


def test_defect_level_counts():
    gt = np.zeros((64, 64), bool)
    gt[5:8, 5:8] = True
    gt[40:44, 40:44] = True
    pred = np.zeros_like(gt)
    pred[5:8, 5:8] = True
    pred[20:22, 50:52] = True  # false alarm
    s = M.image_stats(pred, gt)
    assert s["gt_comp"] == 2 and s["gt_comp_hit"] == 1
    assert s["pred_comp"] == 2 and s["pred_comp_hit"] == 1


def test_pixel_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(0)
    g = [rng.rand(30, 30) > 0.9 for _ in range(3)]
    p = [np.clip(x * 0.3 + rng.rand(30, 30) * 0.8, 0, 1) for x in g]
    ref = roc_auc_score(np.concatenate([x.ravel() for x in g]), np.concatenate([x.ravel() for x in p]))
    assert M.pixel_auroc(p, g) == pytest.approx(ref, abs=1e-3)


def test_hysteresis_keeps_faint_connected_parts():
    p = np.zeros((20, 40), np.float32)
    p[10, 5:35] = 0.3
    p[10, 20] = 0.9          # strong seed on the line
    p[3, 3:8] = 0.35          # isolated faint blob
    m = hysteresis(p, 0.8, 0.25)
    assert m[10, 5:35].all() and not m[3].any()


def test_conformal_threshold_coverage():
    rng = np.random.RandomState(0)
    alpha, hits = 0.1, []
    for _ in range(2000):
        s = rng.beta(2, 1, size=31)
        t = C.conformal_threshold(s[:30], alpha)
        hits.append(s[30] >= t)
    assert np.mean(hits) >= 1 - alpha - 0.02


def test_conformal_too_few_samples_is_honest():
    assert C.conformal_threshold(np.array([0.9, 0.8, 0.7]), 0.1) == 0.0
    assert C.pac_threshold(np.array([0.9] * 5), 0.1, 0.1) == 0.0


def test_ppi_ratio_coverage_and_width():
    rng = np.random.RandomState(1)
    N = 600
    b = rng.poisson(30, N).astype(float) * (rng.rand(N) < 0.4)        # many images without defects
    a = rng.binomial(b.astype(int), rng.beta(4, 1, N)).astype(float)    # per-image recall varies
    fb = b * rng.uniform(0.9, 1.1, N)                                   # good but imperfect self-assessment
    fa = a * rng.uniform(0.9, 1.1, N)
    true = a.sum() / b.sum()
    cov_p, w_p, w_c, w_t = [], [], [], []
    for _ in range(400):
        s = rng.choice(N, 40, replace=False)
        r = np.setdiff1d(np.arange(N), s)
        _, lo, hi = ppi.estimate_ratio(a[s], b[s], fa[s], fb[s], fa[r], fb[r], 0.1)
        _, lot, hit = ppi.estimate_ratio(a[s], b[s], fa[s], fb[s], fa[r], fb[r], 0.1, lam=None)
        _, loc, hic = ppi.classical_ratio(a[s], b[s], 0.1, N)
        cov_p.append(lo <= true <= hi)
        w_p.append(hi - lo)
        w_t.append(hit - lot)
        w_c.append(hic - loc)
    assert np.mean(cov_p) > 0.85
    assert np.mean(w_p) < 0.5 * np.mean(w_c)
    assert np.mean(w_t) < 1.05 * np.mean(w_c)


def test_ppi_ratio_trivial_interval_without_denominator():
    # no defect in the audit and no predicted defect anywhere -> "no interval", not NaN bounds
    z = np.zeros(10)
    assert ppi.classical_ratio(z, z, 0.1, 100)[1:] == (0.0, 1.0)
    e, lo, hi = ppi.estimate_ratio(z, z, z, z, np.zeros(90), np.zeros(90), 0.1)
    assert np.isnan(e) and (lo, hi) == (0.0, 1.0)
    # estimated denominator total <= 0 (self-assessment far above the audited truth)
    fb = np.full(10, 5.0)
    e, lo, hi = ppi.estimate_ratio(z, z, z, fb, np.zeros(90), np.zeros(90), 0.1)
    assert np.isnan(e) and (lo, hi) == (0.0, 1.0)


def test_ppi_mean_matches_truth_on_average():
    rng = np.random.RandomState(2)
    y = rng.beta(5, 2, 500)
    f = np.clip(y + rng.normal(0.05, 0.05, 500), 0, 1)   # biased predictor
    ests = []
    for _ in range(300):
        s = rng.choice(500, 30, replace=False)
        r = np.setdiff1d(np.arange(500), s)
        ests.append(ppi.estimate_mean(y[s], f[s], f[r])[0])
    assert abs(np.mean(ests) - y.mean()) < 0.005       # audit removes the predictor's bias


def test_isotonic_calibrator_monotone():
    rng = np.random.RandomState(0)
    p = [rng.rand(40, 40) for _ in range(5)]
    y = [rng.rand(40, 40) < x ** 2 for x in p]
    cal = IsotonicCalibrator().fit(p, y)
    grid = np.linspace(0.01, 0.99, 50)
    out = cal(grid)
    assert np.all(np.diff(out) >= -1e-9)
    assert abs(cal(np.array([0.5]))[0] - 0.25) < 0.1


def test_copy_paste_and_procedural():
    rng = np.random.RandomState(0)
    img = np.full((80, 80, 3), 120, np.uint8)
    img[40, 10:60] = 60
    bank = build_defect_bank([img], [_line_mask(80, 80, 40)])
    assert len(bank) == 1 and bank[0].mode == "residual"
    canvas = np.full((64, 64, 3), 150, np.float32)
    tgt = np.zeros((64, 64), bool)
    m = paste_defect(rng, canvas, tgt, bank[0], gain_range=(1, 1), scale_range=(0.8, 0.8))
    assert m.any() and tgt.any()
    assert canvas[m].mean() < 140          # the dark scratch was transferred as a residual
    m2 = procedural_defect(rng, canvas, tgt, np.array([-40.0]), kind="scratch")
    assert m2.any()


def test_random_crop_pads_consistently():
    rng = np.random.RandomState(0)
    img = np.zeros((50, 70, 3), np.uint8)
    msk = np.zeros((50, 70), bool)
    img[10, :, :] = 255
    msk[10, :] = True
    ci, cm = random_crop(rng, [img, msk], 96)
    assert ci.shape[:2] == (96, 96) and cm.shape == (96, 96)
    assert np.array_equal(ci[..., 0] == 255, cm)  # reflected defects keep their labels


def test_unet_forward_shape():
    import torch
    from autolabel.model import UNet
    m = UNet("resnet18d", pretrained=False).eval()
    with torch.no_grad():
        y = m(torch.randn(1, 3, 64, 96))
    assert y.shape == (1, 1, 64, 96)


def test_normality_knn_excludes_own_image():
    import torch
    from autolabel.anomaly import NormalityModel
    torch.manual_seed(0)
    m = NormalityModel.__new__(NormalityModel)
    m.k, m.ppi = 3, 5
    feats = {i: torch.randn(16, 8) for i in range(6)}
    bank, ids = [], []
    for i in range(6):
        sel = torch.randperm(16)[:5]
        bank.append(feats[i][sel])
        ids.append(torch.full((5,), i))
    m.bank, m.bank_ids = torch.cat(bank), torch.cat(ids)
    m.bank_sq = (m.bank ** 2).sum(1)
    for i in range(6):
        q = feats[i]
        dist = (q ** 2).sum(1, keepdim=True) - 2 * q @ m.bank.T + m.bank_sq[None]
        dist[:, m.bank_ids == i] = float("inf")
        ref = torch.topk(dist, 3, dim=1, largest=False).values[:, -1].clamp(min=0).sqrt()
        assert torch.allclose(m.knn_distance(q, i), ref, atol=1e-5)


def test_summarize_get_keys_with_dots():
    from scripts.summarize import get
    d = {"conformal": {"alpha=0.1": {"t": 0.3}}, "review": {"risk/0.05": {"defect_recall": 0.9}},
         "rounds": [{"U": {"tol_f1": 0.5}}]}
    assert get(d, "conformal.alpha=0.1.t") == 0.3
    assert get(d, "review.risk/0.05.defect_recall") == 0.9
    assert get(d, "rounds.0.U.tol_f1") == 0.5
    assert get(d, "rounds.3.U") is None and get(d, "nope.x") is None


# --------------------------------------------------------------------------- synthesis label contract

def _bright_line_bank(delta=80.0):
    img = np.full((80, 80, 3), 120, np.uint8)
    img[40, 10:50] = 120 + delta      # exactly the masked pixels (clean inpainting)
    return build_defect_bank([img], [_line_mask(80, 80, 40)])


def test_realised_label_clipped_defect_is_not_positive():
    """A bright residual pasted on a saturated background is invisible -> ignore, never positive."""
    from autolabel.augment import realised_label
    rng = np.random.RandomState(0)
    bank = _bright_line_bank(80.0)
    canvas = np.full((64, 64, 3), 255, np.float32)
    tgt = np.zeros((64, 64), bool)
    ign = np.zeros((64, 64), bool)
    before = canvas.copy()
    pos = paste_defect(rng, canvas, tgt, bank[0], gain_range=(1, 1), scale_range=(1, 1), ignore=ign)
    assert np.abs(canvas - before).max() < 2.0       # (almost) nothing changed after clipping
    assert not pos.any() and not tgt.any()
    assert ign.sum() >= 30                           # the whole source line became ignore
    # direct call: same result
    m = np.zeros((20, 20), bool)
    m[10, 3:17] = True
    p2, i2, _ = realised_label(np.full((20, 20, 3), 255.0), np.full((20, 20, 3), 255.0), m)
    assert not p2.any() and (i2 & m).sum() == m.sum()


def test_realised_label_faint_and_partially_clipped():
    rng = np.random.RandomState(1)
    bank = build_defect_bank([np.repeat(np.where(_line_mask(80, 80, 40)[..., None], 60, 120), 3, -1).astype(np.uint8)],
                             [_line_mask(80, 80, 40)])
    noise = np.random.RandomState(2).normal(0, 1.0, (64, 64, 1)).astype(np.float32)
    # (a) clearly visible dark scratch on mid-gray: positive ~ source mask
    canvas = np.full((64, 64, 3), 150, np.float32) + noise
    tgt, ign = np.zeros((64, 64), bool), np.zeros((64, 64), bool)
    pos = paste_defect(rng, canvas, tgt, bank[0], gain_range=(1, 1), scale_range=(1, 1), ignore=ign)
    assert pos.sum() >= 35 and not (ign & pos).any()
    # (b) gain so small that the change is below the visibility threshold: no positive
    canvas = np.full((64, 64, 3), 150, np.float32) + noise
    tgt, ign = np.zeros((64, 64), bool), np.zeros((64, 64), bool)
    pos = paste_defect(rng, canvas, tgt, bank[0], gain_range=(0.02, 0.02), scale_range=(1, 1), ignore=ign)
    assert not pos.any() and ign.sum() >= 35
    # (c) left half black (dark residual clipped away), right half gray: positive only on the right
    canvas = np.full((64, 64, 3), 150, np.float32)
    canvas[:, :32] = 0
    tgt, ign = np.zeros((64, 64), bool), np.zeros((64, 64), bool)
    pos = paste_defect(rng, canvas, tgt, bank[0], gain_range=(1, 1), scale_range=(1, 1), center=(32, 32),
                       ignore=ign)
    assert pos.any() and not pos[:, :32].any()
    assert ign[:, :30].any()                          # the clipped part is ignore, not negative


def test_synthetic_sample_weights_ignore_invisible_paste():
    """End-to-end through the sample generator: invisible pasted defects get weight 0, label 0."""
    from autolabel.train import SampleGenerator, TrainConfig, TrainData
    img = np.full((80, 80, 3), 120, np.uint8)
    img[40, 10:50] = 200
    bg = np.full((96, 96, 3), 255, np.uint8)
    cfg = TrainConfig(crop=64, p_synth=1.0, scale_range=(1.0, 1.0))
    gen = SampleGenerator(TrainData(labeled=[(img, _line_mask(80, 80, 40))], backgrounds=[bg]), cfg,
                          np.random.RandomState(0))
    for _ in range(5):
        _, tgt, w = gen._synthetic()
        assert not tgt.any()
        assert (w == 0).any()
