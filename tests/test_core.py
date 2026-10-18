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
