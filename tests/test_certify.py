"""Tests for autolabel.certify (CertAL-lite certification core). Fast, CPU only."""
import json
import math

import numpy as np
import pytest
from scipy.stats import hypergeom

from autolabel import certify as CT

DELTA = 0.05


# ------------------------------------------------------------------------------ hypergeom_ucb
def _ucb_brute(found, n, N, delta):
    Ks = np.arange(found, N - (n - found) + 1)
    return int(Ks[hypergeom.cdf(found, N, Ks, n) > delta].max())


def test_hypergeom_ucb_reference_numbers():
    # 04 S4 / 07 section 2.1: N=2000, nothing found -> 188 / 57 / 18
    assert [CT.hypergeom_ucb(0, n, 2000, DELTA) for n in (30, 100, 300)] == [188, 57, 18]
    # 07 F10: smallest n with UCB <= 7 at found=0 is 624 (not 640)
    n624 = next(n for n in range(500, 800) if CT.hypergeom_ucb(0, n, 2000, DELTA) <= 7)
    assert n624 == 624
    # 07 section 3.4 / 5.2: zero-found n for K* = 40 is 140 (N=2000) and 129 (N'=1850)
    assert next(n for n in range(1, 2001) if CT.hypergeom_ucb(0, n, 2000, DELTA) <= 40) == 140
    assert next(n for n in range(1, 1851) if CT.hypergeom_ucb(0, n, 1850, DELTA) <= 40) == 129


def test_hypergeom_ucb_matches_brute_force_and_edges():
    N = 150
    for n in (1, 7, 40, 150):
        for found in sorted({0, 1, n // 3, n // 2, n}):
            assert CT.hypergeom_ucb(found, n, N, DELTA) == _ucb_brute(found, n, N, DELTA)
    assert CT.hypergeom_ucb(3, 150, 150, DELTA) == 3          # full census is exact
    assert CT.hypergeom_ucb(0, 0, 100, DELTA) == 100           # no data -> trivial bound
    with pytest.raises(ValueError):
        CT.hypergeom_ucb(5, 4, 100, DELTA)


def test_hypergeom_ucb_exact_coverage_all_K():
    # exact (not Monte-Carlo) false-certificate probability P_K(K > UCB(X)) <= delta for every K
    N, n = 120, 30
    ucb = np.array([CT.hypergeom_ucb(x, n, N, DELTA) for x in range(n + 1)])
    for K in range(N + 1):
        xs = np.arange(0, n + 1)
        p_fail = float(np.sum(hypergeom.pmf(xs, N, K, n) * (ucb < K)))
        assert p_fail <= DELTA + 1e-12


def test_hypergeom_ucb_monte_carlo_fixed_n():
    rng = np.random.default_rng(1)
    N, n, reps = 2000, 300, 400
    for K in (20, 41):
        pop = np.zeros(N, int)
        pop[:K] = 1
        fails = 0
        for _ in range(reps):
            found = int(pop[rng.choice(N, n, replace=False)].sum())
            fails += CT.hypergeom_ucb(found, n, N, DELTA) < K
        # nominal <= 0.05; 3 binomial sd of slack for 400 reps
        assert fails / reps <= DELTA + 3 * math.sqrt(DELTA * (1 - DELTA) / reps)


# ------------------------------------------------------------------------------ planning
def test_plan_n_reference_and_power():
    # fixed n exact SRSWOR, K=10 of 2000, certify K <= 40 without credit:
    # redteam2_lens_fixedn_vs_seq.py (5-step grid) gives n50 = 220, n90 = 355
    n90, pw90 = CT.plan_n(2000, 10, 40, DELTA, 0.9, credit_found=False)
    n50, pw50 = CT.plan_n(2000, 10, 40, DELTA, 0.5, credit_found=False)
    assert 350 < n90 <= 355 and pw90 >= 0.9
    assert 215 < n50 <= 220 and pw50 >= 0.5
    assert CT.certification_power(n90 - 1, 2000, 10, 40, DELTA, credit_found=False) < 0.9
    # 07 section 5.2: n = 310 planned for K'=10 has power 0.93 / 0.56 / 0.23 at K' = 10/20/30
    # (reproduced with N' = 1850 and credited corrections)
    pws = [CT.certification_power(310, 1850, K, 40, DELTA) for K in (10, 20, 30)]
    assert np.allclose(pws, [0.93, 0.56, 0.23], atol=0.006)
    # crediting corrections never needs a larger n
    assert CT.plan_n(2000, 10, 40, DELTA)[0] <= n90


def test_certification_power_matches_brute_force():
    N, n, K, Kt = 400, 90, 12, 15
    xs = np.arange(0, K + 1)
    pmf = hypergeom.pmf(xs, N, K, n)
    for credit in (True, False):
        ok = np.array([CT.hypergeom_ucb(x, n, N, DELTA) - (x if credit else 0) <= Kt for x in xs])
        assert math.isclose(CT.certification_power(n, N, K, Kt, DELTA, credit), float((pmf * ok).sum()),
                            rel_tol=1e-9, abs_tol=1e-12)


def test_plan_n_monotone():
    N = 1000
    by_forecast = [CT.plan_n(N, kf, 20, DELTA)[0] for kf in (0, 2, 4, 6, 8, 10)]
    assert all(a <= b for a, b in zip(by_forecast, by_forecast[1:]))
    by_target = [CT.plan_n(N, 5, kt, DELTA)[0] for kt in (10, 15, 20, 30, 50)]
    assert all(a >= b for a, b in zip(by_target, by_target[1:]))
    by_power = [CT.plan_n(N, 5, 20, DELTA, pw)[0] for pw in (0.5, 0.8, 0.9, 0.95)]
    assert all(a <= b for a, b in zip(by_power, by_power[1:]))
    by_delta = [CT.plan_n(N, 5, 20, d)[0] for d in (0.1, 0.05, 0.01)]
    assert all(a <= b for a, b in zip(by_delta, by_delta[1:]))
    # not reachable without credit when the forecast exceeds the target
    n, best = CT.plan_n(N, 30, 20, DELTA, credit_found=False)
    assert n is None and best < 0.9
    tab = CT.n90_table(N, [20], [0, 5], DELTA)
    assert tab[(20, 0)] == by_forecast[0] and tab[(20, 5)] == CT.plan_n(N, 5, 20, DELTA)[0]


# ------------------------------------------------------------------------------ certify_zeta
def test_certify_zeta_values_and_refusal():
    c = CT.certify_zeta(found=2, n=300, N=2000, delta=DELTA, n_fixed_registered=300, K_target=40)
    ucb = CT.hypergeom_ucb(2, 300, 2000, DELTA)
    assert c.K_ucb == ucb and c.K_final_ucb == ucb - 2
    assert math.isclose(c.zeta_final_ucb, (ucb - 2) / 2000)
    assert c.certified is (ucb - 2 <= 40)
    assert "LTPD" in c.statement
    with pytest.raises(CT.CertificateRefused):
        CT.certify_zeta(found=2, n=250, N=2000, delta=DELTA, n_fixed_registered=300)


# ------------------------------------------------------------------------------ PPR (anytime)
def test_ppr_basic_properties():
    N = 300
    u = CT.ppr_wor_ucb([0] * 100, N, DELTA, prior=(1, 30))
    assert u.shape == (100,) and np.all(np.diff(u) <= 0) and u[-1] < u[0] <= N
    seq = [1, 0, 1, 1, 0] * 6
    u = CT.ppr_wor_ucb(seq, N, DELTA)
    assert np.all(u >= np.cumsum(seq))                       # never below what was found
    # full census: the bound collapses to the true count
    pop = np.zeros(60, int)
    pop[:7] = 1
    assert CT.ppr_wor_ucb(pop, 60, DELTA)[-1] == 7
    # evaluating less often is still a (weaker or equal) valid bound
    rng = np.random.default_rng(3)
    s = rng.permutation(np.r_[np.ones(5, int), np.zeros(N - 5, int)])[:150]
    assert np.all(CT.ppr_wor_ucb(s, N, DELTA, every=5) >= CT.ppr_wor_ucb(s, N, DELTA, every=1))
    with pytest.raises(ValueError):
        CT.ppr_wor_ucb([0, 2], N, DELTA)


def test_ppr_monitored_false_certificate_rate_and_naive_contrast():
    # boundary case: K_true = K_target + 1, monitored every 5 draws, stop at first certificate
    rng = np.random.default_rng(7)
    N, K_target, n_max, reps = 500, 10, 300, 300
    pop = np.zeros(N, int)
    pop[:K_target + 1] = 1
    checks = np.arange(5, n_max + 1, 5)
    ppr_false = naive_false = 0
    for _ in range(reps):
        s = rng.permutation(pop)[:n_max]
        ppr_false += bool((CT.ppr_wor_ucb(s, N, DELTA, prior=(1, 50), every=5) <= K_target).any())
        found = np.cumsum(s)[checks - 1]
        naive_false += bool(CT._certify_mask(found, checks, N, K_target, DELTA, False).any())
    slack = 3 * math.sqrt(DELTA * (1 - DELTA) / reps)
    assert ppr_false / reps <= DELTA + slack
    # monitoring the fixed-n hypergeometric bound is anti-conservative (07 F6: 0.168 at N=2000)
    assert naive_false / reps > DELTA + slack


def test_ppr_has_power_with_centred_prior():
    rng = np.random.default_rng(11)
    N, reps = 2000, 30
    pop = np.zeros(N, int)
    pop[:10] = 1                                             # true zeta 0.005, target 0.02
    hits = sum(bool((CT.ppr_wor_ucb(rng.permutation(pop)[:800], N, DELTA, prior=(1, 100),
                                    every=5) <= 40).any()) for _ in range(reps))
    assert hits / reps >= 0.9


# ------------------------------------------------------------------------------ derived bounds
def test_derived_recall_bound_algebra():
    assert math.isclose(CT.derived_recall_bound(150, 40), 1 - 40 / 190)   # 07 L6 example: 0.79
    assert round(CT.derived_recall_bound(150, 40), 2) == 0.79
    assert CT.derived_recall_bound(10, 0) == 1.0
    assert CT.derived_recall_bound(0, 5) == 0.0
    assert CT.derived_recall_bound(100, 10) > CT.derived_recall_bound(100, 20)
    assert CT.derived_recall_bound(200, 10) > CT.derived_recall_bound(100, 10)
    # worst case: all U residual bad units are missed defective images -> bound is attained
    A, missed = 37, 9
    assert math.isclose(1 - missed / (A + missed), CT.derived_recall_bound(A, missed))
    with pytest.raises(ValueError):
        CT.derived_recall_bound(-1, 3)


def test_regime_check():
    r = CT.regime_check(0.005, 0.015, 0.02)
    assert r.empty and r.regime == "low"
    r = CT.regime_check(0.01, 0.03, 0.02)
    assert (not r.empty) and r.possibly_empty
    assert CT.regime_check(0.25, 0.35, 0.02).regime == "high"
    assert CT.regime_check(0.06, 0.12, 0.02).regime == "medium"
    assert CT.regime_check(0.025, 0.035, 0.02).regime == "low"
    with pytest.raises(ValueError):
        CT.regime_check(0.3, 0.2, 0.02)


def test_rule_of_three():
    assert math.isclose(CT.rule_of_three(100), -math.log(0.05) / 100)
    assert abs(CT.rule_of_three(1000) - 0.003) < 1e-4
    assert CT.rule_of_three(90) >= 1 - 0.05 ** (1 / 90)          # conservative vs exact binomial
    assert CT.rule_of_three(1) == 1.0


# ------------------------------------------------------------------------------ manifest
def _manifest():
    rng = np.random.default_rng(0)
    masks = [rng.random((8, 8)) > 0.7 for _ in range(3)]
    return {"labels": masks, "risk": rng.random(3), "conventions": {"tol": 2, "merge_radius": 2},
            "targets": {"zeta": 0.02}, "delta_split": {"zeta": 0.05}, "R_max": 1, "seed": 1234}


def test_manifest_hash_stability_and_sensitivity():
    m = _manifest()
    h = CT.freeze_manifest(m)
    assert len(h) == 64 and int(h, 16) >= 0
    assert CT.freeze_manifest(_manifest()) == h                               # deterministic
    assert CT.freeze_manifest(dict(reversed(list(m.items())))) == h           # key order irrelevant
    m2 = _manifest()
    m2["labels"][1] = m2["labels"][1].copy()
    m2["labels"][1][3, 4] = ~m2["labels"][1][3, 4]                            # one pixel flipped
    assert CT.freeze_manifest(m2) != h
    m3 = _manifest()
    m3["labels"][0] = m3["labels"][0].astype(np.uint8)                        # dtype is registered
    assert CT.freeze_manifest(m3) != h
    m4 = _manifest()
    m4["risk"] = m4["risk"].copy()
    m4["risk"][0] = np.nextafter(m4["risk"][0], 2.0)                          # 1 ulp change
    assert CT.freeze_manifest(m4) != h
    for key, val in (("seed", 1235), ("R_max", 2), ("conventions", {"tol": 3, "merge_radius": 2}),
                     ("targets", {"zeta": 0.021}), ("delta_split", {"zeta": 0.025})):
        m5 = _manifest()
        m5[key] = val
        assert CT.freeze_manifest(m5) != h
    m6 = _manifest()
    m6["labels"] = m6["labels"][::-1]                                          # order matters
    assert CT.freeze_manifest(m6) != h
    with pytest.raises(TypeError):
        CT.freeze_manifest({"x": object()})


def test_audit_plan():
    p = CT.AuditPlan(N=2000, n_fixed=332, delta=0.05, K_target=40, seed=7,
                     delta_split={"zeta": 0.04, "phi": 0.01}, R_max=2,
                     labels_digest=CT.freeze_manifest({"labels": _manifest()["labels"]}))
    assert math.isclose(p.zeta_target, 0.02) and math.isclose(p.round_delta("zeta"), 0.02)
    d = p.digest()
    assert d == CT.AuditPlan(**{**p.__dict__}).digest()
    assert d != CT.AuditPlan(**{**p.__dict__, "seed": 8}).digest()
    with pytest.raises(ValueError):
        CT.AuditPlan(N=2000, n_fixed=332, delta=0.05, K_target=40, seed=7,
                     delta_split={"zeta": 0.04, "phi": 0.02})
    o1, o2 = CT.audit_order(100, 7), CT.audit_order(100, 7)
    assert np.array_equal(o1, o2) and sorted(o1.tolist()) == list(range(100))


def test_audit_plan_default_r_max():
    p = CT.AuditPlan(N=2000, n_fixed=332, delta=0.05, K_target=40, seed=7)
    assert p.R_max == 2 and math.isclose(p.round_delta("zeta"), 0.025)     # delta_r = delta / 2


# ------------------------------------------------------------------------------ S3 floor
def test_floored_inclusion_and_check_floor():
    rng = np.random.default_rng(3)
    s = rng.exponential(size=500)
    s[:50] = 0.0                                                            # no signal at all
    for gamma in (0.1, 0.3, 1.0):
        q = CT.floored_inclusion(s, gamma)
        assert math.isclose(q.sum(), 1.0) and q.min() >= gamma / 500 * (1 - 1e-12)
    assert np.allclose(CT.floored_inclusion(s, 1.0), 1 / 500)               # gamma = 1: uniform
    assert np.allclose(CT.floored_inclusion(np.zeros(10), 0.3), 0.1)        # all-zero scores
    q_bad = s / s.sum()                                                     # pure PPS, floor 0
    with pytest.raises(ValueError):
        CT.check_floor(q_bad, 0.3)
    with pytest.raises(ValueError):
        CT.check_floor(np.full(10, 0.2), 0.3)                               # not a distribution
    with pytest.raises(ValueError):
        CT.floored_inclusion(-s, 0.3)


# ------------------------------------------------------------------------------ CLI
def test_cli_certify(capsys):
    pytest.importorskip("torch")
    from autolabel import cli
    cli.main(["certify-plan", "--N", "2000", "--forecast", "10", "--target", "40", "--no-credit"])
    out = json.loads(capsys.readouterr().out)
    assert out["n"] == CT.plan_n(2000, 10, 40, DELTA, credit_found=False)[0]
    cli.main(["certify", "--found", "0", "--n", "300", "--N", "2000", "--registered-n", "300"])
    out = json.loads(capsys.readouterr().out)
    assert out["K_final_ucb"] == 18
    with pytest.raises(SystemExit) as e:
        cli.main(["certify", "--found", "0", "--n", "290", "--N", "2000", "--registered-n", "300"])
    assert e.value.code == 2
