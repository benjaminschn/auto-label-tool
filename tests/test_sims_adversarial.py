"""Adversarial unit test for the betting LCB prototype in docs/research/sims/sim_recall_cert.py.

07_red_team.md F4 / section 7 no. 6: with the bet cap evaluated at the CURRENT draw (v1, 'per_draw')
the betting test is not a supermartingale test and certifies a too-high recall on an adversarial pool
(all misses in units without signal, q at the floor gamma/N). The predictable cap (v2) must keep the
false-certificate rate at or below delta. Small number of repetitions (CPU, seconds); the full numbers
come from running the sim script itself.
"""
import importlib.util
import os

import numpy as np
import pytest

_SIM = os.path.join(os.path.dirname(__file__), "..", "docs", "research", "sims", "sim_recall_cert.py")


def _load_sim():
    spec = importlib.util.spec_from_file_location("sim_recall_cert", _SIM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("n_distinct", [300])
def test_predictable_cap_valid_per_draw_cap_invalid_on_adversarial_pool(n_distinct):
    sim = _load_sim()
    D, M, fH, fM, fD, s2, q = sim.make_adversarial_pool()
    rho = 1 - M.sum() / D.sum()                                    # true recall 0.8
    assert q.min() >= 0.3 / len(q) * (1 - 1e-12)                   # misses sit at the floor
    reps, delta = 20, 0.05
    rng = np.random.default_rng(11)
    false_pred, false_v1 = 0, 0
    for _ in range(reps):
        lcb, _, viol = sim.betting_lcb(D, M, fH, fM, fD, s2, n_distinct, delta, rng, q=q,
                                       cap="predictable")
        assert not viol                                            # b = r * D_MAX holds by construction
        false_pred += lcb > rho
        lcb1, _, _ = sim.betting_lcb(D, M, fH, fM, fD, s2, n_distinct, delta, rng, q=q,
                                     cap="per_draw")
        false_v1 += lcb1 > rho
    # predictable cap: false-certificate rate <= delta (0 of 20 expected; allow 1 for MC noise)
    assert false_pred <= 1
    # v1 cap: clearly invalid (07 F4 reports 0.63-0.99)
    assert false_v1 >= reps // 2


def test_runtime_bound_check_invalidates_certificate():
    sim = _load_sim()
    D, M, fH, fM, fD, s2, q = sim.make_adversarial_pool()
    D2, M2 = D.copy(), M.copy()
    D2[160:200] = M2[160:200] = sim.DMAX + 5                       # violates the hard bound D_i <= D_MAX
    rng = np.random.default_rng(5)
    res = [sim.betting_lcb(D2, M2, fH, fM, fD, s2, 300, 0.05, rng, q=q) for _ in range(8)]
    assert any(v for _, _, v in res)                               # a drawn violating unit is detected
    assert all(lcb == 0.0 for lcb, _, v in res if v)               # violation -> no certificate
