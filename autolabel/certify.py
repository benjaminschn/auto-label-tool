"""CertAL-lite certification core: exact, design-based bounds for the delivered label pool.

This module implements PART of the CertAL-lite certificate set described in
docs/research/07_red_team.md section 6 (see "Not implemented" below):

* ``freeze_manifest`` / ``AuditPlan``  - L0: freeze and hash every registered quantity before the
  first audit draw (labels, risk scores, conventions, targets, delta split, R_max, seed).
* ``hypergeom_ucb``                    - exact one-sided upper confidence bound on the number of
  bad units in a finite population from a uniform sample WITHOUT replacement of fixed size n.
* ``plan_n`` / ``n90_table``           - exact power planning of the fixed audit size.
* ``certify_zeta``                     - L5: certificate for the delivered (rectified) pool.
* ``ppr_wor_ucb``                      - anytime-valid alternative (prior-posterior-ratio
  martingale, Waudby-Smith & Ramdas 2020) that may be monitored and stopped at any time.
* ``derived_recall_bound``             - L6: image-level recall bound derived from the zeta bound.
* ``regime_check``                     - L9: prevalence regime switch incl. the "empty" rule.
* ``rule_of_three``                    - bound on the prevalence of an undiscovered defect type.
* ``floored_inclusion`` / ``check_floor`` - S3 validity condition (ii) of 07 section 7 no. 7: a
  non-uniform inclusion distribution q must keep an S3-independent floor q_i >= gamma / N.

Not implemented (open, see docs/research/00_SYNTHESE.md section 4.7): a phi (false-component)
certificate and hence the two-sided zeta + phi statement of L5; the review stratum R as a separate
stratum with a combined bound; a registered extension of the fixed audit (``certify_zeta`` accepts
exactly one registered n); a non-uniform (q-weighted) audit itself; adversarial tests for "R outside
the frame", multiple rounds and verify-first. The only adversarial check in tests/ for this module is
monitoring of the fixed-n bound against the PPR confidence sequence.

Everything here is pure numpy/scipy and deterministic. The statistical statements hold only under
the assumptions listed in each docstring: uniform sampling with a registered seed, a registered
fixed n (for the fixed-n bound), frozen labels, and measurement of every audited unit under the
registered inspection protocol P. A certificate is a statistical statement about one delivered
pool (design-based), not a process certification in the sense of ISO/IEC 5259.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import betaln, gammaln
from scipy.stats import hypergeom

MANIFEST_VERSION = "certal-lite-manifest/1"


# --------------------------------------------------------------------------------------------- L0
def _canonical(obj: Any) -> Any:
    """Map ``obj`` to a JSON-serialisable structure with a unique, platform-independent encoding.

    * numpy arrays (e.g. label masks, risk scores) are encoded by dtype (``arr.dtype.str``, which
      includes byte order), shape and the sha256 of their C-contiguous bytes. A bool mask and a uint8
      mask with the same values therefore hash differently (the dtype is a registered quantity);
    * dicts are encoded with keys converted to ``str`` and sorted;
    * tuples and lists are both encoded as lists (order matters);
    * numpy scalars become Python scalars; floats are kept as floats (JSON uses ``repr``, which
      round-trips exactly); ``bytes`` become hex;
    * dataclasses (e.g. ``AuditPlan``) are encoded via ``dataclasses.asdict`` plus their class name.
    """
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        return {"__ndarray__": a.dtype.str, "shape": list(a.shape),
                "sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {"__dataclass__": type(obj).__name__, "fields": _canonical(dataclasses.asdict(obj))}
    if isinstance(obj, dict):
        items = [(str(k), _canonical(v)) for k, v in obj.items()]
        keys = [k for k, _ in items]
        if len(set(keys)) != len(keys):
            raise ValueError("manifest keys collide after conversion to str")
        return {k: v for k, v in sorted(items)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": bytes(obj).hex()}
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if not math.isfinite(f):
            return {"__float__": repr(f)}
        return f
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f"cannot freeze object of type {type(obj).__name__}")


def freeze_manifest(obj_dict: Dict[str, Any]) -> str:
    """Return the sha256 hex digest of a canonical serialisation of the registered quantities.

    ``obj_dict`` holds everything that must be fixed before the first audit draw (L0 in
    07_red_team.md section 6), e.g.::

        {"labels": [mask_0, mask_1, ...],        # delivered label masks (numpy arrays)
         "risk": risk_scores,                    # frozen risk/review scores (numpy array)
         "conventions": {"tol": 2, "merge_radius": 2},
         "targets": {"zeta": 0.02},
         "delta_split": {"zeta": 0.05}, "R_max": 2, "seed": 1234,
         "plan": audit_plan}                     # optional AuditPlan instance

    The digest is invariant to dict key order and changes if any array value, dtype or shape, any
    scalar, or the order of a list changes. Publish (or at least store out of reach of the audit
    process) the digest before drawing; recompute it at certification time and refuse on mismatch.
    """
    if not isinstance(obj_dict, dict):
        raise TypeError("freeze_manifest expects a dict")
    payload = {"version": MANIFEST_VERSION, "content": _canonical(obj_dict)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class AuditPlan:
    """Registered (frozen) quantities of one CertAL-lite audit.

    Attributes
    ----------
    N:            size of the audit frame (the whole delivered pool, review stratum included).
    n_fixed:      registered fixed audit size n for the fixed-n hypergeometric certificate.
    delta:        total error budget (consumer risk) of all certificates of this task.
    K_target:     registered target on the number of bad units after rectification
                  (zeta_target = K_target / N).
    seed:         seed of the uniform audit order (see ``audit_order``).
    R_max:        maximal number of label versions (retraining rounds) that may be audited; each
                  round gets ``delta_split[c] / R_max`` (Bonferroni over rounds). Default 2 (at most
                  one retraining, delta_r = delta / 2), as registered in 00_SYNTHESE.md section 6.
    delta_split:  absolute delta allocation per certificate component, must sum to <= delta.
    conventions:  counting conventions (e.g. tol, merge_radius, "bad unit" rule).
    targets:      further registered targets (e.g. {"zeta": 0.02, "phi": 0.05}).
    labels_digest, risk_digest: ``freeze_manifest`` digests of the delivered labels and the risk
                  scores (computed before the first draw).
    protocol:     short identifier/description of the inspection protocol P.
    """

    N: int
    n_fixed: int
    delta: float
    K_target: int
    seed: int
    R_max: int = 2
    delta_split: Dict[str, float] = field(default_factory=lambda: {"zeta": 0.05})
    conventions: Dict[str, Any] = field(default_factory=lambda: {"tol": 2, "merge_radius": 2})
    targets: Dict[str, float] = field(default_factory=dict)
    labels_digest: str = ""
    risk_digest: str = ""
    protocol: str = "P: blind full search, then overlay, union"

    def __post_init__(self):
        if not (isinstance(self.N, (int, np.integer)) and self.N >= 1):
            raise ValueError("N must be a positive integer")
        if not (1 <= self.n_fixed <= self.N):
            raise ValueError("n_fixed must be in [1, N]")
        if not (0.0 < self.delta < 1.0):
            raise ValueError("delta must be in (0, 1)")
        if not (0 <= self.K_target <= self.N):
            raise ValueError("K_target must be in [0, N]")
        if self.R_max < 1:
            raise ValueError("R_max must be >= 1")
        if any(v <= 0 for v in self.delta_split.values()):
            raise ValueError("delta_split entries must be positive")
        if sum(self.delta_split.values()) > self.delta * (1 + 1e-12):
            raise ValueError("delta_split must sum to <= delta")

    @property
    def zeta_target(self) -> float:
        return self.K_target / self.N

    def round_delta(self, component: str = "zeta") -> float:
        """Error budget of ``component`` for one audited label version: delta_split[c] / R_max."""
        return self.delta_split[component] / self.R_max

    def digest(self) -> str:
        """``freeze_manifest`` of all fields of this plan."""
        return freeze_manifest({"plan": self})


def audit_order(N: int, seed: int) -> np.ndarray:
    """Uniform random permutation of ``range(N)`` from the registered seed.

    The fixed-n audit uses the first ``n_fixed`` entries; the anytime (PPR) audit consumes the same
    stream in order. Either way the inclusion probabilities are known and uniform.
    """
    return np.random.default_rng(seed).permutation(int(N))


# ------------------------------------------------------------------------------- exact fixed-n
def _check_counts(found: int, n: int, N: int, delta: float):
    if not (0 <= n <= N) or N < 1:
        raise ValueError("need 0 <= n <= N and N >= 1")
    if not (0 <= found <= n):
        raise ValueError("need 0 <= found <= n")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")


def hypergeom_ucb(found: int, n: int, N: int, delta: float) -> int:
    """Exact one-sided (1 - delta) upper confidence bound on the NUMBER K of bad units.

    Setting: a finite population of N units of which an unknown number K is bad; a uniform random
    sample WITHOUT replacement of registered fixed size n contains ``found`` bad units (measured
    under protocol P). The bound is

        UCB = max{ K in [found, N - (n - found)] : P_hyp(X <= found | N, K, n) > delta },

    i.e. the set of K not rejected by the one-sided exact test. Because P(X <= x | K) is
    non-increasing in K this set is an interval and the maximum is found by bisection.
    Coverage: P_K(K > UCB) = P_K(F_K(X) <= delta) <= delta for every K (exact, not asymptotic).

    Valid ONLY for a fixed n registered before the first draw and uniform sampling. Recomputing the
    bound while sampling and stopping when it is small (monitoring) is NOT covered - use
    ``ppr_wor_ucb`` for that.
    """
    found, n, N = int(found), int(n), int(N)
    _check_counts(found, n, N, delta)
    lo, hi = found, N - (n - found)        # lo always satisfies cdf = 1 > delta
    if hypergeom.cdf(found, N, hi, n) > delta:
        return hi
    while hi - lo > 1:                     # invariant: cdf(lo) > delta >= cdf(hi)
        mid = (lo + hi) // 2
        if hypergeom.cdf(found, N, mid, n) > delta:
            lo = mid
        else:
            hi = mid
    return lo


def _certify_mask(xs: np.ndarray, n: np.ndarray, N: int, K_target: int, delta: float,
                  credit_found: bool) -> np.ndarray:
    """Boolean array: does observing x bad units in a sample of size n certify the target?

    Uses ucb(x) <= T  <=>  P(X <= x | K = T + 1) <= delta (monotonicity of the cdf in K), with
    T = x + K_target when audited bad units are corrected (credit_found) and T = K_target otherwise.
    """
    T = xs + K_target if credit_found else np.broadcast_to(np.asarray(K_target), xs.shape)
    Kt = T + 1
    feasible = Kt <= N
    cdf = hypergeom.cdf(xs, N, np.where(feasible, Kt, N), n)
    return np.where(feasible, cdf <= delta, True)


def certification_power(n: int, N: int, K_forecast: int, K_target: int, delta: float,
                        credit_found: bool = True) -> float:
    """Exact probability that the fixed-n test certifies the target when K_forecast units are bad.

    With ``credit_found=True`` (rectifying inspection, default) the certified statement is
    ``hypergeom_ucb(found) - found <= K_target`` (bad units remaining after correcting the audited
    ones); with ``credit_found=False`` it is ``hypergeom_ucb(found) <= K_target``.
    """
    n, N, K_forecast = int(n), int(N), int(K_forecast)
    xs = np.arange(max(0, n - (N - K_forecast)), min(n, K_forecast) + 1)
    pmf = hypergeom.pmf(xs, N, K_forecast, n)
    ok = _certify_mask(xs, np.full_like(xs, n), N, int(K_target), delta, credit_found)
    return float(np.sum(pmf * ok))


def plan_n(N: int, K_forecast: int, K_target: int, delta: float, power: float = 0.9,
           credit_found: bool = True, n_max: Optional[int] = None) -> Tuple[Optional[int], float]:
    """Smallest fixed audit size n with P(certify | K = K_forecast) >= power (exact hypergeometric).

    The certified statement is the one of ``certify_zeta`` (``credit_found=True``: bad units left
    after correcting the audited ones <= K_target) or, with ``credit_found=False``, K <= K_target.
    The forecast K_forecast only affects power, never validity: the certificate itself is computed
    from the observed count. A conservative (pessimistic) forecast is recommended.

    Every n in 1..n_max is checked (power is not guaranteed to be monotone in n because of the
    discreteness of the exact test), so the returned n is the exact minimum.

    Returns ``(n, achieved_power)``; ``(None, best_power)`` if no n <= n_max reaches ``power``.
    """
    N, K_forecast, K_target = int(N), int(K_forecast), int(K_target)
    if not (0 <= K_forecast <= N and 0 <= K_target <= N):
        raise ValueError("K_forecast and K_target must be in [0, N]")
    if not (0.0 < delta < 1.0 and 0.0 < power <= 1.0):
        raise ValueError("delta in (0,1) and power in (0,1] required")
    n_max = N if n_max is None else min(int(n_max), N)
    best = 0.0
    xs_all = np.arange(0, K_forecast + 1)
    block = 128
    for start in range(1, n_max + 1, block):
        ns = np.arange(start, min(start + block, n_max + 1))
        X, Nn = np.meshgrid(xs_all, ns)                       # shape (len(ns), K_forecast + 1)
        pmf = hypergeom.pmf(X, N, K_forecast, Nn)             # zero outside the support
        valid = (X <= Nn) & (Nn - X <= N - K_forecast)
        ok = _certify_mask(np.where(valid, X, 0), Nn, N, K_target, delta, credit_found)
        pw = np.sum(np.where(valid, pmf, 0.0) * ok, axis=1)
        best = max(best, float(pw.max()))
        hit = np.nonzero(pw >= power - 1e-12)[0]
        if hit.size:
            return int(ns[hit[0]]), float(pw[hit[0]])
    return None, best


def n90_table(N: int, K_targets: Sequence[int], K_forecasts: Sequence[int], delta: float,
              power: float = 0.9, credit_found: bool = True) -> Dict[Tuple[int, int], Optional[int]]:
    """Planning table {(K_target, K_forecast): smallest n with power >= ``power``} (see plan_n)."""
    return {(int(kt), int(kf)): plan_n(N, kf, kt, delta, power, credit_found)[0]
            for kt in K_targets for kf in K_forecasts}


class CertificateRefused(ValueError):
    """Raised when a certificate request violates the registered design."""


@dataclass(frozen=True)
class ZetaCertificate:
    """Result of ``certify_zeta``. All bounds hold jointly with probability >= 1 - delta."""

    N: int
    n: int
    found: int
    delta: float
    K_ucb: int                 # UCB on bad units in the pool at audit time
    K_final_ucb: int           # UCB on bad units after correcting the ``found`` audited ones
    zeta_final_ucb: float      # K_final_ucb / N
    K_target: Optional[int] = None
    certified: Optional[bool] = None   # K_final_ucb <= K_target (None if no target given)
    statement: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def certify_zeta(found: int, n: int, N: int, delta: float, n_fixed_registered: int,
                 K_target: Optional[int] = None) -> ZetaCertificate:
    """Certificate on the bad-unit rate zeta_final of the DELIVERED (rectified) pool.

    Design: before the first draw, n = ``n_fixed_registered`` was registered and frozen together
    with the labels (L0). A uniform sample without replacement of exactly n units of the whole pool
    (N units, review stratum included) was inspected under protocol P; ``found`` units were bad and
    have been corrected under P. Then, with probability >= 1 - delta,

        K_final <= K_final_ucb = hypergeom_ucb(found, n, N, delta) - found,
        zeta_final <= zeta_final_ucb = K_final_ucb / N,

    because K_final = K - found and P(K > hypergeom_ucb) <= delta. In acceptance-sampling terms this
    is an LTPD/LQ acceptance plan with consumer risk delta, used as Dodge-Romig rectifying
    inspection (the sampled defectives are replaced by correct units); it is NOT an AQL plan (AQL
    controls producer risk). The fixed-n bound must not be monitored: if n differs from the
    registered n the request is refused (``CertificateRefused``), since computing the bound at a
    data-dependent stopping time invalidates the coverage. Early stopping is allowed only to NOT
    certify. For an anytime-valid certificate use ``ppr_wor_ucb``.

    Assumptions (to be listed in every report): uniform sampling with registered seed, fixed n,
    frozen labels/manifest, every audited unit measured under protocol P (verify-first corrections
    do not count as measurements), delta is the budget of this round (delta_split / R_max).
    """
    found, n, N = int(found), int(n), int(N)
    _check_counts(found, n, N, delta)
    if int(n_fixed_registered) != n:
        raise CertificateRefused(
            f"audit size n={n} differs from the registered n={int(n_fixed_registered)}; the fixed-n "
            "hypergeometric bound is only valid at the registered n (no monitoring)")
    ucb = hypergeom_ucb(found, n, N, delta)
    kf = ucb - found
    cert = None if K_target is None else bool(kf <= int(K_target))
    stmt = (f"With confidence >= {1 - delta:.4g}: at most {kf} of {N} delivered units are bad "
            f"(zeta_final <= {kf / N:.4g}); LTPD/LQ plan, consumer risk {delta:.4g}, n={n}, "
            f"found={found} (corrected).")
    return ZetaCertificate(N=N, n=n, found=found, delta=float(delta), K_ucb=int(ucb),
                           K_final_ucb=int(kf), zeta_final_ucb=kf / N,
                           K_target=None if K_target is None else int(K_target),
                           certified=cert, statement=stmt)


# ------------------------------------------------------------------------------- anytime (PPR)
def _beta_binomial_logpmf(N: int, a: float, b: float) -> np.ndarray:
    K = np.arange(N + 1, dtype=np.float64)
    return (gammaln(N + 1) - gammaln(K + 1) - gammaln(N - K + 1)
            + betaln(K + a, N - K + b) - betaln(a, b))


def _log_falling(x: np.ndarray, k: int) -> np.ndarray:
    """log(x (x-1) ... (x-k+1)) for integer arrays x, -inf where x < k."""
    out = np.full(x.shape, -np.inf)
    ok = x >= k
    out[ok] = gammaln(x[ok] + 1.0) - gammaln(x[ok] - k + 1.0)
    return out


def ppr_wor_ucb(sequence_of_outcomes: Iterable[int], N: int, delta: float,
                prior: Tuple[float, float] = (1.0, 1.0), every: int = 1) -> np.ndarray:
    """Anytime-valid upper confidence sequence for the number K of bad units (sampling WITHOUT
    replacement), via the prior-posterior-ratio (PPR) martingale.

    Reference: I. Waudby-Smith and A. Ramdas, "Confidence sequences for sampling without
    replacement", NeurIPS 2020, arXiv:2006.04347.

    ``sequence_of_outcomes`` are the 0/1 outcomes (1 = bad under protocol P) of a uniform random
    sequence of DISTINCT units of a population of N units, in draw order. With a Beta-Binomial(N, a, b)
    prior pi_0 on K and the sequence likelihood
        L_t(K) = K^(k_t falling) (N-K)^(t-k_t falling)  (common factor N^(t falling) dropped),
    the ratio
        M_t(K) = (sum_K' pi_0(K') L_t(K')) / L_t(K)  = pi_0(K) / pi_t(K)
    is a non-negative martingale with M_0 = 1 under the true K. By Ville's inequality
    P(exists t: M_t(K_true) >= 1/delta) <= delta, so C_t = {K : M_t(K) < 1/delta} is a
    (1 - delta) confidence sequence, valid under optional stopping and continuous monitoring.

    Returns an array u of length t_max with u[t-1] = running UCB after draw t:
    u_t = min_{s <= t, s evaluated} max C_s (the running intersection is valid because the
    true K lies in every C_s simultaneously with probability >= 1 - delta). The set is evaluated
    after every ``every``-th draw and after the last draw; in between the previous value is carried
    (still valid). Certify "K <= K_target" at the first t with u[t-1] <= K_target.

    The prior (a, b) affects only power, never validity; centring it on a forecast (e.g.
    a = 1, b = 100 for a few % bad units) is much cheaper than the uniform prior (1, 1)
    (07_red_team.md section 5.2). If audited bad units are corrected, the bound on remaining
    bad units at time t is u[t-1] - k_t.
    """
    x = np.asarray(list(sequence_of_outcomes), dtype=np.int64)
    N = int(N)
    if x.ndim != 1 or np.any((x != 0) & (x != 1)):
        raise ValueError("outcomes must be a 1-d sequence of 0/1")
    if len(x) > N:
        raise ValueError("more draws than population units (draws must be distinct units)")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    a, b = float(prior[0]), float(prior[1])
    if a <= 0 or b <= 0:
        raise ValueError("Beta-Binomial prior parameters must be positive")
    every = max(1, int(every))
    T = len(x)
    out = np.empty(T, dtype=np.int64)
    if T == 0:
        return out
    Ks = np.arange(N + 1)
    logprior = _beta_binomial_logpmf(N, a, b)
    log_thr = math.log(1.0 / delta)
    cum = np.cumsum(x)
    current = N
    for t in range(1, T + 1):
        if t % every == 0 or t == T:
            k = int(cum[t - 1])
            logL = _log_falling(Ks, k) + _log_falling(N - Ks, t - k)
            ok = np.isfinite(logL)
            lm = logprior[ok] + logL[ok]
            mx = lm.max()
            logmarg = mx + math.log(np.exp(lm - mx).sum())
            inside = (logmarg - logL[ok]) < log_thr
            ucb_t = int(Ks[ok][inside].max()) if inside.any() else k
            current = min(current, ucb_t)
        out[t - 1] = current
    return out


# ------------------------------------------------------------------------------ derived bounds
def derived_recall_bound(A_verified_defective: int, U_ucb_residual_bad: float) -> float:
    """Lower bound on image-level recall rho_b derived from the residual-bad bound (L6).

        rho_b >= 1 - U / (A + U)

    Definitions and assumptions:
    * A = number of distinct delivered images that were verified to be defective AND whose labels
      were corrected under protocol P (from review and audit). These images are not missed.
    * U = valid upper confidence bound on the number of residual bad images in the delivered pool
      (e.g. ``certify_zeta(...).K_final_ucb``), where every defective image whose delivered label
      misses it (under the registered counting rule) counts as bad.
    * rho_b = (#defective images not missed) / (#defective images).
    Then #missed <= |B| <= U and #defective >= A + #missed, and x -> x / (A + x) is increasing, so
    1 - rho_b = #missed/#defective <= #missed/(A + #missed) <= U/(A + U). The bound holds on the
    same event as U (no extra delta) and needs no D_max. It is invalid if the A images were only
    "verified" by verify-first without correction under P.
    """
    A = float(A_verified_defective)
    U = float(U_ucb_residual_bad)
    if A < 0 or U < 0:
        raise ValueError("A and U must be non-negative")
    if U == 0:
        return 1.0
    return 1.0 - U / (A + U)


@dataclass(frozen=True)
class RegimeResult:
    regime: str                 # "high" | "medium" | "low"
    empty: bool                 # p_ucb <= zeta_target: an empty labeler passes the zeta target
    possibly_empty: bool        # p_lcb <= zeta_target < p_ucb
    implied_recall_lcb: float   # 1 - zeta_target / p_lcb (worst case over the prevalence interval)
    implied_recall_ucb: float   # 1 - zeta_target / p_ucb (best case)
    note: str = ""


def regime_check(p_lcb: float, p_ucb: float, zeta_target: float, high_recall: float = 0.8,
                 low_recall: float = 0.5) -> RegimeResult:
    """Regime switch (L9) before any human minute, from a confidence interval [p_lcb, p_ucb] on the
    prevalence p of defective images (from triage).

    A zeta certificate "zeta_final <= zeta_target" alone implies only an image recall of about
    1 - zeta_target / p (all residual bad units could be missed defective images). Hence:
    * empty = p_ucb <= zeta_target: a labeler that outputs nothing passes the target; the
      zeta certificate must be marked as empty;
    * "high": 1 - zeta_target / p_lcb >= high_recall -> zeta + phi + rho_b certificates;
    * "low":  empty, or 1 - zeta_target / p_ucb < low_recall -> full inspection with model support;
    * "medium" otherwise -> stratified statement (miss rate <= UCB in the no-signal stratum).
    The thresholds high_recall / low_recall are registered conventions (defaults 0.8 / 0.5 are a
    choice of this implementation, not derived results).
    """
    if not (0.0 <= p_lcb <= p_ucb <= 1.0):
        raise ValueError("need 0 <= p_lcb <= p_ucb <= 1")
    if not (0.0 <= zeta_target < 1.0):
        raise ValueError("zeta_target must be in [0, 1)")
    imp = lambda p: 1.0 - zeta_target / p if p > 0 else -math.inf
    lo, hi = imp(p_lcb), imp(p_ucb)
    empty = p_ucb <= zeta_target
    possibly_empty = (not empty) and p_lcb <= zeta_target
    if empty or hi < low_recall:
        regime = "low"
    elif lo >= high_recall:
        regime = "high"
    else:
        regime = "medium"
    note = ("zeta certificate is EMPTY (an empty labeler passes it)" if empty else
            "zeta certificate may be empty (prevalence interval reaches the target)"
            if possibly_empty else "")
    return RegimeResult(regime=regime, empty=empty, possibly_empty=possibly_empty,
                        implied_recall_lcb=lo, implied_recall_ucb=hi, note=note)


def floored_inclusion(scores: Sequence[float], gamma: float) -> np.ndarray:
    """Inclusion distribution q_i = (1 - gamma) s_i / sum(s) + gamma / N with an S3-independent floor.

    Validity condition (ii) of 07_red_team.md section 7 no. 7: errors of the risk model S3 cost only
    efficiency (never validity) only if every unit keeps q_i >= gamma / N with a gamma fixed before
    the audit and not chosen from S3. ``scores`` must be non-negative and frozen (F_0-measurable);
    ``gamma`` in (0, 1]; gamma = 1 gives the uniform design. If all scores are zero, the uniform design
    is returned. The result is checked with ``check_floor``.

    This only builds and checks q. A q-weighted audit with a valid certificate (Satz 1') is NOT
    implemented in this module.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or s.size < 1:
        raise ValueError("scores must be a non-empty 1-d array")
    if np.any(~np.isfinite(s)) or np.any(s < 0):
        raise ValueError("scores must be finite and non-negative")
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1]")
    N = s.size
    tot = s.sum()
    base = s / tot if tot > 0 else np.full(N, 1.0 / N)
    q = (1.0 - gamma) * base + gamma / N
    check_floor(q, gamma)
    return q


def check_floor(q: Sequence[float], gamma: float, tol: float = 1e-12) -> None:
    """Raise ``ValueError`` unless q is a probability vector with q_i >= gamma / N for all i.

    Call this on every non-uniform inclusion distribution before the first draw (07 section 7 no. 7,
    condition (ii)). ``tol`` is a relative tolerance for floating-point rounding.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1 or q.size < 1:
        raise ValueError("q must be a non-empty 1-d array")
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1]")
    N = q.size
    if np.any(~np.isfinite(q)) or abs(q.sum() - 1.0) > 1e-9:
        raise ValueError("q must be a finite probability vector (sum 1)")
    floor = gamma / N
    if q.min() < floor * (1.0 - tol):
        raise ValueError(f"inclusion floor violated: min q = {q.min():.3e} < gamma/N = {floor:.3e}")


def rule_of_three(n_uniform: int, delta: float = 0.05) -> float:
    """Upper (1 - delta) bound on the prevalence of a defect type that was never found in
    ``n_uniform`` uniformly sampled units: -ln(delta) / n  (about 3/n for delta = 0.05).

    Valid for uniform sampling with or without replacement: P(0 found) <= (1 - p)^n <= exp(-p n),
    which is <= delta for p >= -ln(delta)/n. Slightly conservative; the exact binomial bound is
    1 - delta**(1/n). Only uniformly sampled units with known inclusion probabilities count
    (not reviewed or risk-selected units).
    """
    n = int(n_uniform)
    if n < 1:
        raise ValueError("n_uniform must be >= 1")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    return min(1.0, -math.log(delta) / n)
