"""Shared experiment layer for the theory-driven adaptive unmixing study.

All degradation, loaders and core algebra are IDENTICAL to W8_code so that
numbers remain comparable with the released paper tables.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "W8_code"))

from w8_core import (                                   # noqa: E402
    fcls, rmse, lfe_of, lfe_gate, neighbor_interp,
    weighted_masked_nmf, spectral_angle_credibility,
    trust_region_consensus, K_DEFAULT, W3_MIX,
)
import run_w8_tests as R                                # noqa: E402

EPS = 1e-9
CONDS = R.CONDITIONS[1:]
DATASETS = R.DATASETS


def make_case(Y, E, A_ref, H, Wb, seed, cond):
    """Exactly the degradation of run_w8_tests.evaluate()."""
    _, df, ns = cond
    L, N = Y.shape
    rng = np.random.default_rng(seed)
    sigma = np.linalg.norm(Y) / np.sqrt(N * L) / (10 ** (ns / 20))
    Yn = Y + rng.standard_normal(Y.shape) * sigma
    dead = rng.random(N) < df
    Yn[:, dead] = 0.0
    obs = (~dead).astype(float)
    Yimp = neighbor_interp(Yn, dead, L, H, Wb) if dead.any() else Yn
    A0 = fcls(E, Yimp)
    A0 = np.clip(A0, 1e-3, None)
    A0 /= A0.sum(0, keepdims=True)
    return Yn, obs, dead, A0, A_ref, float(sigma)


def estimate_sigma(Yn, obs):
    """Noise s.d. estimated ONLY from the observed cube (no ground truth).

    Along-wavelength first differences cancel the smooth spectral signal:
        sigma^2 ~ median_l,n (y[l+1,n]-y[l,n])^2 / 2
    Dead columns are excluded.
    """
    ob = obs > 0.5
    dif = np.diff(Yn[:, ob], axis=0)
    return float(np.sqrt(np.median(dif ** 2) / 2.0))


def core_trace(Yn, obs, E0, A0, dead, w, kmax, A_ref=None, want_E=False):
    """Run T_k and record per-step diagnostics.

    Returns (snaps, diag) where diag has, for k=1..kmax:
      resid : mean squared reconstruction residual over observed entries
      dE    : ||E_k - E_0||_F / ||E_0||_F          (endmember drift)
      fp    : ||E_k - E*_k||_F / ||E*_k||_F        (Theorem-1 fixed-point gap)
      rmse  : abundance RMSE (only if A_ref given; DIAGNOSTIC ONLY)
    """
    E = E0.copy()
    A = A0.copy()
    Ainit_dead = A0[:, dead].copy()
    Yd = Yn * obs[None, :]
    wgt = (obs * w)[None, :]
    ob = obs > 0.5
    N_obs = int(ob.sum())
    L = Yn.shape[0]
    snaps, Es = {}, {}
    diag = {"resid": {}, "dE": {}, "fp": {}, "rmse": {}}
    for k in range(1, kmax + 1):
        numE = Yd @ A.T
        denE = E @ (A * wgt) @ A.T + EPS
        E = np.clip(E * numE / denE, 1e-4, None)
        numA = E.T @ Yd
        denA = (E.T @ E) @ (A * wgt) + EPS
        A = A * numA / denA
        A = np.clip(A, 1e-4, None)
        A = A / A.sum(0, keepdims=True)
        A[:, dead] = Ainit_dead
        snaps[k] = A.copy()
        if want_E:
            Es[k] = E.copy()
        res = (Yd - E @ A)[:, ob]
        diag["resid"][k] = float((res ** 2).sum() / (L * N_obs))
        diag["dE"][k] = float(np.linalg.norm(E - E0) /
                              (np.linalg.norm(E0) + 1e-12))
        # Theorem 1 fixed point for the CURRENT A and W
        Gw = (A * wgt) @ A.T
        E_star = (Yd @ A.T) @ np.linalg.pinv(Gw)
        diag["fp"][k] = float(np.linalg.norm(E - E_star) /
                              (np.linalg.norm(E_star) + 1e-12))
        if A_ref is not None:
            diag["rmse"][k] = rmse(A, A_ref)
    return snaps, (Es if want_E else None), diag


def fuse(Ac, A3, A7, dead, A0, eta):
    """Consensus with scalar or per-pixel shrinkage eta (N,)."""
    A_alt = 0.5 * (A3 + A7)
    A_alt = np.clip(A_alt, 1e-4, None)
    A_alt = A_alt / A_alt.sum(0, keepdims=True)
    A_alt[:, dead] = A0[:, dead]
    e = np.asarray(eta)
    if e.ndim == 0:
        A = (1.0 - float(e)) * Ac + float(e) * A_alt
    else:
        A = (1.0 - e[None, :]) * Ac + e[None, :] * A_alt
    A = np.clip(A, 1e-4, None)
    A = A / A.sum(0, keepdims=True)
    A[:, dead] = A0[:, dead]
    return A


def eta_star(ec, eb, per_pixel=False):
    """Theorem 2 minimiser of ||(1-eta)ec + eta*eb||^2, clipped to [0,1]."""
    if not per_pixel:
        num = float((ec * ec).sum() - (ec * eb).sum())
        den = float(((ec - eb) ** 2).sum()) + 1e-12
        return float(np.clip(num / den, 0.0, 1.0))
    num = (ec * ec).sum(0) - (ec * eb).sum(0)
    den = ((ec - eb) ** 2).sum(0) + 1e-12
    return np.clip(num / den, 0.0, 1.0)


def load_all():
    out = {}
    for nm, ld in DATASETS:
        out[nm] = ld()
    return out
