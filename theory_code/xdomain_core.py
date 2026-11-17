"""Cross-domain estimators for incomplete linear unmixing.

NOTHING in this file comes from the hyperspectral-unmixing literature.  The
three ingredients are imported from other fields:

  (1) statistical decision theory  -- Stein's unbiased risk estimate (SURE),
      James--Stein shrinkage, empirical-Bayes hierarchical pooling;
  (2) graph / image denoising      -- leave-one-out non-local (spatial) pooling
      as a Gauss--Markov prior mean, a.k.a. Wiener filtering in the graph
      Fourier basis;
  (3) robust statistics            -- bounded-influence Huber M-estimation for
      heavy-tailed band noise.

Model (well-posed, exactly the Round-12 reference model):

    y_n = E a_n + eps_n ,  eps_n ~ N(0, sigma^2 I_L),  a_n in Delta^{P-1},
    observed columns only; dead columns carry no likelihood term.

Strong baseline: EXACT fully-constrained least squares (KKT support search),
not the clip+normalise heuristic used by the released W8 code.

Key structural fact exploited here: the ASC constraint 1^T a = 1 removes one
degree of freedom.  In the tangent space of the simplex the whitened MLE has
dimension d = P-1, so Stein dominance holds exactly when P >= 4 (d >= 3).
Samson (P=3) is a built-in falsification test: JS must give ~zero gain there.
"""
import itertools

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------
# Exact FCLS: the strong baseline (KKT / active-set, P <= 5)
# --------------------------------------------------------------------------

def build_supports(P):
    return [tuple(S) for k in range(1, P + 1)
            for S in itertools.combinations(range(P), k)]


def exact_fcls(G, B, y2, supports):
    """Exact solution of min ||y-Ea||^2 s.t. a>=0, 1'a=1, per column.

    G=E'E (P,P), B=E'Y (P,N), y2=sum_l Y_ln^2 (N,).
    Returns (A, residual, chosen_support_index).
    """
    P, N = B.shape
    best_res = np.full(N, np.inf)
    best_A = np.zeros((P, N))
    chosen = np.full(N, -1, dtype=int)
    for ci, S in enumerate(supports):
        idx = np.array(S)
        Gi = np.linalg.inv(G[np.ix_(idx, idx)])
        gi1 = Gi @ np.ones(len(S))
        denom = float(gi1.sum())
        U = Gi @ B[idx]
        lam = (U.sum(0) - 1.0) / denom
        aS = U - gi1[:, None] * lam[None, :]
        ok = (aS > 0).all(0)
        if not ok.any():
            continue
        A = np.zeros((P, N))
        A[idx] = aS
        Ga_minus_b = G @ A - B
        outside = np.setdiff1d(np.arange(P), idx)
        if outside.size:
            ok &= (Ga_minus_b[outside] + lam[None, :] >= -1e-9).all(0)
        res = y2 - 2.0 * (B * A).sum(0) + (A * (G @ A)).sum(0)
        take = ok & (res < best_res)
        if take.any():
            best_res[take] = res[take]
            best_A[:, take] = A[:, take]
            chosen[take] = ci
    bad = ~np.isfinite(best_res)
    if bad.any():
        U = np.linalg.pinv(G) @ B[:, bad]
        A = np.maximum(U, 1e-8)
        A /= A.sum(0, keepdims=True)
        best_A[:, bad] = A
    return best_A, best_res, chosen


def fcls_from_cube(Yd, E, supports=None):
    P = E.shape[1]
    if supports is None:
        supports = build_supports(P)
    G = E.T @ E
    B = E.T @ Yd
    y2 = (Yd ** 2).sum(0)
    A, _, _ = exact_fcls(G, B, y2, supports)
    return A, G, B, y2


# --------------------------------------------------------------------------
# Tangent space of the ASC simplex and Stein whitening
# --------------------------------------------------------------------------

def tangent_basis(P):
    """Helmert basis C (P, P-1): 1' C = 0, C'C = I_{P-1}."""
    C = np.zeros((P, P - 1))
    for j in range(P - 1):
        q = j + 1
        C[:q, j] = 1.0 / np.sqrt(q * (q + 1))
        C[q, j] = -q / np.sqrt(q * (q + 1))
    return C


def tangent_metric(G, C):
    """Symmetric sqrt / inverse-sqrt of the tangent-space Hessian C' G C."""
    H = C.T @ G @ C
    ev, U = np.linalg.eigh(H)
    ev = np.maximum(ev, EPS)
    W = (U * np.sqrt(ev)) @ U.T
    Wi = (U / np.sqrt(ev)) @ U.T
    return W, Wi


def fisher_trace(G, C):
    """tr(Sigma)/sigma^2 where Sigma is the MLE covariance on the simplex.

    a = a_0 + C theta, theta_hat ~ N(theta, sigma^2 (C'GC)^{-1}), so
        Sigma = sigma^2 C H^{-1} C'  and  tr(Sigma) = sigma^2 tr(H^{-1}).
    This is the V term in the Euclidean SURE identity.
    """
    H = C.T @ G @ C
    return float(np.trace(np.linalg.inv(H)))


def sigma_from_residual(residual, L, d):
    """Unbiased noise-variance MLE from the exact-FCLS residual.

    For an interior solution the residual is sigma^2 * chi^2_{L-d} with
    d = P-1 free abundance parameters, hence sigma^2 = mean(res)/(L-d).
    No ground truth is used.  (Boundary/active-set pixels are a diagnostic
    caveat, quantified by interior_fraction in the experiment log.)
    """
    return float(np.sqrt(max(float(np.mean(np.maximum(residual, 0.0))), EPS)
                        / max(L - d, 1)))


def gdist_sq(X, G):
    """Column-wise squared G-weighted distance (P,N) -> (N,)."""
    return np.einsum("pn,pq,qn->n", X, G, X)


# --------------------------------------------------------------------------
# Leave-one-out spatial pooling = Gauss--Markov prior mean (graph Wiener)
# --------------------------------------------------------------------------

def _shift_sum(X3, o3):
    """Sum over the 4-neighbourhood, excluding the centre itself."""
    s = np.zeros_like(X3)
    c = np.zeros_like(o3, dtype=float)
    # vertical
    s[:, :-1] += X3[:, 1:]
    c[:-1] += o3[1:]
    s[:, 1:] += X3[:, :-1]
    c[1:] += o3[:-1]
    # horizontal
    s[:, :, :-1] += X3[:, :, 1:]
    c[:, :-1] += o3[:, 1:]
    s[:, :, 1:] += X3[:, :, :-1]
    c[:, 1:] += o3[:, :-1]
    return s, c


def spatial_target(Ahat, ob, H, Wb, P, mode="mean", diag=None):
    """LOO prior mean t_n built ONLY from other observed pixels.

    Because t_n never uses y_n, it is independent of eps_n; this is exactly
    the condition under which SURE is an unbiased risk estimate and under
    which Stein dominance holds for an arbitrary (even misspecified) target.

    mode = 'mean'  : 4-neighbour average          (graph Wiener / EB pooling)
           'median': 4-neighbour coordinate median (robust pooling)
           'global': global observed mean         (ablation, no spatial info)
    """
    N = Ahat.shape[1]
    o = ob.reshape(1, H, Wb)
    Ao = (Ahat * ob[None, :]).reshape(P, H, Wb)
    if mode == "global":
        t = Ao.sum((1, 2))[:, None] / max(ob.sum(), 1.0)
        return np.repeat(t, N, axis=1), np.full(N, ob.sum(), dtype=float)
    if mode == "median":
        t = np.zeros((P, H, Wb))
        cnt = np.zeros((H, Wb))
        for p in range(P):
            stack, c = [], None
            A3 = Ao[p]
            for dx, dy, sl_a, sl_b in (
                    (-1, 0, (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                    (1, 0, (slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                    (0, -1, (slice(None), slice(None), slice(None, -1)), (slice(None), slice(None), slice(1, None))),
                    (0, 1, (slice(None), slice(None), slice(1, None)), (slice(None), slice(None), slice(None, -1)))):
                M = np.zeros_like(A3)
                M[sl_a] = A3[sl_b]
                O = np.zeros_like(o[0], dtype=float)
                O[sl_a] = o[0][sl_b]
                stack.append(M)
                if c is None:
                    c = O
                else:
                    c = c + O
            V = np.stack(stack, 0)                     # (4,H,W)
            valid = np.stack([
                np.pad((o[0] > .5)[1:], ((0, 1), (0, 0))),
                np.pad((o[0] > .5)[:-1], ((1, 0), (0, 0))),
                np.pad((o[0] > .5)[:, 1:], ((0, 0), (0, 1))),
                np.pad((o[0] > .5)[:, :-1], ((0, 0), (1, 0)))], 0)
            V[~valid] = np.nan
            with np.errstate(invalid="ignore"):
                t[p] = np.nanmedian(V, axis=0)
            t[p] = np.nan_to_num(t[p], nan=0.0)
            cnt = c
        t = t.reshape(P, N)
        t = np.maximum(t, 0.0)
        s = t.sum(0, keepdims=True)
        t = np.where(s > EPS, t / np.maximum(s, EPS),
                     np.ones((P, 1)) / P)
    else:
        s, c = _shift_sum(Ao, o[0])
        t = s.reshape(P, N) / np.maximum(c.reshape(1, N), 1.0)
        cnt = c.reshape(N)
    # isolated pixels fall back to the global observed mean
    gm = Ao.sum((1, 2))[:, None] / max(ob.sum(), 1.0)
    iso = cnt <= 0
    if iso.any():
        t[:, iso] = gm
    if diag is not None:
        diag["target_neighbours_mean"] = float(cnt[ob > .5].mean())
        diag["isolated_fraction"] = float(iso.mean())
    return t, cnt


# --------------------------------------------------------------------------
# Cross-fitted non-local means prior (image denoising + causal cross-fitting)
# --------------------------------------------------------------------------

def _shift2(X3, dy, dx):
    """Shift a (P,H,W) array so that out [:,i,j] takes X3[:,i+dy,j+dx]."""
    P, H, Wb = X3.shape
    out = np.zeros_like(X3)
    o = np.zeros((H, Wb))
    i0, i1 = max(0, -dy), min(H, H - dy)
    j0, j1 = max(0, -dx), min(Wb, Wb - dx)
    if i1 <= i0 or j1 <= j0:
        return out, o
    out[:, i0:i1, j0:j1] = X3[:, i0 + dy:i1 + dy, j0 + dx:j1 + dx]
    o[i0:i1, j0:j1] = 1.0
    return out, o


def nlm_target(Ahat, center, ob, H, Wb, P, radius=3, h2=None,
               h_mode="noise", V=1.0, kappa=4.0, diag=None):
    """Leave-one-out non-local-means prior mean in abundance space.

    w_nm = exp(-||ahat_m - center_n||^2 / (2 h^2)),  m in a (2r+1)^2 window,
    m != n, m observed.  center_n is the LOO 4-neighbour mean, hence both
    center_n and every ahat_m (m != n) are independent of eps_n.  Therefore
    t_n _|_ eps_n and SURE stays unbiased / Theorem 8 still applies.

    All averaged points lie in the simplex, so t_n is in the simplex without
    any projection.
    """
    N = Ahat.shape[1]
    A3 = (Ahat * ob[None, :]).reshape(P, H, Wb)
    o = (ob > .5).reshape(1, H, Wb)
    c3 = center.reshape(P, H, Wb)
    offs = [(dy, dx) for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1) if (dy, dx) != (0, 0)]
    shifts, masks, dists = [], [], []
    for dy, dx in offs:
        S, M = _shift2(A3, dy, dx)
        shifts.append(S.reshape(P, N))
        masks.append(M.reshape(N) * ob)
        dists.append(((S - c3) ** 2).sum(0).reshape(N))
    D = np.stack(dists, 0)                      # (K,N)
    if h_mode == "noise":
        # E||ahat_m - center_n||^2 ~ 1.25 * tr(Sigma) under pure noise,
        # so a multiple of V (not V/P) is the right bandwidth scale.
        h2 = kappa * V
    elif h_mode == "median":
        h2 = float(np.median(D[np.stack(masks, 0) > .5]))
    elif h_mode == "fixed":
        h2 = float(h2)
    else:
        raise ValueError(h_mode)
    h2 = max(float(h2), 1e-10)
    Wt = np.exp(-D / (2.0 * h2)) * np.stack(masks, 0)
    num = np.zeros((P, N))
    for k, S in enumerate(shifts):
        num += Wt[k][None, :] * S
    den = Wt.sum(0)
    t = num / np.maximum(den[None, :], 1e-12)
    bad = den <= 1e-12
    if bad.any():
        t[:, bad] = center[:, bad]
    if diag is not None:
        diag["nlm_h2"] = h2
        diag["nlm_eff_neighbours"] = float(
            (Wt.sum(0) ** 2 / np.maximum((Wt ** 2).sum(0), 1e-12))[ob > .5].mean())
        diag["nlm_isolated"] = float(bad.mean())
    return t


# --------------------------------------------------------------------------
# SURE-adaptive shrinkage in the whitened tangent space
# --------------------------------------------------------------------------

def sw_shrink(Ahat, t, G, C, sigma2, ob, mode="sure", lam_override=None,
              diag=None):
    """Stein--Wiener estimator.

    In whitened tangent coordinates z_n ~ N(mu_n, I_d), d=P-1,
      hat_mu = (1-lam) z  <=>  hat_a = (1-lam) Ahat + lam t .
    SURE(lam) = d + lam^2 ||z||^2 - 2 lam d  is an unbiased estimate of the
    whitened risk; minimising gives lam* = d / mean_n ||z_n||^2.

    mode:
      'none'   : lam=0, returns exact FCLS (strong baseline)
      'sure_e' : Euclidean SURE, lam = clip(V/mean||Ahat-t||^2, 0, 1),
                 V = sigma^2 tr((C'GC)^{-1}).  NO ground truth.  Matches
                 the evaluation metric and dominates FCLS for any P.
      'sure'   : Fisher/whitened SURE, lam = clip(d/mean||z||^2,0,1).
      'js'     : lam_n = clip((d-2)/||z_n||^2, 0, 1), certifiable Stein
                 dominance when d>=3, exactly zero when d=2.
      'prior'  : lam=1 (pure prior mean, ablation)
    """
    P, N = Ahat.shape
    d = P - 1
    ob = ob > .5
    # Fisher-metric whitened squared distance
    r2_all = gdist_sq(Ahat - t, G) / max(sigma2, EPS)
    rbar = float(r2_all[ob].mean()) if ob.any() else 1.0
    # Euclidean squared distance (the metric we are actually scored on)
    e2_all = ((Ahat - t) ** 2).sum(0)
    ebar = float(e2_all[ob].mean()) if ob.any() else 1.0
    V = fisher_trace(G, C) * max(sigma2, EPS)
    if mode == "none":
        lam = np.zeros(N)
    elif mode == "prior":
        lam = np.ones(N)
    elif mode == "js":
        lam = np.clip(float(d - 2) / np.maximum(r2_all, EPS), 0.0, 1.0)
    elif mode == "sure":
        lam = np.full(N, np.clip(float(d) / max(rbar, EPS), 0.0, 1.0))
    elif mode == "sure_e":
        lam = np.full(N, np.clip(V / max(ebar, EPS), 0.0, 1.0))
    else:
        raise ValueError(mode)
    if lam_override is not None:
        lam = np.full(N, float(lam_override))
    A = (1.0 - lam[None, :]) * Ahat + lam[None, :] * t
    # convex combination of two simplex points is again in the simplex:
    # no projection, no clip, no fixed-point change.
    if diag is not None:
        diag["d"] = d
        diag["sigma2"] = float(sigma2)
        diag["r2_mean"] = rbar
        diag["e2_mean"] = ebar
        diag["V"] = V
        diag["lam_sure_e"] = float(np.clip(V / max(ebar, EPS), 0.0, 1.0))
        diag["lam_mean"] = float(lam.mean())
        diag["lam_js_mean"] = float(np.clip(float(d - 2) /
                                           np.maximum(r2_all, EPS), 0, 1).mean())
    return A, lam


def sure_lambda_grid(Ahat, t, G, C, sigma2, ob, P, Atrue=None,
                     metric="euclid"):
    """SURE risk curve over a lam grid; also the true risk when Atrue given.

    metric='euclid': SURE(lam) = lam^2 mean||y-t||^2 - 2 lam V + V,
                     V = sigma^2 tr((C'GC)^{-1}), unbiased for Euclidean risk.
    metric='fisher': SURE(lam) = d + lam^2 mean||z||^2 - 2 lam d,
                     unbiased for the whitened (Fisher) risk.
    """
    grid = np.linspace(0.0, 1.0, 21)
    d = P - 1
    ob = ob > .5
    V = fisher_trace(G, C) * max(sigma2, EPS)
    if metric == "euclid":
        rbar = float(((Ahat - t) ** 2).sum(0)[ob].mean())
        sure = (grid ** 2) * rbar - 2.0 * grid * V + V
    else:
        rbar = float((gdist_sq(Ahat - t, G) / max(sigma2, EPS))[ob].mean())
        sure = d + (grid ** 2) * rbar - 2.0 * grid * d
    out = {"grid": grid.tolist(), "sure": sure.tolist(),
           "sure_argmin": float(grid[int(np.argmin(sure))]),
           "r2_mean": rbar, "V": V, "metric": metric}
    if Atrue is not None:
        risk = []
        for lam in grid:
            A = (1 - lam) * Ahat + lam * t
            X = A - Atrue
            if metric == "euclid":
                risk.append(float((X ** 2).sum(0)[ob].mean()))
            else:
                risk.append(float(gdist_sq(X, G)[ob].mean() /
                                  max(sigma2, EPS)))
        out["true_risk"] = risk
        out["true_argmin"] = float(grid[int(np.argmin(risk))])
    return out


# --------------------------------------------------------------------------
# Huber M-estimation (robust statistics) for heavy-tailed band noise
# --------------------------------------------------------------------------

def huber_fcls(Yd, E, obs, delta, iters=3, supports=None, d_band=None):
    """IRLS Huber ASC-constrained estimator, per-pixel band weights.

    min sum_l rho_delta(y_l - (Ea)_l), a in simplex, with rho the Huber loss.
    The influence function is bounded by delta; LS is the delta->inf limit.
    """
    P = E.shape[1]
    L, N = Yd.shape
    if supports is None:
        supports = build_supports(P)
    if d_band is None:
        d_band = np.ones(L)
    db = d_band[:, None]
    A = np.full((P, N), 1.0 / P)
    ob = obs > .5
    for _ in range(iters):
        R = Yd - E @ A
        wl = np.minimum(1.0, delta / np.maximum(np.abs(R), EPS)) * db
        # A shared band-wise weight keeps one SPD normal matrix G for the
        # whole cube, so the KKT support search stays exact and vectorised.
        # This is a band-wise Huber M-estimator (exact per-pixel Huber would
        # need N separate support searches); the influence bound still holds.
        wg = np.median(wl[:, ob], axis=1) if ob.any() else wl.mean(1)
        Ew = E * wg[:, None]
        G = E.T @ Ew
        B = Ew.T @ Yd
        y2 = (wg[:, None] * (Yd ** 2)).sum(0)
        A, _, _ = exact_fcls(G, B, y2, supports)
        A[:, ~ob] = 1.0 / P
    return A


# --------------------------------------------------------------------------
# Per-pixel Huber M-estimator with EXACT ASC/KKT support search
# --------------------------------------------------------------------------

def huber_exact_fcls(Yd, E, obs, delta, iters=2, supports=None,
                     d_band=None, A_init=None):
    """Robust ASC-constrained M-estimator with bounded influence.

    min_a sum_l rho_delta(y_l - (Ea)_l),  a in simplex, rho = Huber loss.
    IRLS: w_ln = min(1, delta/|r_ln|).  Unlike the band-wise version, the
    weights are PER PIXEL, so a spike in one pixel is down-weighted there
    without discarding the band for the whole cube.  The active-set/KKT
    search is solved in batch over N pixels with batched (N,P,P) inverses.
    """
    L, N = Yd.shape
    P = E.shape[1]
    if supports is None:
        supports = build_supports(P)
    if d_band is None:
        d_band = np.ones(L)
    ob = obs > .5
    if A_init is None:
        A_init, _, _ = exact_fcls(E.T @ E, E.T @ Yd, (Yd ** 2).sum(0),
                                  supports)
    A = A_init.copy()
    eye = np.eye(P)[None, :, :]
    for _ in range(iters):
        R = Yd - E @ A
        wl = np.minimum(1.0, delta / np.maximum(np.abs(R), EPS))
        wl = wl * d_band[:, None]
        Gw = np.einsum("ln,li,lj->nij", wl, E, E, optimize=True)
        bw = np.einsum("ln,li->ni", wl * Yd, E, optimize=True)
        y2w = (wl * (Yd ** 2)).sum(0)
        best_res = np.full(N, np.inf)
        best_A = A.T.copy()                 # (N,P), row-wise assignment below
        for S in supports:
            idx = np.array(S)
            k = len(S)
            Gs = Gw[:, idx[:, None], idx[None, :]] + 1e-9 * np.eye(k)
            Gi = np.linalg.inv(Gs)
            gi1 = Gi @ np.ones(k)
            den = gi1.sum(1)
            U = np.einsum("nij,nj->ni", Gi, bw[:, idx])
            lam = (U.sum(1) - 1.0) / np.maximum(den, EPS)
            aS = U - gi1 * lam[:, None]
            ok = (aS > 0).all(1)
            if not ok.any():
                continue
            Ac = np.zeros((N, P))
            Ac[:, idx] = aS
            Ga = np.einsum("nij,nj->ni", Gw, Ac)
            outside = np.setdiff1d(np.arange(P), idx)
            if outside.size:
                ok &= (Ga[:, outside] - bw[:, outside]
                       + lam[:, None] >= -1e-9).all(1)
            res = (y2w - 2.0 * (bw * Ac).sum(1)
                   + (Ac * Ga).sum(1))
            take = ok & (res < best_res)
            if take.any():
                best_res[take] = res[take]
                best_A[take] = Ac[take]
        bad = ~np.isfinite(best_res)
        if bad.any():
            best_A[bad] = A.T[bad]
        A = best_A.T
        A[:, ~ob] = 1.0 / P
    return A


def gls_exact_fcls(Yd, E, d_band, supports, A_dead=None, dead=None):
    """ASC-constrained Gauss-Markov (GLS) estimator with band precisions d."""
    Ew = E * d_band[:, None]
    G = E.T @ Ew
    B = Ew.T @ Yd
    y2 = (d_band[:, None] * (Yd ** 2)).sum(0)
    A, res, ch = exact_fcls(G, B, y2, supports)
    if dead is not None and A_dead is not None and dead.any():
        A[:, dead] = A_dead[:, dead]
    return A, G, B, y2


def band_precision_from_residual(Yd, E, A, ob, clip=(0.05, 20.0)):
    """GT-free band precision estimate (robust median of squared residuals)."""
    R = Yd - E @ A
    v = np.median(R[:, ob] ** 2, axis=1)
    d = 1.0 / np.maximum(v, 1e-10)
    d = np.clip(d / d.mean(), clip[0], clip[1])
    return d / d.mean()


def gm_predicted_risk_gap(E, C, sigma2, d_band):
    """Closed-form Gauss-Markov risk dominance on the ASC tangent space.

    For y ~ N(Ea, sigma^2 diag(1/d)), the ASC-constrained OLS covariance is
        V_ols = sigma^2 H^{-1} (C' E' diag(1/d) E C) H^{-1},  H = C' E'E C,
    while the constrained GLS/BLUE covariance is
        V_gls = sigma^2 (C' E' diag(d) E C)^{-1} .
    Gauss-Markov gives V_gls <= V_ols (Loewner), equality iff d ~ 1.
    Returns (trace(V_ols), trace(V_gls), relative predicted gain).
    """
    H = C.T @ (E.T @ E) @ C
    Hi = np.linalg.inv(H)
    M = C.T @ (E.T @ (E / d_band[:, None])) @ C
    Hg = C.T @ (E.T @ (d_band[:, None] * E)) @ C
    v_ols = float(np.trace(Hi @ M @ Hi) * sigma2)
    v_gls = float(np.trace(np.linalg.inv(Hg)) * sigma2)
    return v_ols, v_gls, (v_ols - v_gls) / max(v_ols, EPS)


# --------------------------------------------------------------------------
# One-shot end-to-end estimator
# --------------------------------------------------------------------------

def sw_unmix(Yn, obs, E, H, Wb, sigma2, mode="sure", target_mode="mean",
             Ahat=None, supports=None, diag=None):
    """Full cross-domain estimator on one degraded cube.

    Returns abundance map (P,N).  Dead pixels receive the prior mean t_n,
    which is the posterior mean when the likelihood term is absent.
    """
    Yd = Yn * obs[None, :]
    P, N = E.shape[1], Yn.shape[1]
    if supports is None:
        supports = build_supports(P)
    if Ahat is None:
        Ahat, G, _, _ = fcls_from_cube(Yd, E, supports)
    else:
        G = E.T @ E
    C = tangent_basis(P)
    t, _ = spatial_target(Ahat, obs, H, Wb, P, mode=target_mode, diag=diag)
    A, lam = sw_shrink(Ahat, t, G, C, sigma2, obs, mode=mode, diag=diag)
    if mode != "none":
        # With no likelihood term the posterior mean IS the prior mean:
        # lam_n = 1 for dead columns (conditional Bayes optimality).
        bad = obs < .5
        A[:, bad] = t[:, bad]
        lam[bad] = 1.0
    return A, Ahat, t, lam, G, C


def mc_sure_lambda(Ahat, t, G, C, sigma2, ob, prior_fn,
                   tau=None, n_probe=3, seed=0, diag=None):
    """Monte-Carlo SURE for DATA-DEPENDENT priors (Stein's lemma).

    Works in tangent coordinates theta = C'(a - 1/P), where
        theta_hat ~ N(theta, S),  S = sigma^2 H^{-1},  H = C' G C,
    and the evaluation loss ||a_hat-a||^2 equals ||theta_hat-theta||^2.

    For f_lam(theta) = (1-lam) theta + lam theta_t(theta), Stein's lemma gives
        SURE(lam) = ||f-theta||^2 + 2 tr(S J_f) - tr(S).
    With b~N(0,I) and a small tau,
        tr(S J_f) ~ b' S [ f(theta+tau b) - f(theta) ] / tau,
    and because f is affine in lam we only need ONE response of the prior:
        dt = b' S [ theta_t(theta+tau b) - theta_t(theta) ] / tau .
    Hence
        SURE(lam) = lam^2 ||theta_t-theta||^2 + N V - 2 lam (N V - dt),
        V = tr(S) = sigma^2 tr(H^{-1}),
        lam* = clip((N V - dt) / ||theta_t-theta||^2, 0, 1).

    When the prior is leave-one-out independent, dt -> 0 and lam* reduces
    EXACTLY to the Theorem-8 Euclidean SURE selector.  This is the
    generalisation that lets us use self-centred non-local means.
    """
    P, N = Ahat.shape
    d = P - 1
    ob = ob > .5
    H = C.T @ G @ C
    S = sigma2 * np.linalg.inv(H)                 # (d,d)
    V = float(np.trace(S))
    a0 = 1.0 / P
    theta = C.T @ (Ahat - a0)
    theta_t = C.T @ (t - a0)
    if tau is None:
        tau = 0.05 * float(np.sqrt(V / d))
    rng = np.random.default_rng(seed)
    dt_acc = []
    for _ in range(n_probe):
        b = rng.standard_normal((d, N))
        theta_pert = theta + tau * b
        A_pert = a0 + C @ theta_pert
        t_pert = prior_fn(A_pert)
        # Only observed pixels obey the Gaussian MLE model; dead pixels
        # have no likelihood and are handled by the Bayesian prior alone.
        dt = float(((S @ b)[:, ob] * (C.T @ (t_pert - t))[:, ob]).sum() / tau)
        dt_acc.append(dt)
    dt_hat = float(np.mean(dt_acc))
    n_ob = int(ob.sum())
    sse = float(((theta_t - theta)[:, ob] ** 2).sum())
    num = n_ob * V - dt_hat
    lam = float(np.clip(num / max(sse, EPS), 0.0, 1.0))
    if diag is not None:
        diag["mc_dt"] = dt_hat
        diag["mc_dt_per_pixel"] = dt_hat / n_ob
        diag["mc_NV"] = n_ob * V
        diag["mc_sse"] = sse
        diag["mc_lam"] = lam
        diag["mc_tau"] = tau
        diag["mc_dt_std"] = float(np.std(dt_acc))
    return lam


def rmse(A, B):
    return float(np.sqrt(np.mean((A - B) ** 2)))
