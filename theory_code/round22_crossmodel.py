"""Round 22 -- cross-paradigm validation of the reference-consistency diagnosis.

Referee request (fourth round): the diagnosis must NOT be shown only on the
authors' own case-study estimator and conventional FCLS.  Add a small,
representative set of algorithms with DIFFERENT mathematical modelling
mechanisms and ask whether the reference--model inconsistency and the
reference-dependent ranking reversal also affect them -- without turning the
paper into a SOTA leaderboard.

Three arms, all recomputed from released data; nothing hard-coded.

[A] FIXED-ENDMEMBER COMPARISON (Experiment A).
    All estimators share the same E, the same degraded cube Y, the same mask,
    the same noise realisation and the same evaluation reference A_ref.
    M = {Exact-FCLS, SUnSAL, SUnSAL-TV, GLS, Huber, SCW-DC} plus the two
    internal controls S2-E / Baseline.  The evaluation reference is
    interpolated A_ref(lam) = clip_norm((1-lam) A_H + lam A_X), where A_H is
    the clip-and-renormalise pseudo-reference and A_X the exact ASC-LS solution
    of the CLEAN cube.  The SAME fixed estimator outputs are scored at every
    lam, so any ranking change is caused by the evaluation reference alone.

[B] MODEL-MISMATCH CROSS-MODEL CHECK (Experiment B).
    Synthesise a cube with smooth per-pixel endmember variability and NO
    additive noise; the generating abundances are the physical truth by
    construction.  Fit three DIFFERENT mixing models: fixed-E FCLS, an
    ELMM-type multiplicative scale model, and a PLMM-type additive
    perturbation model.  Their RMSEs are reported in SEPARATE blocks (never
    in one leaderboard) because the model assumptions are not comparable.
    Question: does allowing spectral variability remove the apparent
    reference inconsistency (i.e. is it a property of the adopted mixing
    model rather than of the reference abundance)?

[C] INDEPENDENT SCENE.
    Cuprite (AVIRIS, 188 retained bands, 100x100 centre ROI, 4 endmembers
    chosen by greedy max-min spectral angle from the 12 library spectra).
    This scene was never used in W8 design, R0--R21 experiments or in any
    hyperparameter choice.  No pixel-level abundance ground truth is supplied
    with this copy, so it is used ONLY for the GT-free reference-consistency
    diagnostic and the ranking-reversal reproduction.

Estimator implementations are self-contained NumPy and are explicitly
documented in the paper as *representatives of a modelling mechanism*, not as
exact reproductions of the original published solvers.
"""
import os
import json
import time
import itertools

import numpy as np
import scipy.io

from common import (HERE, CONDS, load_all, fcls, rmse,
                    lfe_of, lfe_gate, weighted_masked_nmf, K_DEFAULT)
from w8_core import w8_unmix
from final_validate import make_case
from round11_exactfcls import build_supports as bs11, exact_fcls as ef11
from xdomain_core import (build_supports, exact_fcls,
                          band_precision_from_residual, gls_exact_fcls,
                          huber_exact_fcls, sigma_from_residual)

OUT_JSON = os.path.join(HERE, "results", "round22_crossmodel.json")
OUT_TXT = os.path.join(HERE, "results", "round22_summary.txt")

GT = ("Urban", "Samson", "JasperRidge")
LAMBDAS = (0.0, 0.25, 0.50, 0.75, 1.0)
SEED = 123
# (condition name, noise protocol) -- identical degradation for all estimators
UNITS = (("noisy20", "homo"),
         ("dead0.20", "homo"),
         ("noisy20", "hetero_strong"))
METHODS = ("Exact-FCLS", "SUnSAL", "SUnSAL-TV", "GLS", "Huber",
           "SCW-DC", "S2-E", "Baseline")
NU_LEVELS = (0.0, 0.02, 0.05)
CUPRITE = (os.environ.get("HSU_DATA", os.path.join(os.path.dirname(HERE), "data")),
           "Cuprite.mat", "cuprite_ref.mat")


def clip_norm(A):
    B = np.clip(A, 1e-9, None)
    return B / B.sum(0, keepdims=True)


def exact_clean(E, Y):
    """Exact ASC-constrained LS on an all-observed cube."""
    P, N = E.shape[1], Y.shape[1]
    A, _, _ = ef11(E.T @ E, E.T @ Y, (Y ** 2).sum(0), bs11(P), P, N)
    return A


# ---------------------------------------------------------------------------
# Sparse unmixing: ADMM with an ASC soft penalty (SUnSAL-type)
# ---------------------------------------------------------------------------
def _admm_const(E):
    G = E.T @ E
    lam_max = float(np.linalg.eigvalsh(G).max())
    rho = 0.5 * lam_max
    gamma = 3.0 * lam_max           # soft ASC, fixed a priori
    P = E.shape[1]
    M = G + gamma * np.ones((P, P)) + rho * np.eye(P)
    return np.linalg.inv(M), rho, gamma


def sunsal(E, Yd, A0, iters=220, l1_rel=0.05):
    """min 1/2||Y-EA||^2 + gamma/2||1'A-1||^2 + l1||A||_1 + iota_{A>=0}.

    ADMM splitting A = Z.  A-step is a single P x P solve shared by all
    pixels; Z-step is a non-negative soft threshold.  This is the *sparsity*
    mechanism (L1 on abundances) with no spatial prior.  Penalties are fixed
    relative to rho = lambda_max(E'E)/2 so that they are scale-free.
    """
    Minv, rho, gamma = _admm_const(E)
    B = E.T @ Yd
    P, N = E.shape[1], Yd.shape[1]
    l1 = l1_rel * rho
    Z = np.clip(A0, 0, None)
    U = np.zeros_like(Z)
    ones = np.ones((P, N))
    for _ in range(iters):
        A = Minv @ (B + gamma * ones + rho * (Z - U))
        Z = np.maximum(A + U - l1 / rho, 0.0)
        U += A - Z
    return Z


# ---------------------------------------------------------------------------
# Spatial prior: TV-regularised ASC unmixing (SUnSAL-TV-type)
# ---------------------------------------------------------------------------
def _grad2(a):
    """Forward spatial differences, zero at the far boundary. a:(P,H,W)."""
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, :, :-1] = a[:, :, 1:] - a[:, :, :-1]
    gy[:, :-1, :] = a[:, 1:, :] - a[:, :-1, :]
    return np.stack((gx, gy), 0)


def _div2(g):
    """Adjoint of _grad2 (so <_grad2(a),g> = <a,_div2(g)>)."""
    gx, gy = g[0], g[1]
    d = np.zeros_like(gx)
    d[:, :, 1:] += gx[:, :, :-1]
    d[:, :, :-1] -= gx[:, :, :-1]
    d[:, 1:, :] += gy[:, :-1, :]
    d[:, :-1, :] -= gy[:, :-1, :]
    return d


def _tv_prox(V, rho, l1, ltv, Z0, p0, inner=6):
    """Warm-started Chambolle--Pock prox of l1|Z| + ltv TV(Z) + iota_{Z>=0}
    at V with quadratic parameter rho.  Returns (Z, p) for warm restart."""
    tau = 1.0 / rho
    sig_q = 0.45 * rho
    sig_p = 0.45 * rho / 8.0
    Z = Z0.copy()
    p = p0.copy()
    q = np.zeros_like(Z)
    Zbar = Z.copy()
    for _ in range(inner):
        g = _grad2(Zbar)
        q = np.clip(q + sig_q * Zbar, -l1, l1)
        pn = p + sig_p * g
        nrm = np.sqrt((pn ** 2).sum(0))
        scale = np.minimum(1.0, ltv / np.maximum(nrm, 1e-12))
        p = pn * scale[None, ...]
        Znew = np.maximum(V - tau * (q + _div2(p)), 0.0)
        Zbar = 2.0 * Znew - Z
        Z = Znew
    return Z, p


def sunsal_tv(E, Yd, A0, H, Wb, iters=90, l1_rel=0.05, tv_rel=0.05):
    """Same objective as sunsal() plus an isotropic spatial TV penalty.

    ADMM splitting A = Z; the Z-step is the TV/L1 proximal solved by a
    warm-started primal-dual iteration.  This is the *spatial-prior*
    mechanism.  No claim of reproducing the original SUnSAL-TV solver.
    """
    Minv, rho, gamma = _admm_const(E)
    B = E.T @ Yd
    P, N = E.shape[1], Yd.shape[1]
    assert H * Wb == N
    l1 = l1_rel * rho
    ltv = tv_rel * rho
    Z = np.clip(A0, 0, None)
    U = np.zeros_like(Z)
    ones = np.ones((P, N))
    p = np.zeros((2, P, H, Wb))
    for _ in range(iters):
        A = Minv @ (B + gamma * ones + rho * (Z - U))
        V = A + U
        Z, p = _tv_prox(V.reshape(P, H, Wb), rho, l1, ltv,
                        Z.reshape(P, H, Wb), p)
        Z = Z.reshape(P, N)
        U += A - Z
    return Z


# ---------------------------------------------------------------------------
# Fixed-endmember estimator set (one call per unit; outputs are fixed)
# ---------------------------------------------------------------------------
def fixed_estimators(E, Yn, obs, dead, A0, H, Wb):
    Yd = Yn * obs[None, :]
    ob = obs > .5
    P = E.shape[1]
    supports = build_supports(P)
    G, B, y2 = E.T @ E, E.T @ Yd, (Yd ** 2).sum(0)
    A_fc, residual, _ = exact_fcls(G, B, y2, supports)

    def post(A):
        A = clip_norm(np.array(A, copy=True))
        A[:, dead] = A0[:, dead]
        return A

    out = {"Exact-FCLS": post(A_fc)}

    A_su = sunsal(E, Yd, A0)
    out["SUnSAL"] = post(A_su)
    A_tv = sunsal_tv(E, Yd, A0, H, Wb)
    out["SUnSAL-TV"] = post(A_tv)

    d_hat = band_precision_from_residual(Yd, E, out["Exact-FCLS"], ob)
    A_gls, _, _, _ = gls_exact_fcls(Yd, E, d_hat, supports, A0, dead)
    out["GLS"] = post(A_gls)

    sigma2 = sigma_from_residual(residual[ob], Yd.shape[0], P - 1) ** 2
    delta = 2.5 * float(np.sqrt(sigma2))
    A_hu = huber_exact_fcls(Yd, E, obs, delta, iters=2,
                            supports=supports, A_init=A_fc)
    out["Huber"] = post(A_hu)

    _, A_w8 = w8_unmix(Yn, obs, E, A0, dead)
    out["SCW-DC"] = post(A_w8)
    w0 = lfe_gate(lfe_of(Yn), obs)
    _, A_s2e = weighted_masked_nmf(Yn, obs, E, A0, dead, K_DEFAULT,
                                   dw=w0, fix_E=False)
    out["S2-E"] = post(A_s2e)
    _, A_base = weighted_masked_nmf(Yn, obs, E, A0, dead, 50,
                                    dw=None, fix_E=False)
    out["Baseline"] = post(A_base)
    return out


# ---------------------------------------------------------------------------
# Per-pixel-design ASC-LS via batched KKT support enumeration
# ---------------------------------------------------------------------------
def exact_fcls_batched(Gw, bw, y2w, supports, P, N):
    """Exact ASC-LS when each pixel has its OWN Gram matrix Gw[n]."""
    best_res = np.full(N, np.inf)
    best_A = np.zeros((N, P))
    for S in supports:
        idx = np.array(S)
        k = len(S)
        Gs = Gw[:, idx[:, None], idx[None, :]] + 1e-9 * np.eye(k)
        Gi = np.linalg.inv(Gs)
        gi1 = Gi @ np.ones(k)
        den = gi1.sum(1)
        U = np.einsum("nij,nj->ni", Gi, bw[:, idx])
        lam = (U.sum(1) - 1.0) / np.maximum(den, 1e-12)
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
        best_A[bad] = 1.0 / P
    return best_A.T


def plmm_augmented(Yv, E, supports, lam_d=1.0):
    """PLMM-type: additive spectral perturbation in a fixed low-rank subspace.

        min_{a in simplex, d} ||y - E a - B d||^2 + lam_d ||d||^2,
        B = diag-free cosine basis modulated by the endmembers.

    Eliminating d gives W = I - B(B'B + lam_d I)^{-1}B', so the abundance
    subproblem is an ordinary ASC-constrained LS with Gram E'WE and can be
    solved exactly by the same KKT support enumeration.  This is NOT the
    original PLMM solver; it is a transparent convex representative of the
    'additive perturbation' modelling mechanism.
    """
    L, P = E.shape
    B = np.stack([np.cos(2 * np.pi * np.arange(L) / L
                         + ph) for ph in np.linspace(0, np.pi, P)], 1) * E
    T = np.linalg.inv(B.T @ B + lam_d * np.eye(P))
    WE = E - B @ (T @ (B.T @ E))
    WY = Yv - B @ (T @ (B.T @ Yv))
    G, Bm, y2 = E.T @ WE, E.T @ WY, (Yv * WY).sum(0)
    A, _, _ = exact_fcls(G, Bm, y2, supports)
    D = T @ (B.T @ (Yv - E @ A))
    Yhat = E @ A + B @ D
    return A, D, float(np.sqrt(((Yv - Yhat) ** 2).mean()))


def _elmm_fit(Yv, E, A_init, supports, iters=8, ridge=5.0):
    """ELMM-type: per-pixel multiplicative endmember scale factors.

        y_n = sum_p a_pn * psi_pn * e_p,   psi_pn > 0, ridge to 1.

    Alternating minimisation: (i) given psi, solve the ASC-constrained LS with
    a per-pixel design matrix by batched KKT support enumeration; (ii) given a,
    solve a small ridge-regularised least-squares problem for psi.  This is a
    transparent representative of the 'spectral scale variability' mechanism,
    not a reproduction of the original ELMM solver.
    """
    L, N = Yv.shape
    P = E.shape[1]
    A = A_init.copy()
    Q = np.ones((P, N))
    eye = np.eye(P)[None, :, :]
    for _ in range(iters):
        Phi = E[:, :, None] * Q[None, :, :]             # L,P,N
        Gw = np.einsum("lpn,lqn->npq", Phi, Phi, optimize=True)
        bw = np.einsum("lpn,ln->np", Phi, Yv, optimize=True)
        y2w = (Yv ** 2).sum(0)
        A = exact_fcls_batched(Gw, bw, y2w, supports, P, N)
        Z = E[:, :, None] * A[None, :, :]               # L,P,N
        Gz = np.einsum("lpn,lqn->npq", Z, Z, optimize=True)
        bz = np.einsum("lpn,ln->np", Z, Yv, optimize=True)
        Q = np.linalg.solve(Gz + ridge * eye,
                            bz[:, :, None] + ridge * np.ones((N, P, 1))
                            )[:, :, 0].T
        Q = np.clip(Q, 0.2, 5.0)
    Phi = E[:, :, None] * Q[None, :, :]
    Yhat = np.einsum("lpn,pn->ln", Phi, A, optimize=True)
    return A, Q, float(np.sqrt(((Yv - Yhat) ** 2).mean()))


# ---------------------------------------------------------------------------
# Cuprite: a genuinely independent scene
# ---------------------------------------------------------------------------
def load_cuprite():
    base, fy, fr = CUPRITE
    d = scipy.io.loadmat(os.path.join(base, fy))
    r = scipy.io.loadmat(os.path.join(base, fr))
    bands = r["BANDS"].ravel().astype(int) - 1
    Yall = d["Y"].astype(float)[bands] / 65535.0
    Mall = d["M"].astype(float)[bands]
    cube = Yall.reshape(len(bands), 250, 190)
    Y3 = np.clip(cube[:, 40:140, 40:140], 0.0, 1.0)
    Y = Y3.reshape(len(bands), -1)
    # greedy max-min spectral angle: fixed BEFORE any estimator is run
    C = Mall.T @ Mall
    nrm = np.sqrt(np.diag(C))
    ang = np.degrees(np.arccos(np.clip(C / np.outer(nrm, nrm), -1, 1)))
    sel = [int(np.argmax(ang.mean(1)))]
    for _ in range(3):
        dist = ang[:, sel].min(1)
        dist[sel] = -1
        sel.append(int(np.argmax(dist)))
    E = Mall[:, sorted(sel)]
    # the same global scalar alignment used by the four released loaders
    Ah = fcls(E, Y)
    k = float((Y * (E @ Ah)).sum() / ((E @ Ah) ** 2).sum())
    Y = Y / (k if k != 0 else 1.0)
    return Y, E, 100, 100, sorted(sel)


def _ranks(d):
    return [m for m, _ in sorted(d.items(), key=lambda kv: kv[1])]


def _mechanism(est, P, H, Wb):
    """Show that the added estimators really instantiate different priors."""
    fx = est["Exact-FCLS"]
    return {m: {
        "tv": float(np.abs(_grad2(A.reshape(P, H, Wb))).mean()),
        "dist_to_fx": rmse(A, fx),
    } for m, A in est.items()}


def _reversals(rec, lam0, lam1, methods):
    pairs = 0
    flips = []
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            a = rec[lam0]["rmse"][m1] - rec[lam0]["rmse"][m2]
            b = rec[lam1]["rmse"][m1] - rec[lam1]["rmse"][m2]
            if a * b < 0:
                pairs += 1
                flips.append("%s/%s" % (m1, m2))
    return pairs, flips


# ---------------------------------------------------------------------------
# Arm A: reference-dependent ranking across independent estimators
# ---------------------------------------------------------------------------
def arm_a(data):
    out = {}
    for nm in GT + ("Houston",):
        Y, E, A_ref, H, Wb, P = data[nm]
        A_H = fcls(E, Y)
        A_X = exact_clean(E, Y)
        refs = {"lam|%.2f" % lam: clip_norm((1.0 - lam) * A_H + lam * A_X)
                for lam in LAMBDAS}
        for cname, proto in UNITS:
            cond = next(c for c in CONDS if c[0] == cname)
            Yn, obs, dead, A0, _ = make_case(
                Y, E, A_ref, H, Wb, SEED, cond, proto)
            est = fixed_estimators(E, Yn, obs, dead, A0, H, Wb)
            rec = {}
            for tag, ref in refs.items():
                d = {m: rmse(est[m], ref) for m in METHODS}
                rec[tag] = {"rmse": d, "rank": _ranks(d)}
            key = "%s|%s|%s|%d" % (nm, cname, proto, SEED)
            out[key] = {"rec": rec,
                        "mechanism": _mechanism(est, P, H, Wb),
                        "ref_gap": rmse(A_H, A_X),
                        "clean_resid": float(np.sqrt(((Y - E @ A_H) ** 2).mean())),
                        "reversals": _reversals(
                            rec, "lam|0.00", "lam|1.00", METHODS)[0],
                        "flip_pairs": _reversals(
                            rec, "lam|0.00", "lam|1.00", METHODS)[1]}
            print("[A] %-12s %-10s %-14s done" % (nm, cname, proto),
                  flush=True)
    return out


# ---------------------------------------------------------------------------
# Arm B: model mismatch, three different mixing models, no merged ranking
# ---------------------------------------------------------------------------
def arm_b(data, n_pix=3000):
    out = {}
    for nm in GT:
        Y, E, A, H, Wb, P = data[nm]
        L, N = Y.shape
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(N, min(n_pix, N), replace=False))
        g = np.stack([np.sin(2 * np.pi * np.arange(L) / L + ph)
                      for ph in np.linspace(0, np.pi, P)], 1)
        u = rng.standard_normal((P, N))
        supports = build_supports(P)
        Asub = A[:, idx]
        rows = {}
        for nu in NU_LEVELS:
            drift = 1.0 + nu * (g.T[:, :, None] * u[:, None, :])
            Eeff = E.T[:, :, None] * drift
            Yv = np.einsum("pn,pln->ln", A, Eeff)[:, idx]
            Yc = (E @ Asub)
            fixed_res = float(np.sqrt(((Yv - Yc) ** 2).mean()))
            A_fc = exact_clean(E, Yv)
            A_el, _, r_el = _elmm_fit(Yv, E, A_fc, supports)
            A_pl, _, r_pl = plmm_augmented(Yv, E, supports)
            rows["nu=%.2f" % nu] = {
                "fixed_residual": fixed_res,
                "fixed_rmse": rmse(A_fc, Asub),
                "elmm_residual": r_el,
                "elmm_rmse": rmse(A_el, Asub),
                "plmm_residual": r_pl,
                "plmm_rmse": rmse(A_pl, Asub),
            }
            print("[B] %-12s nu=%.2f residual fixed %.5f / ELMM %.5f /"
                  " PLMM %.5f"
                  % (nm, nu, fixed_res, r_el, r_pl), flush=True)
        out[nm] = rows
    return out


# ---------------------------------------------------------------------------
# Arm C: independent scene, GT-free diagnostic + ranking reproduction
# ---------------------------------------------------------------------------
def arm_c():
    Y, E, H, Wb, sel = load_cuprite()
    A_H = fcls(E, Y)
    A_X = exact_clean(E, Y)
    refs = {"lam|%.2f" % lam: clip_norm((1.0 - lam) * A_H + lam * A_X)
            for lam in LAMBDAS}
    cond = next(c for c in CONDS if c[0] == "noisy20")
    Yn, obs, dead, A0, _ = make_case(
        Y, E, A_H, H, Wb, 20261110, cond, "homo")
    est = fixed_estimators(E, Yn, obs, dead, A0, H, Wb)
    rec = {}
    for tag, ref in refs.items():
        d = {m: rmse(est[m], ref) for m in METHODS}
        rec[tag] = {"rmse": d, "rank": _ranks(d)}
    rev, flips = _reversals(rec, "lam|0.00", "lam|1.00", METHODS)
    return {"rec": rec, "reversals": rev, "flip_pairs": flips,
            "mechanism": _mechanism(est, E.shape[1], H, Wb),
            "ref_gap": rmse(A_H, A_X),
            "clean_resid": float(np.sqrt(((Y - E @ A_H) ** 2).mean())),
            "endmembers": sel, "bands": int(Y.shape[0]),
            "pixels": int(Y.shape[1]), "model_consistency_gain":
            100.0 * (rec["lam|0.00"]["rmse"]["SCW-DC"]
                     - rec["lam|1.00"]["rmse"]["Exact-FCLS"])
            / rec["lam|0.00"]["rmse"]["SCW-DC"]}


def summarise(a, b, c):
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("ROUND 22 -- CROSS-PARADIGM VALIDATION OF REFERENCE CONSISTENCY")
    emit("=" * 88)
    emit("")
    gt_keys = [k for k in a if not k.startswith("Houston|")]
    emit("[A] Mean RMSE (3 provided-GT scenes x 3 units) as the evaluation")
    emit("    reference moves from the heuristic A_H to exact ASC-LS A_X.")
    hdr = "    %-13s" % "lambda" + "".join("%11s" % m[:10] for m in METHODS)
    emit(hdr)
    curve = {}
    for lam in LAMBDAS:
        tag = "lam|%.2f" % lam
        vals = {m: float(np.mean([a[k]["rec"][tag]["rmse"][m]
                                 for k in gt_keys])) for m in METHODS}
        curve["%.2f" % lam] = vals
        emit("    %-13.2f" % lam + "".join("%11.5f" % vals[m]
                                          for m in METHODS))
    emit("")
    emit("    rank at lam=0 : " + " < ".join(
        m for m, _ in sorted(curve["0.00"].items(), key=lambda kv: kv[1])))
    emit("    rank at lam=1 : " + " < ".join(
        m for m, _ in sorted(curve["1.00"].items(), key=lambda kv: kv[1])))
    rev = int(np.mean([a[k]["reversals"] for k in gt_keys]))
    emit("    mean pairwise ranking reversals per unit: %d / %d pairs"
         % (rev, len(METHODS) * (len(METHODS) - 1) // 2))
    pairs = set()
    for k in gt_keys:
        pairs.update(a[k]["flip_pairs"])
    emit("    distinct flipping pairs: %s" % (", ".join(sorted(pairs)) or "none"))
    emit("")
    mech = {}
    for m in METHODS:
        mech[m] = (float(np.mean([a[k]["mechanism"][m]["tv"]
                                 for k in gt_keys])),
                   float(np.mean([a[k]["mechanism"][m]["dist_to_fx"]
                                  for k in gt_keys])))
    emit("    mechanism check (mean spatial TV / distance to exact FCLS):")
    for m in ("Exact-FCLS", "SUnSAL", "SUnSAL-TV", "GLS", "Huber",
              "SCW-DC"):
        emit("      %-12s TV %.5f  dist %.5f" % (m, mech[m][0], mech[m][1]))
    emit("")
    emit("[B] Model-mismatch control (NO additive noise, truth = generating A).")
    emit("    Separate blocks: fixed-E | ELMM-type | PLMM-type; no merged ranking.")
    for nm, rows in b.items():
        for key, v in rows.items():
            emit("    %-12s %-8s fixed res %.5f / rmse %.5f | ELMM res %.5f /"
                 " rmse %.5f | PLMM res %.5f / rmse %.5f"
                 % (nm, key, v["fixed_residual"], v["fixed_rmse"],
                    v["elmm_residual"], v["elmm_rmse"],
                    v["plmm_residual"], v["plmm_rmse"]))
    emit("")
    nv = "nu=0.05"
    fr = np.mean([b[k][nv]["fixed_rmse"] for k in b])
    er = np.mean([b[k][nv]["elmm_rmse"] for k in b])
    pr = np.mean([b[k][nv]["plmm_rmse"] for k in b])
    f0 = np.mean([b[k]["nu=0.00"]["fixed_rmse"] for k in b])
    emit("    nu=0.05 mean RMSE: fixed %.5f -> ELMM %.5f / PLMM %.5f"
         " ; no-variability fixed %.5f" % (fr, er, pr, f0))
    emit("")
    emit("[C] Independent Cuprite scene (never used in design or tuning).")
    emit("    L=%d, N=%d, endmembers %s; heuristic-vs-exact gap %.5f,"
         " clean residual %.5f" % (c["bands"], c["pixels"], c["endmembers"],
                                   c["ref_gap"], c["clean_resid"]))
    for lam in LAMBDAS:
        tag = "lam|%.2f" % lam
        d = c["rec"][tag]["rmse"]
        emit("    %-9.2f SCW-DC %.5f | Exact-FCLS %.5f | SUnSAL-TV %.5f"
             % (lam, d["SCW-DC"], d["Exact-FCLS"], d["SUnSAL-TV"]))
    emit("    pairwise reversals %d ; distinct flipping pairs %s"
         % (c["reversals"], ", ".join(c["flip_pairs"]) or "none"))
    emit("    rank lam=0 : " + " < ".join(c["rec"]["lam|0.00"]["rank"]))
    emit("    rank lam=1 : " + " < ".join(c["rec"]["lam|1.00"]["rank"]))
    emit("")
    return "\n".join(lines), {
        "mean_reversals": rev,
        "flip_pairs": sorted(pairs),
        "curve": curve,
        "rank0": sorted(curve["0.00"], key=lambda m: curve["0.00"][m]),
        "rank1": sorted(curve["1.00"], key=lambda m: curve["1.00"][m]),
        "b_fixed": float(fr), "b_elmm": float(er), "b_plmm": float(pr),
        "b_fixed_nu0": float(f0),
        "tv_sunsal_tv": float(mech["SUnSAL-TV"][0]),
        "tv_sunsal": float(mech["SUnSAL"][0]),
        "tv_fcls": float(mech["Exact-FCLS"][0]),
        "cuprite_reversals": int(c["reversals"]),
        "cuprite_ref_gap": float(c["ref_gap"]),
        "cuprite_rank0": c["rec"]["lam|0.00"]["rank"],
        "cuprite_rank1": c["rec"]["lam|1.00"]["rank"],
    }


def main():
    t0 = time.time()
    data = load_all()
    a = arm_a(data)
    b = arm_b(data)
    c = arm_c()
    txt, stat = summarise(a, b, c)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"meta": {"lambdas": list(LAMBDAS),
                            "units": list(UNITS), "seed": SEED,
                            "methods": list(METHODS),
                            "nu": list(NU_LEVELS)},
                   "arm_a": a, "arm_b": b, "arm_c": c, "stats": stat},
                  f, indent=2)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("[out]", OUT_JSON)
    print("elapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
