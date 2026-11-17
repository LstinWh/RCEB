"""Round 9 -- decisive audit and a theory-derived one-shot estimator.

Two questions are answered with real runs, no placeholders.

Q1  Is the +17% "converged fixed-E GLS" of Round 7 a genuine estimator, or
    another semiconvergence artefact with a hidden iteration budget K=200?
    We run the fixed-E MU up to K=2000 on 8 representative cells, compute its
    algebraic fixed point in closed form, and record RMSE(K) plus the distance
    to that fixed point.

Q2  Lemma 1 says a per-pixel scalar credibility w_n is unidentifiable in the
    abundance step.  Does a *shared* ridge prior break that invariance?  For
        J(E,A) = sum_n w_n ||y_n - E a_n||^2 + rho_bar sum_n ||a_n||^2
    the normal equations give
        a_n = (E^T E + (rho_bar / w_n) I)^{-1} E^T y_n ,
    i.e. credibility becomes an identifiable per-pixel shrinkage strength
    rho_n = rho_bar / w_n.  The closed-form, non-iterative estimator is CARS
    (credibility-adaptive ridge shrinkage); rho_bar is selected by SURE
    (ground-truth free).

Everything else (degradation, loaders, W8 core, credibility gate) is identical
to W8_code so the numbers stay comparable with the paper tables.
"""
import os
import json
import time

import numpy as np

from common import (HERE, EPS, CONDS, load_all, lfe_of, lfe_gate,
                    fcls, rmse, estimate_sigma, weighted_masked_nmf,
                    K_DEFAULT)
from w8_core import w8_unmix
from final_validate import make_case

OUT_JSON = os.path.join(HERE, "results", "round9_shrink.json")
OUT_TXT = os.path.join(HERE, "results", "round9_summary.txt")
SEEDS = [123, 7, 2024]
KS_LONG = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
KS_SHORT = [1, 3, 5, 10, 20, 50, 100, 200]
RHO_GRID = np.power(10.0, np.linspace(-4.0, 2.0, 25))
FLOOR = 1e-8


def clip_norm(U):
    """Implemented ASC feasibility map: clip, then column-normalise."""
    U = np.maximum(U, FLOOR)
    return U / U.sum(0, keepdims=True)


def eig_basis(E):
    G = E.T @ E
    lam, Q = np.linalg.eigh(G)
    return np.maximum(lam, 1e-12), Q


def ridge_U(lam, Q, B, rn):
    """U = (G + diag(rn) I)^{-1} B, vectorised over pixels via the eigenbasis."""
    Bt = Q.T @ B
    return Q @ (Bt / (lam[:, None] + rn[None, :]))


def sure_rho(lam, Q, B, Yd, ob, sig2):
    """Ground-truth-free SURE choice of a scalar ridge strength."""
    Bo = B[:, ob]
    Yo = Yd[:, ob]
    y2 = float((Yo ** 2).sum())
    Bto = Q.T @ Bo
    nobs = int(ob.sum())
    scale = float(lam.mean())
    best, score_best = None, np.inf
    scores = []
    for r in RHO_GRID:
        rho = r * scale
        Ut = Bto / (lam[:, None] + rho)
        BU = float((Bo * (Q @ Ut)).sum())
        UGU = float((lam[:, None] * Ut ** 2).sum())
        resid = y2 - 2.0 * BU + UGU
        df = float((lam / (lam + rho)).sum())
        score = resid + 2.0 * sig2 * nobs * df
        scores.append([round(float(rho), 8), round(float(score), 4),
                       round(float(df), 5)])
        if score < score_best:
            score_best, best = score, float(rho)
    return best, scores


def mu_fixedE(Yd, E, A0, dead, K, want_k):
    """Fixed-E, d=1 MU used in Round 7, with an explicit audit trail."""
    G = E.T @ E
    num = E.T @ Yd
    A = A0.copy()
    Ad = A0[:, dead].copy()
    snaps = {}
    for k in range(1, K + 1):
        Anew = A * num / (G @ A + EPS)
        Anew = np.maximum(Anew, FLOOR)
        Anew = Anew / Anew.sum(0, keepdims=True)
        Anew[:, dead] = Ad
        A = Anew
        if k in want_k:
            snaps[k] = A.copy()
    return snaps


def one_unit(Y, E, A_ref, H, Wb, cond, seed, protocol="homo", audit=False):
    Yn, obs, dead, A0, Aref = make_case(
        Y, E, A_ref, H, Wb, seed, cond, protocol)
    L, N = Yn.shape
    Yd = Yn * obs[None, :]
    ob = obs > .5
    w0 = lfe_gate(lfe_of(Yn), obs)
    w = np.maximum(w0.copy(), 1e-3)
    lam, Q = eig_basis(E)
    B = E.T @ Yd
    sig2 = estimate_sigma(Yn, obs) ** 2
    rho_hat, sure_curve = sure_rho(lam, Q, B, Yd, ob, sig2)

    res, diag = {}, {}
    res["Oracle(clean)"] = rmse(fcls(E, Y), Aref)
    res["FCLS-direct"] = rmse(A0, Aref)
    _, A_s2e = weighted_masked_nmf(Yn, obs, E, A0, dead, K_DEFAULT,
                                   dw=w0, fix_E=False)
    res["S2-E"] = rmse(A_s2e, Aref)
    _, A_w8 = w8_unmix(Yn, obs, E, A0, dead)
    res["W8"] = rmse(A_w8, Aref)

    # ---- Q1: MU trajectory and its algebraic fixed point ------------------
    ks = KS_LONG if audit else KS_SHORT
    snaps = mu_fixedE(Yd, E, A0, dead, max(ks), set(ks))
    U_fp = np.linalg.solve(E.T @ E + 1e-12 * np.eye(E.shape[1]), B)
    A_fp = clip_norm(U_fp)
    A_fp[:, dead] = A0[:, dead]
    res["MU200"] = rmse(snaps[200], Aref)
    res["MU2000"] = (rmse(snaps[2000], Aref) if audit else None)
    res["MU-fixedpoint"] = rmse(A_fp, Aref)
    diag["muK"] = {str(k): round(rmse(snaps[k], Aref), 6) for k in ks}
    diag["mu_fp_dist"] = {
        str(k): round(float(np.abs(snaps[k][:, ob] -
                                   A_fp[:, ob]).max()), 6) for k in ks}
    diag["rho_hat"] = rho_hat
    diag["sigma_hat"] = float(np.sqrt(sig2))
    diag["sure_curve"] = sure_curve
    diag["audit"] = bool(audit)

    # ---- Q2: closed-form estimators --------------------------------------
    def apply(rn, tag):
        A = clip_norm(ridge_U(lam, Q, B, rn))
        A[:, dead] = A0[:, dead]
        res[tag] = rmse(A, Aref)
        return A

    one = np.ones(N)
    A_u = apply(rho_hat * one, "CARS-uniform")       # credibility OFF
    A_w = apply(rho_hat / w, "CARS-w")               # credibility ON (CARS)
    apply(np.zeros(N), "ridge0")
    diag["identifiability_gap"] = float(
        np.abs(A_u[:, ob] - A_w[:, ob]).max())

    # Lemma-1 check: with rho=0, a column weight w_n cancels exactly in
    #   (w_n G)^{-1} (w_n E^T y_n) = G^{-1} E^T y_n .
    # The check is run on the OBSERVED columns only (dead columns are 0/0).
    U_plain = Q @ ((Q.T @ B) / lam[:, None])
    Bw = B * w[None, :]
    U_weighted = Q @ ((Q.T @ Bw) / (lam[:, None] * w[None, :]))
    diag["rho0_gap"] = float(
        np.abs(clip_norm(U_plain)[:, ob] - clip_norm(U_weighted)[:, ob]).max())

    # oracle ridge (diagnostic upper bound only, uses ground truth)
    best, best_rho = np.inf, None
    for r in RHO_GRID:
        rho = r * float(lam.mean())
        A = clip_norm(ridge_U(lam, Q, B, rho / w))
        A[:, dead] = A0[:, dead]
        v = rmse(A, Aref)
        if v < best:
            best, best_rho = v, rho
    res["CARS-oracle-rho"] = best
    diag["rho_oracle"] = best_rho

    diag["deadRMSE_W8"] = (rmse(A_w8[:, dead], Aref[:, dead])
                           if dead.any() else None)
    diag["deadRMSE_CARS"] = (rmse(A_w[:, dead], Aref[:, dead])
                             if dead.any() else None)
    return res, diag


def main():
    t0 = time.time()
    data = load_all()
    out = {"meta": {"seeds": SEEDS, "conditions": [c[0] for c in CONDS],
                    "rho_grid": RHO_GRID.tolist(),
                    "ks_long": KS_LONG, "ks_short": KS_SHORT},
           "units": {}}
    jobs = [(s, nm, c) for s in SEEDS for nm in data for c in CONDS]
    for i, (seed, nm, cond) in enumerate(jobs):
        Y, E, A_ref, H, Wb, P = data[nm]
        key = "%s|%s|%d" % (nm, cond[0], seed)
        audit = (seed == 123 and cond[0] in ("noisy20", "dead0.20"))
        r, dg = one_unit(Y, E, A_ref, H, Wb, cond, seed, audit=audit)
        out["units"][key] = {"rmse": r, "diag": dg}
        if i % 6 == 0:
            print("[%2d/%d] %-30s S2-E %.4f W8 %.4f MU200 %.4f "
                  "CARS-w %.4f CARS-u %.4f" %
                  (i + 1, len(jobs), key, r["S2-E"], r["W8"],
                   r["MU200"], r["CARS-w"], r["CARS-uniform"]), flush=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    u = out["units"]
    lines = []
    def emit(s):
        print(s, flush=True)
        lines.append(s)

    au = [x for x in u if u[x]["diag"]["audit"]]
    emit("ROUND 9 -- MU FIXED-POINT AUDIT + CARS (all numbers computed)")
    emit("=" * 78)
    emit("")
    emit("[Q1] fixed-E MU trajectory, mean RMSE over %d audit cells" % len(au))
    for k in KS_LONG:
        v = np.array([u[x]["diag"]["muK"][str(k)] for x in au])
        d = np.array([u[x]["diag"]["mu_fp_dist"][str(k)] for x in au])
        emit("   K=%4d  rmse %.5f   max|MU_K - fixedpoint| %.5f" %
             (k, v.mean(), d.mean()))
    v200 = np.array([u[x]["rmse"]["MU200"] for x in au])
    v2000 = np.array([u[x]["rmse"]["MU2000"] for x in au])
    vfp = np.array([u[x]["rmse"]["MU-fixedpoint"] for x in au])
    emit("   >> MU200 vs MU2000 : %+.2f%%  (MU200 better in %d/%d)" %
         (100 * (v2000 - v200).mean() / v2000.mean(),
          int((v200 < v2000).sum()), len(au)))
    emit("   >> MU2000 vs closed-form fixed point: RMSE gap %.6f" %
         float((vfp - v2000).mean()))
    emit("")
    emit("[Q2] method ranking, 48 cells (4 datasets x 4 conditions x 3 seeds)")
    ref = np.array([u[x]["rmse"]["S2-E"] for x in u])
    refw = np.array([u[x]["rmse"]["W8"] for x in u])
    methods = ["Oracle(clean)", "FCLS-direct", "ridge0", "MU-fixedpoint",
               "MU200", "CARS-uniform", "CARS-w", "CARS-oracle-rho",
               "S2-E", "W8"]
    rows = []
    for m in methods:
        v = np.array([u[x]["rmse"][m] for x in u])
        rows.append((v.mean(), m,
                     100 * float(np.mean((ref - v) / ref)),
                     100 * float(np.mean((refw - v) / refw)),
                     int((v < ref).sum()), int((v < refw).sum()),
                     float((100 * (refw - v) / refw).min())))
    for mean, m, g, gw, wins, winsw, worst in sorted(rows):
        emit("   %-20s rmse %.5f  vs S2-E %+7.2f%%  vs W8 %+7.2f%%  "
             "wins %2d/48 vs W8 %2d/48  worst %+7.2f%%" %
             (m, mean, g, gw, wins, winsw, worst))
    vw = np.array([u[x]["rmse"]["CARS-w"] for x in u])
    vu = np.array([u[x]["rmse"]["CARS-uniform"] for x in u])
    emit("")
    emit("   >> credibility contribution (CARS-w vs CARS-uniform): %+.2f%%  "
         "wins %d/48" % (100 * (vu - vw).mean() / vu.mean(),
                         int((vw < vu).sum())))
    rh = [u[x]["diag"]["rho_hat"] for x in u]
    emit("   >> SURE rho: median %.4g  range %.4g -- %.4g" %
         (float(np.median(rh)), float(np.min(rh)), float(np.max(rh))))
    emit("   >> identifiability gap max|A_w - A_u|: median %.5f" %
         float(np.median([u[x]["diag"]["identifiability_gap"] for x in u])))
    emit("")
    emit("[per-dataset, vs W8, mean per-cell gain %]")
    for nm in data:
        keys = [x for x in u if x.startswith(nm + "|")]
        r = np.array([u[x]["rmse"]["W8"] for x in keys])
        v = np.array([u[x]["rmse"]["CARS-w"] for x in keys])
        m = np.array([u[x]["rmse"]["CARS-uniform"] for x in keys])
        emit("   %-14s CARS-w %+7.2f%%  wins %2d/%d   (uniform %+7.2f%%)" %
             (nm, 100 * float(np.mean((r - v) / r)), int((v < r).sum()),
              len(keys), 100 * float(np.mean((r - m) / r))))
    emit("")
    emit("[dead-pixel RMSE, seed 123]")
    for tag in ("deadRMSE_W8", "deadRMSE_CARS"):
        vals = [u[x]["diag"][tag] for x in u if x.endswith("|123")
                and u[x]["diag"][tag] is not None]
        emit("   %-20s mean %.5f" % (tag, float(np.mean(vals))))

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[out]", OUT_JSON, OUT_TXT, "elapsed %.1fs" % (time.time() - t0),
          flush=True)


if __name__ == "__main__":
    main()
