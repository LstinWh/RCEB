"""Round 7 (final): full validation of the dual-weight GLS estimator.

Two noise protocols:
  homo  : the ORIGINAL protocol (i.i.d. Gaussian, band-wise homoscedastic).
          Here Gauss-Markov theory predicts d_l = const, so the row weighting
          SHOULD NOT help much -- we report this honestly.
  hetero: band-wise heteroscedastic noise (water-vapour-like bands), where
          d_l ~ 1/sigma_l^2 is the provably optimal weight.  This isolates the
          independent contribution of the proposed data-driven row weight.

References include FCLS-direct (the plain initial solution), so that no gain
from "returning to a sensible baseline" can be mis-attributed to the method.

The GLS iteration uses a CONVERGENCE CRITERION instead of a tuned budget K.
"""
import os
import json
import time

import numpy as np

from common import (HERE, ROOT, CONDS, EPS, K_DEFAULT,
                    load_all, lfe_of, lfe_gate, neighbor_interp, fcls,
                    rmse, weighted_masked_nmf)
from w8_core import w8_unmix

OUT_JSON = os.path.join(HERE, "results", "final_results.json")
OUT_TXT = os.path.join(HERE, "results", "final_summary.txt")
SEEDS = [123, 7, 2024]
KMAX = 200
TOL = 1e-6


def band_scale(L, kind):
    """Deterministic per-band noise s.d. multiplier (no dependence on seed)."""
    if kind == "homo":
        return np.ones(L)
    x = np.linspace(0.0, 1.0, L)
    s = (1.0
         + 3.0 * np.exp(-((x - 0.28) / 0.07) ** 2)
         + 2.2 * np.exp(-((x - 0.74) / 0.05) ** 2)
         + 0.45 * np.sin(2 * np.pi * x))
    s = np.clip(s, 0.3, None)
    # keep the TOTAL noise energy identical to the homoscedastic protocol
    s = s / np.sqrt(np.mean(s ** 2))
    return s


def make_case(Y, E, A_ref, H, Wb, seed, cond, protocol):
    _, df, ns = cond
    L, N = Y.shape
    rng = np.random.default_rng(seed)
    sigma = np.linalg.norm(Y) / np.sqrt(N * L) / (10 ** (ns / 20))
    s = band_scale(L, protocol)
    Yn = Y + rng.standard_normal(Y.shape) * sigma * s[:, None]
    dead = rng.random(N) < df
    Yn[:, dead] = 0.0
    obs = (~dead).astype(float)
    Yimp = neighbor_interp(Yn, dead, L, H, Wb) if dead.any() else Yn
    A0 = fcls(E, Yimp)
    A0 = np.clip(A0, 1e-3, None)
    A0 /= A0.sum(0, keepdims=True)
    return Yn, obs, dead, A0, A_ref


def band_variance(Yn, obs, H, Wb, kind):
    ob = obs > .5
    Yo = Yn[:, ob]
    sec = Yo[2:] - 2.0 * Yo[1:-1] + Yo[:-2]
    v_sec = np.median(sec ** 2, axis=1)
    v_sec = np.concatenate([[v_sec[0]], v_sec, [v_sec[-1]]])
    Y3 = Yn.reshape(Yn.shape[0], H, Wb)
    o3 = (obs > .5).reshape(H, Wb)
    m = (o3[:, 1:] & o3[:, :-1]).ravel()
    dif = np.diff(Y3, axis=2).reshape(Yn.shape[0], -1)
    v_spa = np.median(dif[:, m] ** 2, axis=1) / 2.0
    if kind == "sec":
        v = v_sec
    elif kind == "spatial":
        v = v_spa
    elif kind == "hybrid":
        v = np.sqrt(np.maximum(v_sec, 1e-12) * np.maximum(v_spa, 1e-12))
    elif kind == "uniform":
        v = np.ones_like(v_sec)
    else:
        raise ValueError(kind)
    return np.maximum(v, 1e-10)


def make_d(Yn, obs, H, Wb, kind):
    v = band_variance(Yn, obs, H, Wb, kind)
    d = 1.0 / v
    d = d / d.mean()
    d = np.clip(d, 0.05, 20.0)
    return d / d.mean()


def gls_converge(Yn, obs, E, A0, dead, d, A_ref):
    """Fixed-E GLS MU with a CONVERGENCE criterion (no tuned K)."""
    Yd = Yn * obs[None, :]
    Ad = A0[:, dead].copy()
    EtD = E.T * d[None, :]
    G = EtD @ E
    num = EtD @ Yd
    A = A0.copy()
    K = 0
    for k in range(1, KMAX + 1):
        Anew = A * num / (G @ A + EPS)
        Anew = np.maximum(Anew, 1e-8)
        Anew = Anew / Anew.sum(0, keepdims=True)
        Anew[:, dead] = Ad
        change = np.max(np.abs(Anew - A) / np.maximum(np.abs(A), 1e-6))
        A = Anew
        K = k
        if change < TOL:
            break
    return rmse(A, A_ref), K, A


def rmse_dead(A, A_ref, dead):
    if dead.sum() == 0:
        return None
    return float(np.sqrt(((A[:, dead] - A_ref[:, dead]) ** 2).mean()))


def one_unit(Y, E, A_ref, H, Wb, cond, seed, protocol):
    Yn, obs, dead, A0, Aref = make_case(
        Y, E, A_ref, H, Wb, seed, cond, protocol)
    w0 = lfe_gate(lfe_of(Yn), obs)
    res, diag = {}, {"dead_frac": float(dead.mean())}

    _, A_b = weighted_masked_nmf(Yn, obs, E, A0, dead, 50, dw=None, fix_E=False)
    res["Baseline"] = rmse(A_b, Aref)
    _, A_s2 = weighted_masked_nmf(Yn, obs, E, A0, dead, 50, dw=w0, fix_E=False)
    res["S2"] = rmse(A_s2, Aref)
    _, A_s2e = weighted_masked_nmf(Yn, obs, E, A0, dead, K_DEFAULT,
                                   dw=w0, fix_E=False)
    res["S2-E"] = rmse(A_s2e, Aref)
    _, A_w8 = w8_unmix(Yn, obs, E, A0, dead)
    res["W8"] = rmse(A_w8, Aref)
    res["FCLS-direct"] = rmse(A0, Aref)
    res["Oracle(clean)"] = rmse(fcls(E, Y), Aref)

    for kind in ("uniform", "spatial", "hybrid", "sec"):
        d = make_d(Yn, obs, H, Wb, kind)
        r, K, A = gls_converge(Yn, obs, E, A0, dead, d, Aref)
        tag = "GLS-" + kind
        res[tag] = r
        diag[tag + "_K"] = K
        diag[tag + "_deadRMSE"] = rmse_dead(A, Aref, dead)
        diag[tag + "_d_range"] = [round(float(d.min()), 3),
                                  round(float(d.max()), 3)]
    diag["deadRMSE_S2-E"] = rmse_dead(A_s2e, Aref, dead)
    diag["deadRMSE_W8"] = rmse_dead(A_w8, Aref, dead)
    return res, diag


def main():
    t0 = time.time()
    data = load_all()
    out = {"meta": {"seeds": SEEDS, "conditions": [c[0] for c in CONDS],
                    "protocols": ["homo", "hetero"], "tol": TOL,
                    "KMAX": KMAX}, "units": {}}
    jobs = []
    for protocol in ("homo", "hetero"):
        seeds = SEEDS if protocol == "homo" else [123]
        for seed in seeds:
            for nm in data:
                for cond in CONDS:
                    jobs.append((protocol, seed, nm, cond))
    for i, (protocol, seed, nm, cond) in enumerate(jobs):
        Y, E, A_ref, H, Wb, P = data[nm]
        key = "%s|%s|%s|%d" % (protocol, nm, cond[0], seed)
        r, dg = one_unit(Y, E, A_ref, H, Wb, cond, seed, protocol)
        out["units"][key] = {"rmse": r, "diag": dg}
        if cond[0] == "noisy20" or i % 8 == 0:
            print("[%2d/%d] %-38s S2-E %.4f W8 %.4f GLS-spatial %.4f (K=%d)" %
                  (i + 1, len(jobs), key, r["S2-E"], r["W8"],
                   r["GLS-spatial"], dg["GLS-spatial_K"]), flush=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    u = out["units"]
    lines = []
    def emit(s):
        print(s, flush=True)
        lines.append(s)
    emit("FINAL VALIDATION  (all numbers computed, no placeholders)")
    emit("=" * 78)
    for protocol in ("homo", "hetero"):
        for scope, pred in (("12 GT cells", lambda k: not k.startswith(
                                protocol + "|Houston")),
                            ("Houston only", lambda k: k.startswith(
                                protocol + "|Houston")),
                            ("all 16/48 cells", lambda k: True)):
            keys = [k for k in u if k.startswith(protocol + "|") and pred(k)]
            if not keys:
                continue
            ref = np.array([u[k]["rmse"]["S2-E"] for k in keys])
            emit("")
            emit("[%s | %s | n=%d]" % (protocol, scope, len(keys)))
            rows = []
            for m in ("Baseline", "S2", "S2-E", "W8", "FCLS-direct",
                      "GLS-uniform", "GLS-spatial", "GLS-hybrid", "GLS-sec",
                      "Oracle(clean)"):
                v = np.array([u[k]["rmse"][m] for k in keys])
                rows.append((100 * (ref - v).mean() / ref.mean(), m,
                             v.mean(), int((v < ref).sum()),
                             float((100 * (ref - v) / ref).min())))
            for g, m, mean, wins, worst in sorted(rows, reverse=True):
                emit("  %-16s rmse %.5f  vs S2-E %+7.2f%%  worst %+7.2f%%  wins %2d/%d"
                     % (m, mean, g, worst, wins, len(keys)))
            vg = np.array([u[k]["rmse"]["GLS-spatial"] for k in keys])
            vu = np.array([u[k]["rmse"]["GLS-uniform"] for k in keys])
            emit("  >> row-weight contribution: GLS-spatial vs GLS-uniform "
                 "%+.2f%%  (wins %d/%d)" %
                 (100 * (vu - vg).mean() / vu.mean(),
                  int((vg < vu).sum()), len(keys)))
            ks = [u[k]["diag"]["GLS-spatial_K"] for k in keys]
            emit("  >> converged K: min %d  median %d  max %d" %
                 (min(ks), int(np.median(ks)), max(ks)))
    emit("")
    emit("[dead-pixel RMSE, homo, seed 123, 12 cells with dead pixels]")
    for m in ("S2-E", "W8"):
        vals = [u[k]["diag"]["deadRMSE_" + m]
                for k in u if k.startswith("homo|") and k.endswith("|123")
                and u[k]["diag"]["dead_frac"] > 0
                and not k.startswith("homo|Houston")]
        emit("  %-16s mean %.5f" % (m, np.mean(vals)))
    for m in ("GLS-uniform", "GLS-spatial"):
        vals = [u[k]["diag"][m + "_deadRMSE"]
                for k in u if k.startswith("homo|") and k.endswith("|123")
                and u[k]["diag"]["dead_frac"] > 0
                and not k.startswith("homo|Houston")]
        emit("  %-16s mean %.5f" % (m, np.mean(vals)))
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[out]", OUT_JSON, OUT_TXT, "elapsed %.1fs" % (time.time() - t0),
          flush=True)


if __name__ == "__main__":
    main()
