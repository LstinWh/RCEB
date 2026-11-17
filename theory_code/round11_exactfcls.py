"""Round 11 -- is the closed-form (CF) estimator just exact FCLS?

The released W8 code implements FCLS as iterated clip + column normalisation,
which is a heuristic, not the KKT solution of

    min_a ||y - E a||^2   s.t.  a >= 0,  1^T a = 1 .

For P <= 4 we can enumerate every support S and solve the ASC-constrained
normal equations in closed form:

    a_S = G_SS^{-1} (b_S - lambda 1),
    lambda = (1^T G_SS^{-1} b_S - 1) / (1^T G_SS^{-1} 1),

with feasibility a_S > 0 and (G a - b)_p + lambda >= 0 for p not in S.
The feasible support with the smallest residual is the EXACT FCLS solution.

Round 10's CF estimator solves instead  a_S = G_SS^{-1} b_S / (1^T G_SS^{-1} b_S),
i.e. a scaled (not Lagrange-shifted) system.  This round checks whether the
two coincide.  If exact FCLS already matches CF, the +20% is only "we used a
proper FCLS", NOT a new estimator; if CF is distinctly better, the difference
is a genuine algorithmic effect and must be explained.
"""
import os
import json
import time
import itertools

import numpy as np

from common import (HERE, CONDS, load_all, lfe_of, lfe_gate, fcls, rmse,
                    weighted_masked_nmf, K_DEFAULT)
from w8_core import w8_unmix
from final_validate import make_case
from round10_fixedpoint import active_set_closed_form

OUT_JSON = os.path.join(HERE, "results", "round11_exactfcls.json")
OUT_TXT = os.path.join(HERE, "results", "round11_summary.txt")
SEEDS = [123, 7, 2024]


def build_supports(P):
    return [tuple(S) for k in range(1, P + 1)
            for S in itertools.combinations(range(P), k)]


def exact_fcls(G, B, y2, supports, P, N):
    """Exact ASC-constrained least squares by exhaustive KKT support search."""
    best_res = np.full(N, np.inf)
    best_A = np.zeros((P, N))
    chosen = np.zeros(N, dtype=int)
    for ci, S in enumerate(supports):
        idx = np.array(S)
        Gs = G[np.ix_(idx, idx)]
        Gi = np.linalg.inv(Gs)
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
        kkt = (Ga_minus_b + lam[None, :]) >= -1e-9
        outside = np.setdiff1d(np.arange(P), idx)
        if outside.size:
            ok &= kkt[outside].all(0)
        # residual ||y - E a||^2 = y2 - 2 b^T a + a^T G a
        res = y2 - 2.0 * (B * A).sum(0) + (A * (G @ A)).sum(0)
        take = ok & (res < best_res)
        if take.any():
            best_res[take] = res[take]
            best_A[:, take] = A[:, take]
            chosen[take] = ci
    # fall back for any pixel without a feasible support
    bad = ~np.isfinite(best_res)
    if bad.any():
        U = np.linalg.pinv(G) @ B[:, bad]
        A = np.maximum(U, 1e-8)
        A /= A.sum(0, keepdims=True)
        best_A[:, bad] = A
    return best_A, best_res, chosen


def one_unit(Y, E, A_ref, H, Wb, cond, seed, protocol="homo"):
    Yn, obs, dead, A0, Aref = make_case(
        Y, E, A_ref, H, Wb, seed, cond, protocol)
    Yd = Yn * obs[None, :]
    ob = obs > .5
    P, N = E.shape[1], Yn.shape[1]
    w0 = lfe_gate(lfe_of(Yn), obs)
    G = E.T @ E
    B = E.T @ Yd
    y2 = (Yd ** 2).sum(0)
    supports = build_supports(P)

    res, diag = {}, {}
    res["Oracle(clean)"] = rmse(fcls(E, Y), Aref)
    res["FCLS-direct"] = rmse(A0, Aref)
    _, A_s2e = weighted_masked_nmf(Yn, obs, E, A0, dead, K_DEFAULT,
                                   dw=w0, fix_E=False)
    res["S2-E"] = rmse(A_s2e, Aref)
    _, A_w8 = w8_unmix(Yn, obs, E, A0, dead)
    res["W8"] = rmse(A_w8, Aref)

    A_cf, nsolve = active_set_closed_form(G, B)
    A_cf[:, dead] = A0[:, dead]
    res["CF-active"] = rmse(A_cf, Aref)

    A_fc, _, chosen = exact_fcls(G, B, y2, supports, P, N)
    A_fc[:, dead] = A0[:, dead]
    res["Exact-FCLS"] = rmse(A_fc, Aref)

    diag["cf_vs_exactfcls_maxdiff"] = float(
        np.abs(A_cf[:, ob] - A_fc[:, ob]).max())
    diag["cf_vs_exactfcls_meandiff"] = float(
        np.abs(A_cf[:, ob] - A_fc[:, ob]).mean())
    diag["support_cf"] = float((A_cf[:, ob] > 1e-6).mean())
    diag["support_fc"] = float((A_fc[:, ob] > 1e-6).mean())
    diag["exactfcls_residual"] = float(np.mean(
        np.sqrt(np.maximum(y2[ob] - 2.0 * (B[:, ob] * A_fc[:, ob]).sum(0)
                           + (A_fc[:, ob] * (G @ A_fc[:, ob])).sum(0), 0))))
    return res, diag


def main():
    t0 = time.time()
    data = load_all()
    out = {"meta": {"seeds": SEEDS, "conditions": [c[0] for c in CONDS],
                    "protocols": ["homo", "hetero"]}, "units": {}}
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
        r, dg = one_unit(Y, E, A_ref, H, Wb, cond, seed, protocol=protocol)
        out["units"][key] = {"rmse": r, "diag": dg}
        if i % 8 == 0:
            print("[%2d/%d] %-38s Exact-FCLS %.4f CF %.4f W8 %.4f" %
                  (i + 1, len(jobs), key, r["Exact-FCLS"],
                   r["CF-active"], r["W8"]), flush=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    u = out["units"]
    lines = []
    def emit(s):
        print(s, flush=True)
        lines.append(s)

    emit("ROUND 11 -- EXACT FCLS CONTROL")
    emit("=" * 78)
    emit("")
    emit("[1] are CF-active and exact FCLS the same estimator?")
    emit("   max |CF - exactFCLS| over all 64 cells: %.5f" %
         max(u[x]["diag"]["cf_vs_exactfcls_maxdiff"] for x in u))
    emit("   mean |CF - exactFCLS| over all 64 cells: %.5f" %
         float(np.mean([u[x]["diag"]["cf_vs_exactfcls_meandiff"] for x in u])))
    emit("   mean support size: CF %.3f  exactFCLS %.3f" %
         (float(np.mean([u[x]["diag"]["support_cf"] for x in u])),
          float(np.mean([u[x]["diag"]["support_fc"] for x in u]))))
    emit("")
    for protocol in ("homo", "hetero"):
        keys = [x for x in u if x.startswith(protocol + "|")]
        gt = [x for x in keys if not x.startswith(protocol + "|Houston")]
        for scope, kk in (("12 GT cells", gt), ("all cells", keys)):
            ref = np.array([u[x]["rmse"]["W8"] for x in kk])
            emit("[2] %s | %s | n=%d" % (protocol, scope, len(kk)))
            for m in ("CF-active", "Exact-FCLS", "W8", "S2-E",
                      "FCLS-direct", "Oracle(clean)"):
                v = np.array([u[x]["rmse"][m] for x in kk])
                emit("   %-16s rmse %.5f  vs W8 %+7.2f%%  wins %2d/%d  "
                     "worst %+7.2f%%" %
                     (m, v.mean(), 100 * float(np.mean((ref - v) / ref)),
                      int((v < ref).sum()), len(kk),
                      float((100 * (ref - v) / ref).min())))
            vc = np.array([u[x]["rmse"]["CF-active"] for x in kk])
            vf = np.array([u[x]["rmse"]["Exact-FCLS"] for x in kk])
            emit("   >> CF-active vs Exact-FCLS: %+.2f%%  wins %d/%d" %
                 (100 * float(np.mean((vf - vc) / vf)),
                  int((vc < vf).sum()), len(kk)))
            emit("")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[out]", OUT_JSON, OUT_TXT, "elapsed %.1fs" % (time.time() - t0),
          flush=True)


if __name__ == "__main__":
    main()
