"""Round 10 -- closed form of the ASC-renormalised MU fixed point.

Round 9 found that the fixed-E multiplicative update (MU)

    a <- clip( a * (E^T y) / (E^T E a) );  a <- a / sum(a)

converges to a limit that is ~20% better (in abundance RMSE) than both W8 and
the clip-normalised unconstrained least-squares solution, and that the
trajectory is monotone up to K=2000.  The naive candidate
clip_norm((E^T E)^{-1} E^T y) is NOT that limit (max distance ~0.5).

Claim (proved in THEORY.md as Theorem 7).  Let b=E^T y, G=E^T E, and let
S be the support of the limit a*.  For p in S the fixed-point equation reads
(G a*)_p = b_p / s with s the normalisation constant, and a*_p = 0 outside S.
Hence

    a*_S = G_SS^{-1} b_S / (1^T G_SS^{-1} b_S),

where S is the maximal support for which G_SS^{-1} b_S > 0.  This is an
active-set (complementarity) system, NOT the simplex-constrained LS problem:
G is not diagonal, so clipping the full solution is not equivalent.

This script:
  (1) verifies the claim to machine precision against a 2000-step MU run;
  (2) computes the support by a GT-free active-set loop (<= P solves of a
      P x P system), giving a ONE-SHOT estimator with no iterations, no tuned
      K, no eta and no consensus;
  (3) re-validates that one-shot estimator on all 48 cells and both noise
      protocols, against every released reference.
"""
import os
import json
import time

import numpy as np

from common import (HERE, EPS, CONDS, load_all, lfe_of, lfe_gate,
                    fcls, rmse, weighted_masked_nmf, K_DEFAULT)
from w8_core import w8_unmix
from final_validate import make_case
from round9_shrink import clip_norm, mu_fixedE, FLOOR

OUT_JSON = os.path.join(HERE, "results", "round10_fixedpoint.json")
OUT_TXT = os.path.join(HERE, "results", "round10_summary.txt")
SEEDS = [123, 7, 2024]
AUDIT_CONDS = ("noisy20", "dead0.20")


def active_set_closed_form(G, B, floor=1e-8):
    """One-shot active-set solve of the MU complementarity system.

    G : (P,P),  B : (P,N).  Returns A (P,N) and the number of solves used.
    Pixels are grouped by support pattern, so at most 2^P small linear
    systems are solved (P <= 4 here).
    """
    P, N = B.shape
    S_mask = np.ones((P, N), dtype=bool)       # start with the full support
    A = np.zeros((P, N))
    it = 0
    for it in range(1, P + 2):
        codes = np.array([sum((1 << p) * S_mask[p, n] for p in range(P))
                          for n in range(N)])
        changed = False
        for code in np.unique(codes):
            idx = np.where(codes == code)[0]
            S = [p for p in range(P) if (code >> p) & 1]
            if not S:
                A[:, idx] = 1.0 / P
                continue
            Gs = G[np.ix_(S, S)]
            try:
                U = np.linalg.solve(Gs, B[np.ix_(S, idx)])
            except np.linalg.LinAlgError:
                U = np.linalg.lstsq(Gs, B[np.ix_(S, idx)], rcond=None)[0]
            bad = U <= 0
            if bad.any():
                # drop the most negative coordinate on every offending pixel
                for j in np.where(bad.any(0))[0]:
                    p = S[int(np.argmin(U[:, j]))]
                    S_mask[p, idx[j]] = False
                changed = True
                continue
            a = np.zeros((P, len(idx)))
            a[S] = U
            A[:, idx] = a / np.maximum(a.sum(0, keepdims=True), 1e-12)
        if not changed:
            break
    A = np.maximum(A, FLOOR)
    A = A / A.sum(0, keepdims=True)
    return A, it


def one_mu_step(G, B, A):
    X = A * B / (G @ A + EPS)
    return clip_norm(X)


def one_unit(Y, E, A_ref, H, Wb, cond, seed, protocol="homo", audit=False):
    Yn, obs, dead, A0, Aref = make_case(
        Y, E, A_ref, H, Wb, seed, cond, protocol)
    Yd = Yn * obs[None, :]
    ob = obs > .5
    w0 = lfe_gate(lfe_of(Yn), obs)
    G = E.T @ E
    B = E.T @ Yd

    res, diag = {}, {}
    res["Oracle(clean)"] = rmse(fcls(E, Y), Aref)
    res["FCLS-direct"] = rmse(A0, Aref)
    _, A_s2e = weighted_masked_nmf(Yn, obs, E, A0, dead, K_DEFAULT,
                                   dw=w0, fix_E=False)
    res["S2-E"] = rmse(A_s2e, Aref)
    _, A_w8 = w8_unmix(Yn, obs, E, A0, dead)
    res["W8"] = rmse(A_w8, Aref)

    # one-shot estimator, with dead columns held at the initial estimate
    A_cf, nsolve = active_set_closed_form(G, B)
    A_cf[:, dead] = A0[:, dead]
    res["CF-active"] = rmse(A_cf, Aref)
    diag["active_set_solves"] = nsolve
    diag["support_size"] = float((A_cf[:, ob] > 1e-6).mean())

    if audit:
        snaps = mu_fixedE(Yd, E, A0, dead, 2000, {200, 500, 2000})
        for k, Ak in snaps.items():
            res["MU%d" % k] = rmse(Ak, Aref)
        A_long = snaps[2000]
        diag["cf_vs_MU2000_maxdiff"] = float(
            np.abs(A_cf[:, ob] - A_long[:, ob]).max())
        # is A_cf a genuine fixed point of the MU operator?
        A_next = one_mu_step(G, B, A_cf)
        diag["cf_fixedpoint_residual"] = float(
            np.abs(A_next[:, ob] - A_cf[:, ob]).max())
        A_next_long = one_mu_step(G, B, A_long)
        diag["MU2000_fixedpoint_residual"] = float(
            np.abs(A_next_long[:, ob] - A_long[:, ob]).max())
        diag["muK"] = {str(k): round(res["MU%d" % k], 6)
                       for k in (200, 500, 2000)}

    diag["deadRMSE_W8"] = (rmse(A_w8[:, dead], Aref[:, dead])
                           if dead.any() else None)
    diag["deadRMSE_CF"] = (rmse(A_cf[:, dead], Aref[:, dead])
                           if dead.any() else None)
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
        audit = (protocol == "homo" and seed == 123
                 and cond[0] in AUDIT_CONDS)
        r, dg = one_unit(Y, E, A_ref, H, Wb, cond, seed,
                         protocol=protocol, audit=audit)
        out["units"][key] = {"rmse": r, "diag": dg}
        if i % 8 == 0:
            print("[%2d/%d] %-38s W8 %.4f CF-active %.4f" %
                  (i + 1, len(jobs), key, r["W8"], r["CF-active"]),
                  flush=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    u = out["units"]
    lines = []
    def emit(s):
        print(s, flush=True)
        lines.append(s)

    emit("ROUND 10 -- CLOSED FORM OF THE ASC-MU FIXED POINT")
    emit("=" * 78)
    au = [x for x in u if "cf_vs_MU2000_maxdiff" in u[x]["diag"]]
    emit("")
    emit("[1] algebraic verification on %d audit cells" % len(au))
    for x in au:
        d = u[x]["diag"]
        emit("   %-38s |CF - MU2000| %.3e   FP-resid(CF) %.3e   "
             "FP-resid(MU2000) %.3e   solves %d" %
             (x, d["cf_vs_MU2000_maxdiff"], d["cf_fixedpoint_residual"],
              d["MU2000_fixedpoint_residual"], d["active_set_solves"]))
    emit("   >> max over cells: %.4e" %
         max(u[x]["diag"]["cf_vs_MU2000_maxdiff"] for x in au))
    emit("")
    for protocol in ("homo", "hetero"):
        keys = [x for x in u if x.startswith(protocol + "|")]
        gt = [x for x in keys if not x.startswith(protocol + "|Houston")]
        for scope, kk in (("12 GT cells x 3 seeds / 1 seed", gt),
                          ("all cells", keys)):
            ref = np.array([u[x]["rmse"]["W8"] for x in kk])
            emit("[2] %s | %s | n=%d" % (protocol, scope, len(kk)))
            for m in ("CF-active", "W8", "S2-E", "FCLS-direct",
                      "Oracle(clean)"):
                v = np.array([u[x]["rmse"][m] for x in kk])
                emit("   %-16s rmse %.5f  vs W8 %+7.2f%%  wins %2d/%d  "
                     "worst %+7.2f%%" %
                     (m, v.mean(),
                      100 * float(np.mean((ref - v) / ref)),
                      int((v < ref).sum()), len(kk),
                      float((100 * (ref - v) / ref).min())))
            emit("")
    emit("[3] dead-pixel RMSE, homo seed 123")
    for tag in ("deadRMSE_W8", "deadRMSE_CF"):
        vals = [u[x]["diag"][tag] for x in u
                if x.startswith("homo|") and x.endswith("|123")
                and u[x]["diag"][tag] is not None]
        emit("   %-20s mean %.5f" % (tag, float(np.mean(vals))))

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[out]", OUT_JSON, OUT_TXT, "elapsed %.1fs" % (time.time() - t0),
          flush=True)


if __name__ == "__main__":
    main()
