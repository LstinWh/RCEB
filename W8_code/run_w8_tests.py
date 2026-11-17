"""
Reproduce the W8 paper tables on 4 hyperspectral unmixing datasets.

  Urban / Samson / JasperRidge / Houston
  Oracle(clean), noisy20, dead0.10, dead0.20, dead0.30
  Methods: Baseline | S2 | S2-E | W8

Self-contained: imports ONLY `w8_core` (plus scipy.io / matplotlib for I/O).
Use environment variables to relocate the data:
    HSU_ROOT  : folder containing data/Urban.mat and data/end4_groundTruth.mat
    HSU_DATA  : folder containing samson.mat / jasperRidge.mat / Houston.mat

Usage
-----
    python run_w8_tests.py            # full 4 x 5 evaluation (~1 min)
    QUICK=1 python run_w8_tests.py    # Samson only, 2 conditions (smoke test)

Outputs in results/: w8_results.csv, w8_results.json, w8_summary.txt,
                     w8_figure.png
"""
import os
import sys
import csv
import json
import time

import numpy as np
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w8_core import (
    fcls, vca, rmse, lfe_of, lfe_gate, neighbor_interp,
    weighted_masked_nmf, w8_unmix,
    baseline_unmix, s2_unmix, s2e_core_unmix,
    K_DEFAULT, ETA,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

ROOT = os.environ.get("HSU_ROOT", os.path.dirname(HERE))
DOWNLOADS = os.environ.get("HSU_DATA", os.path.join(ROOT, "data"))

QUICK = os.environ.get("QUICK") == "1"
ITERS_REF = 8 if QUICK else 50       # Baseline / original S2 protocol
K = 4 if QUICK else K_DEFAULT        # explicit early-stopped core
ETA_FIX = ETA
RNG_SEED = 123
METHODS = ["Baseline", "S2", "S2-E", "W8"]
CONDITIONS = [("Oracle(clean)", 0.0, None),
              ("noisy20", 0.0, 20.0),
              ("dead0.10", 0.10, 25.0),
              ("dead0.20", 0.20, 25.0),
              ("dead0.30", 0.30, 25.0)]
if QUICK:
    CONDITIONS = [("noisy20", 0.0, 20.0), ("dead0.20", 0.20, 25.0)]


# ---------------------------------------------------------------------------
# Dataset loaders (same protocol as the W-series study)
# ---------------------------------------------------------------------------
def _align(Y, Yt):
    k = float((Y * Yt).sum() / (Yt ** 2).sum())
    return Y / (k if k != 0 else 1.0)


def load_urban():
    u = scipy.io.loadmat(f"{ROOT}/data/Urban.mat")
    Y = u['A'].astype(float).T
    g = scipy.io.loadmat(f"{ROOT}/data/end4_groundTruth.mat")
    E = g['M'].astype(float)
    A_ref = g['A'].astype(float)
    Y = _align(Y, E @ A_ref)
    H = W = 220
    Y = Y[:, :H * W]
    A_ref = A_ref[:, :H * W]
    return Y, E, A_ref, H, W, 4


def load_samson():
    d = scipy.io.loadmat(f"{DOWNLOADS}/samson.mat")
    Y = d['Y'].astype(float)
    E = d['M'].astype(float)
    A_ref = d['A'].astype(float)
    Y = _align(Y, E @ A_ref)
    H = int(d['nRow'][0, 0])
    W = int(d['nCol'][0, 0])
    return Y, E, A_ref, H, W, 3


def load_jasper():
    d = scipy.io.loadmat(f"{DOWNLOADS}/jasperRidge.mat")
    X = d['X'].astype(float)
    E = d['W'].astype(float)
    A_ref = d['H'].astype(float)
    X = _align(X, E @ A_ref)
    H = W = int(round(np.sqrt(X.shape[1])))
    return X, E, A_ref, H, W, 4


def load_houston():
    h = scipy.io.loadmat(f"{DOWNLOADS}/Houston.mat")
    data = h['data'].astype(float)
    H, Wb, L = data.shape                    # 105 x 128 (NON-square grid)
    Y = data.reshape(L, -1)
    E = vca(Y, 4)
    # pseudo-reference = clip-and-renormalise heuristic (FCLS_h), NOT the
    # exact ASC-constrained LS solution (FCLS*); see paper Section 4.1.
    A_ref = fcls(E, Y)
    return Y, E, A_ref, H, Wb, 4


DATASETS = [("Urban", load_urban), ("Samson", load_samson),
            ("JasperRidge", load_jasper), ("Houston", load_houston)]
if QUICK:
    DATASETS = [("Samson", load_samson)]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(Y, E, A_ref, H, W, P):
    """Return results[method][condition] = (rmse_all, rmse_dead)."""
    L, N = Y.shape
    out = {m: {} for m in METHODS}

    # Honest clean reference: FCLS with the GIVEN endmembers on the CLEAN cube.
    # (Running the biased weighted MU on clean data is NOT a lower bound.)
    oracle = rmse(fcls(E, Y), A_ref)
    for m in METHODS:
        out[m]["Oracle(clean)"] = (oracle, None)

    for cond, df, ns in (CONDITIONS if QUICK else CONDITIONS[1:]):
        rng = np.random.default_rng(RNG_SEED)
        noise = rng.standard_normal(Y.shape) * (
            np.linalg.norm(Y) / np.sqrt(N * L) / (10 ** (ns / 20)))
        Yn = Y + noise
        dead = rng.random(N) < df
        Yn[:, dead] = 0.0
        obs = (~dead).astype(float)
        Yimp = neighbor_interp(Yn, dead, L, H, W) if dead.any() else Yn
        A0 = fcls(E, Yimp)
        A0 = np.clip(A0, 1e-3, None)
        A0 /= A0.sum(0, keepdims=True)

        _, Ab = baseline_unmix(Yn, obs, E, A0, dead, ITERS_REF)
        _, As2 = s2_unmix(Yn, obs, E, A0, dead, ITERS_REF)
        Ec, Ac = s2e_core_unmix(Yn, obs, E, A0, dead, K)
        _, Aw8 = w8_unmix(Yn, obs, E, A0, dead, K=K, eta=ETA_FIX,
                          core_EA=(Ec, Ac))

        for m, A in (("Baseline", Ab), ("S2", As2), ("S2-E", Ac), ("W8", Aw8)):
            rd = rmse(A[:, dead], A_ref[:, dead]) if dead.any() else None
            out[m][cond] = (rmse(A, A_ref), rd)

    return out


def gain(ref, val):
    """Relative RMSE gain (%) = (ref - val)/ref * 100."""
    return 100.0 * (ref - val) / ref if (ref and val is not None) else 0.0


def main():
    t0 = time.time()
    print(f"W8 reproduction | K={K} eta={ETA_FIX} seed={RNG_SEED} "
          f"QUICK={int(QUICK)}")
    print(f"ROOT={ROOT}\nDATA={DOWNLOADS}")

    results = {}
    for nm, ld in DATASETS:
        Y, E, A_ref, H, W, P = ld()
        results[nm] = evaluate(Y, E, A_ref, H, W, P)
        print(f"[done] {nm}  ({Y.shape[0]} bands x {Y.shape[1]} pixels)", flush=True)

    conds = ([c for c, _, _ in CONDITIONS] if QUICK
             else ["Oracle(clean)"] + [c for c, _, _ in CONDITIONS[1:]])
    eval_conds = conds if QUICK else conds[1:]

    # ---------------- CSV / JSON (structure identical to wseries_results) ---
    with open(f"{RES}/w8_results.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["dataset", "method", "condition", "rmse_all", "rmse_dead"])
        for nm in results:
            for m in METHODS:
                for c in conds:
                    v = results[nm][m][c]
                    wr.writerow([nm, m, c,
                                 f"{v[0]:.6f}" if v[0] is not None else "",
                                 f"{v[1]:.6f}" if v[1] is not None else ""])

    js = {"datasets": list(results.keys()), "methods": METHODS,
          "conditions": conds,
          "config": {"K": K, "eta": ETA_FIX, "ITERS_REF": ITERS_REF,
                     "RNG_SEED": RNG_SEED, "QUICK": QUICK},
          "results": {nm: {m: {c: [results[nm][m][c][0],
                                   results[nm][m][c][1]] for c in conds}
                           for m in METHODS} for nm in results}}
    with open(f"{RES}/w8_results.json", "w") as f:
        json.dump(js, f, indent=2)

    # ---------------- Text summary -----------------------------------------
    lines = []
    lines.append("=" * 96)
    lines.append("W8 — RMSE(A_est, A_ref), lower is better")
    lines.append(f"K={K}  eta={ETA_FIX}  seed={RNG_SEED}  "
                 f"references: Baseline/S2 @ {ITERS_REF} iters")
    lines.append("=" * 96)
    for nm in results:
        lines.append(f"\n----- {nm} -----")
        lines.append(f"  {'method':12s}" + "".join(f"{c:>13s}" for c in eval_conds))
        for m in METHODS:
            rows = [f"{results[nm][m][c][0]:.4f}" for c in eval_conds]
            lines.append(f"  {m:12s}" + "".join(f"{v:>13s}" for v in rows))
        lines.append(f"  Oracle(clean) FCLS = {results[nm]['W8']['Oracle(clean)'][0]:.4f}")

    lines.append("\n" + "=" * 96)
    lines.append("Relative RMSE gain of W8 (mean over degradation conditions, %)")
    lines.append("=" * 96)
    lines.append(f"  {'dataset':14s}{'vs Baseline':>14s}{'vs S2':>14s}{'vs S2-E':>14s}")
    all_g = {m: [] for m in ("Baseline", "S2", "S2-E")}
    worst, wins = 1e9, 0
    n_cells = len(results) * len(eval_conds)
    for nm in results:
        row = []
        for ref in ("Baseline", "S2", "S2-E"):
            gs = [gain(results[nm][ref][c][0], results[nm]["W8"][c][0])
                  for c in eval_conds]
            all_g[ref].extend(gs)
            row.append(float(np.mean(gs)))
            if ref == "S2-E":
                worst = min(worst, *gs)
                wins += sum(1 for g in gs if g > 0)
        lines.append(f"  {nm:14s}" + "".join(f"{v:14.2f}" for v in row))
    lines.append(f"  {'MEAN(' + str(n_cells) + ' cells)':14s}" +
                 "".join(f"{np.mean(all_g[r]):14.2f}" for r in ("Baseline", "S2", "S2-E")))
    lines.append(f"\n  W8 vs S2-E : win {wins}/{n_cells} cells, worst cell {worst:+.2f}%")

    # gain decomposition (the honest core message)
    s2e_vs_s2 = [gain(results[nm]["S2"][c][0], results[nm]["S2-E"][c][0])
                 for nm in results for c in eval_conds]
    lines.append(f"  S2-E vs S2 (explicit early stopping alone): {np.mean(s2e_vs_s2):+.2f}%")
    lines.append("  => total W8 gain vs S2 = early stopping + W8 recalibration; "
                 "report ALL THREE references in the paper.")

    dead_cells = [c for c in eval_conds if c.startswith("dead")]
    if dead_cells:
        spreads = [max(results[nm][m][c][1] for m in METHODS)
                   - min(results[nm][m][c][1] for m in METHODS)
                   for nm in results for c in dead_cells]
        lines.append(f"  dead-pixel RMSE spread across methods: "
                     f"max = {max(spreads):.6f} (0 => no missing-spectrum recovery)")

    txt = "\n".join(lines)
    with open(f"{RES}/w8_summary.txt", "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)

    # ---------------- Figure ------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        refs = ["Baseline", "S2", "S2-E"]
        colors = ["#9ecae1", "#fdae6b", "#d62728"]
        fig, axes = plt.subplots(1, len(DATASETS),
                                 figsize=(3.6 * len(DATASETS), 3.6))
        if len(DATASETS) == 1:
            axes = [axes]
        x = np.arange(len(eval_conds))
        bw = 0.26
        for ax, nm in zip(axes, results.keys()):
            for j, ref in enumerate(refs):
                vals = [gain(results[nm][ref][c][0], results[nm]["W8"][c][0])
                        for c in eval_conds]
                ax.bar(x + (j - 1) * bw, vals, bw, label=f"vs {ref}",
                       color=colors[j])
            ax.axhline(0, color="0.4", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(eval_conds, rotation=30, ha="right", fontsize=8)
            ax.set_title(nm, fontsize=10)
            ax.set_ylabel("relative RMSE gain (%)", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle("W8 — relative RMSE gain over the three references", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(f"{RES}/w8_figure.png", dpi=140)
        plt.close()
        print(f"\n[fig] {RES}/w8_figure.png")
    except Exception as e:
        print(f"[warn] figure skipped: {e}")

    print(f"[csv] {RES}/w8_results.csv")
    print(f"[jsn] {RES}/w8_results.json")
    print(f"[txt] {RES}/w8_summary.txt")
    print(f"Total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
