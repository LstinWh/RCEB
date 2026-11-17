"""Audit saved cross-model units against released references and KKT conditions."""
import argparse
import hashlib
import itertools
import json
import os
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
SCENES = ("Urban", "Samson", "JasperRidge")
METHODS = ("Exact-FCLS", "SUnSAL", "SUnSAL-TV", "GLS", "Huber",
           "SCW-DC", "S2-E", "Baseline")
UNITS = (("noisy20", "homo"), ("dead0.20", "homo"),
         ("noisy20", "hetero_strong"))


def metric(A, B):
    return float(np.sqrt(np.mean((A - B) ** 2)))


def inversions(a, b):
    return int(sum((a[x] - a[y]) * (b[x] - b[y]) < 0
                   for x, y in itertools.combinations(METHODS, 2)))


def ranks(values):
    return {name: i + 1 for i, name in
            enumerate(sorted(METHODS, key=lambda n: values[n]))}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def kkt(E, Y, A):
    grad = E.T @ (E @ A - Y)
    active = A > 1e-7
    lagrange = -(grad * active).sum(axis=0) / active.sum(axis=0)
    reduced = grad + lagrange
    return {
        "primal_sum_max": float(np.abs(A.sum(axis=0) - 1).max()),
        "primal_negative_max": float(np.maximum(-A, 0).max()),
        "stationarity_active_max": float(np.abs(reduced[active]).max()),
        "dual_negative_max": float(np.maximum(-reduced[~active], 0).max(initial=0)),
        "complementarity_max": float(np.abs(A * reduced).max()),
    }


def independent_check(E, Y, A, rng, size=40):
    indices = rng.choice(Y.shape[1], min(size, Y.shape[1]), replace=False)
    gaps, diffs, failures = [], [], []
    p = E.shape[1]
    for i in indices:
        y = Y[:, i]
        result = minimize(
            lambda x: 0.5 * np.sum((E @ x - y) ** 2),
            np.ones(p) / p,
            jac=lambda x: E.T @ (E @ x - y),
            bounds=[(0, None)] * p,
            constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1,
                          "jac": lambda x: np.ones(p)}],
            method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
        if not result.success:
            failures.append({"index": int(i), "message": result.message})
            continue
        obj_reference = 0.5 * np.sum((E @ A[:, i] - y) ** 2)
        gaps.append(abs(float(result.fun) - obj_reference))
        diffs.append(float(np.max(np.abs(result.x - A[:, i]))))
    return {"sampled": len(indices), "success": len(gaps), "failures": failures,
            "max_absolute_objective_gap": max(gaps, default=None),
            "max_abundance_difference": max(diffs, default=None)}


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"Exact-FCLS": "FCLS", "SUnSAL": "SU",
              "SUnSAL-TV": "TV", "GLS": "GLS", "Huber": "Huber",
              "SCW-DC": "SCW", "S2-E": "S2-E", "Baseline": "Base"}
    offsets = [(4, 4), (4, -9), (-12, 4), (4, 4),
               (4, -9), (4, 4), (4, -9), (4, 4)]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0),
                             sharex=True, sharey=True, layout="constrained")
    for ax, scene in zip(axes, SCENES):
        entry = results["scene_summary"][scene]
        ar = entry["ranks"]["released"]
        rr = entry["ranks"]["reconstruction"]
        ax.plot([0.5, 8.5], [0.5, 8.5], color="#b0b0b0", linewidth=0.8)
        for method, offset in zip(METHODS, offsets):
            ax.scatter(ar[method], rr[method], color="#0F4D92", s=22, zorder=3)
            ax.annotate(labels[method], (ar[method], rr[method]),
                        xytext=offset, textcoords="offset points", fontsize=7)
        ax.set_title(f"{scene}: {entry['released_vs_reconstruction']}/28", fontsize=10)
        ax.set_xlim(0.3, 8.7)
        ax.set_ylim(8.7, 0.3)
        ax.set_xticks([1, 2, 4, 6, 8])
        ax.set_yticks([1, 2, 4, 6, 8])
        ax.set_xlabel("Released-reference RMSE rank", fontsize=9)
        ax.grid(alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Fixed-E reconstruction rank", fontsize=9)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(ROOT / "paper" / f"fig5_released_audit.{ext}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    os.environ["HSU_ROOT"] = str(ROOT)
    os.environ["HSU_DATA"] = str(args.data_dir)
    os.environ.pop("QUICK", None)
    from common import R, fcls
    from final_validate import make_case
    from round11_exactfcls import build_supports as supports11, exact_fcls as solver11
    from round22_crossmodel import clip_norm, fixed_estimators
    from xdomain_core import build_supports, exact_fcls

    source = ROOT / "theory_code" / "results" / "round22_crossmodel.json"
    saved = json.loads(source.read_text(encoding="utf-8"))
    files = [ROOT / "data" / "Urban.mat", ROOT / "data" / "end4_groundTruth.mat",
             args.data_dir / "samson.mat", args.data_dir / "jasperRidge.mat"]
    results = {"protocol": {"seed": 123, "units": UNITS,
                            "endpoint_tolerance": 1e-6,
                            "reconstruction": "observed degraded cube, common supplied E",
                            "target_probes": "linear mix of released reference and exact clean target",
                            "perturbations": [0, 0.05, 0.1],
                            "independent_check": "SLSQP, 40 random pixels per clean/noisy scene",
                            "python": platform.python_version(),
                            "numpy": np.__version__, "scipy": scipy.__version__,
                            "input_sha256": {p.name: sha256(p) for p in files}},
               "units": {}, "solver_audit": {}, "scene_summary": {}}
    rng = np.random.default_rng(49021)
    loaders = {"Urban": R.load_urban, "Samson": R.load_samson,
               "JasperRidge": R.load_jasper}
    for scene in SCENES:
        Y, E, reference, h, w, p = loaders[scene]()
        G, B, y2 = E.T @ E, E.T @ Y, (Y * Y).sum(0)
        ax, res_clean, _ = solver11(G, B, y2, supports11(p), p, Y.shape[1])
        ah = clip_norm(fcls(E, Y))
        ax_target = clip_norm(ax)
        clean_audit = {"pixels": Y.shape[1],
                       "fallback_count": int((~np.isfinite(res_clean)).sum()),
                       "kkt": kkt(E, Y, ax),
                       "slsqp": independent_check(E, Y, ax, rng)}
        results["solver_audit"][scene] = {"clean_round11": clean_audit, "degraded": {}}
        print(scene, "clean audit", clean_audit, flush=True)
        for cname, proto in UNITS:
            cond = next(c for c in R.CONDITIONS if c[0] == cname)
            Yn, obs, dead, A0, _ = make_case(Y, E, reference, h, w, 123, cond, proto)
            observed = obs > 0.5
            Yd = Yn * obs[None, :]
            afit, res, chosen = exact_fcls(G, E.T @ Yd, (Yd ** 2).sum(0),
                                           build_supports(p))
            details = {"observed_pixels": int(observed.sum()),
                       "fallback_observed": int((chosen[observed] < 0).sum()),
                       "fallback_all": int((chosen < 0).sum()),
                       "kkt_observed": kkt(E, Yn[:, observed], afit[:, observed])}
            if cname == "noisy20" and proto == "homo":
                details["slsqp"] = independent_check(
                    E, Yn[:, observed], afit[:, observed], rng)
            results["solver_audit"][scene]["degraded"][f"{cname}|{proto}"] = details
            estimates = fixed_estimators(E, Yn, obs, dead, A0, h, w)
            key = f"{scene}|{cname}|{proto}|123"
            differences = [abs(metric(estimates[m], target)
                               - saved["arm_a"][key]["rec"][tag]["rmse"][m])
                           for m in METHODS for tag, target in
                           (("lam|0.00", ah), ("lam|1.00", ax_target))]
            if max(differences) > 1e-6:
                raise ValueError(f"{key}: saved endpoint mismatch {max(differences)}")
            alt = E * (1 + 0.02 * np.sin(
                2 * np.pi * np.arange(E.shape[0])[:, None] / E.shape[0]
                + np.arange(p)[None, :] * np.pi / p))
            record = {"saved_max_endpoint_difference": max(differences),
                      "reference_to_heuristic": metric(reference, ah),
                      "reference_to_fcls": metric(reference, ax_target),
                      "reference_sum_error_max": float(np.abs(reference.sum(axis=0)-1).max()),
                      "released": {}, "heuristic": {}, "fcls_target": {},
                      "mix_005": {}, "mix_010": {}, "reconstruction": {},
                      "reconstruction_perturbed_E": {}}
            for m, A in estimates.items():
                for label, target in (("released", reference), ("heuristic", ah),
                                      ("fcls_target", ax_target),
                                      ("mix_005", .95 * reference + .05 * ax_target),
                                      ("mix_010", .90 * reference + .10 * ax_target)):
                    record[label][m] = metric(A, target)
                record["reconstruction"][m] = metric(E @ A[:, observed], Yn[:, observed])
                record["reconstruction_perturbed_E"][m] = metric(
                    alt @ A[:, observed], Yn[:, observed])
            results["units"][key] = record
            print("completed", key, "endpoint max diff", max(differences), flush=True)
        entries = [v for k, v in results["units"].items() if k.startswith(scene + "|")]
        averages = {label: {m: float(np.mean([v[label][m] for v in entries]))
                            for m in METHODS}
                    for label in ("released", "heuristic", "fcls_target", "mix_005",
                                  "mix_010", "reconstruction", "reconstruction_perturbed_E")}
        results["scene_summary"][scene] = {
            "means": averages, "ranks": {k: ranks(v) for k, v in averages.items()},
            "released_vs_reconstruction": inversions(averages["released"], averages["reconstruction"]),
            "released_vs_reconstruction_perturbed_E": inversions(
                averages["released"], averages["reconstruction_perturbed_E"]),
            "released_vs_fcls_target": inversions(averages["released"], averages["fcls_target"]),
            "released_vs_heuristic": inversions(averages["released"], averages["heuristic"]),
            "released_vs_mix_005": inversions(averages["released"], averages["mix_005"]),
            "released_vs_mix_010": inversions(averages["released"], averages["mix_010"])}
        print(scene, "summary", results["scene_summary"][scene], flush=True)
    out = ROOT / "theory_code" / "results" / "released_reference_audit.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot(results)
    print("wrote", out)


if __name__ == "__main__":
    main()
