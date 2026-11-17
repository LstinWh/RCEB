"""Recount saved ranking reversals and optionally reproduce Samson maps."""
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "theory_code" / "results" / "round22_crossmodel.json"
SCENES = ("Urban", "Samson", "JasperRidge")


def recount(data, scenes):
    methods = data["meta"]["methods"]
    units = [v for k, v in data["arm_a"].items()
             if k.split("|")[0] in scenes]
    means = {}
    for endpoint in ("lam|0.00", "lam|1.00"):
        means[endpoint] = {
            m: np.mean([u["rec"][endpoint]["rmse"][m] for u in units])
            for m in methods
        }
    return sum(
        (means["lam|0.00"][a] - means["lam|0.00"][b])
        * (means["lam|1.00"][a] - means["lam|1.00"][b]) < 0
        for a, b in itertools.combinations(methods, 2)
    ), len(units)


def maps(data, data_dir):
    os.environ["HSU_ROOT"] = str(ROOT)
    os.environ["HSU_DATA"] = str(data_dir)
    os.environ.pop("QUICK", None)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common import R, fcls, rmse
    from final_validate import make_case
    from round22_crossmodel import clip_norm, exact_clean
    from xdomain_core import build_supports, exact_fcls
    from w8_core import w8_unmix

    Y, E, A_ref, height, width, p = R.load_samson()
    cond = next(c for c in R.CONDITIONS if c[0] == "noisy20")
    Yn, obs, dead, A0, _ = make_case(
        Y, E, A_ref, height, width, 123, cond, "homo")
    Yd = Yn * obs[None, :]
    A_fit, _, _ = exact_fcls(E.T @ E, E.T @ Yd, (Yd ** 2).sum(0),
                             build_supports(p))
    A_fit = clip_norm(A_fit)
    _, A_scw = w8_unmix(Yn, obs, E, A0, dead)
    A_scw = clip_norm(A_scw)
    A_fit[:, dead] = A0[:, dead]
    A_scw[:, dead] = A0[:, dead]
    A_h = clip_norm(fcls(E, Y))
    A_x = clip_norm(exact_clean(E, Y))
    saved = data["arm_a"]["Samson|noisy20|homo|123"]["rec"]
    for name, estimate in (("Exact-FCLS", A_fit), ("SCW-DC", A_scw)):
        for tag, reference in (("lam|0.00", A_h), ("lam|1.00", A_x)):
            value = rmse(estimate, reference)
            expected = saved[tag]["rmse"][name]
            print(f"{name} {tag}: {value:.10f} (saved {expected:.10f})")
            if abs(value - expected) > 1e-6:
                raise ValueError("Recomputed RMSE differs from saved result")

    columns = [("Released reference", A_ref), ("Heuristic target", A_h),
               ("Exact clean target", A_x), ("Exact-FCLS estimate", A_fit),
               ("SCW-DC estimate", A_scw)]
    fig, axes = plt.subplots(p, len(columns), figsize=(10, 6.2),
                             layout="constrained")
    for row in range(p):
        for col, (name, matrix) in enumerate(columns):
            ax = axes[row, col]
            image = ax.imshow(matrix[row].reshape(height, width), cmap="viridis",
                              vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"Component {row + 1}", fontsize=9)
    fig.colorbar(image, ax=axes, label="Abundance", shrink=0.7, pad=0.02)
    for extension in ("pdf", "png", "svg"):
        path = ROOT / "paper" / f"fig4_abundance_comparison.{extension}"
        fig.savefig(path, dpi=300)
        print(f"Wrote {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", action="store_true")
    parser.add_argument("--data-dir", type=Path,
                        help="Directory containing samson.mat")
    args = parser.parse_args()
    with RESULTS.open(encoding="utf-8") as stream:
        data = json.load(stream)
    for scenes in [(scene,) for scene in SCENES] + [SCENES]:
        count, units = recount(data, scenes)
        print(f'{"+".join(scenes)}: {count}/28, {units} units')
    if args.maps:
        if args.data_dir is None:
            parser.error("--maps requires --data-dir")
        maps(data, args.data_dir)


if __name__ == "__main__":
    main()
