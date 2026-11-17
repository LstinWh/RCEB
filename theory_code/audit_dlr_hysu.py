"""Independent DLR HySU target-ROI audit using the published data archive."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.io import loadmat
from scipy.stats import rankdata
import tifffile

from audit_released_reference import independent_check, kkt
from round11_exactfcls import build_supports, exact_fcls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "W8_code"))
from w8_core import fcls
MATERIALS = ("Bitumen", "Red metal", "Blue fabric", "Red fabric", "Green fabric")
METHODS = ("Heuristic", "FCLS")


def auc(labels, scores):
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if not n_pos or not n_neg:
        raise ValueError("Both ROI classes are required")
    ranks = rankdata(scores, method="average")
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    raw = args.archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        cube = tifffile.imread(io.BytesIO(z.read(
            "DLR_HySU/HySpex/full/DLR_HySU_HS_full.tif"))).astype(float) / 10000
        roi = loadmat(io.BytesIO(z.read(
            "DLR_HySU/HySpex/all_targets/TargetsROIs/Matlab/masks.mat")))["materials"].astype(bool)
        libraries = {}
        for name, path in (
            ("field", "SVC_data/DLR_HySU_SVC.lib"),
            ("image", "HySpex_library/DLR_HySU_HS_library.lib"),
        ):
            libraries[name] = np.frombuffer(z.read(
                "DLR_HySU/spectral_libraries/" + path), dtype="<f8").reshape(6, 135).T / 10000
    h, w, bands = cube.shape
    if (h, w, bands) != (86, 123, 135) or roi.shape != (h, w, 5):
        raise ValueError("Unexpected cube/ROI shape")
    Y = cube.reshape(-1, bands).T
    known = roi.any(axis=2).ravel()
    info = {"source": "DLR HySU, Cerra et al. 2021",
            "archive_sha256": hashlib.sha256(raw).hexdigest(),
            "cube_shape": list(cube.shape), "labeled_pixels": int(known.sum()),
            "overlapping_roi_pixels": int((roi.sum(axis=2) > 1).sum()),
            "roi_order": list(MATERIALS),
            "roi_reference": "target presence regions, not fractional abundances",
            "libraries": {}}
    maps = {}
    for name, E in libraries.items():
        G = E.T @ E
        A, residual, chosen = exact_fcls(
            G, E.T @ Y, (Y ** 2).sum(axis=0), build_supports(6), 6, Y.shape[1])
        estimates = {"Heuristic": fcls(E, Y), "FCLS": A}
        maps[name] = estimates
        entry = {"fallback_count": int((~np.isfinite(residual)).sum()),
                 "kkt": kkt(E, Y, A),
                 "slsqp": independent_check(E, Y, A, np.random.default_rng(49021)),
                 "AUC": {}, "reconstruction_rmse": {}}
        for method, abundances in estimates.items():
            per_material = {}
            for i, material in enumerate(MATERIALS):
                labels = roi[:, :, i].ravel()[known]
                per_material[material] = {
                    "auc": auc(labels, abundances[i, known]),
                    "positive_count": int(labels.sum()),
                    "negative_count": int((~labels).sum())}
            entry["AUC"][method] = per_material
            entry["reconstruction_rmse"][method] = float(
                np.sqrt(np.mean((Y - E @ abundances) ** 2)))
        info["libraries"][name] = entry
        print(name, json.dumps(entry), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.3), layout="constrained")
    for r, method in enumerate(METHODS):
        for i, material in enumerate(MATERIALS):
            ax = axes[r, i]
            image = ax.imshow(maps["field"][method][i].reshape(h, w),
                              cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            ax.contour(roi[:, :, i], levels=[0.5], colors=["#f05a3d"], linewidths=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(material, fontsize=9)
            if i == 0:
                ax.set_ylabel(method, fontsize=9)
    fig.colorbar(image, ax=axes, label="Estimated abundance", shrink=.75)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(ROOT / "paper" / f"fig6_dlr_audit.{ext}", dpi=300)
    plt.close(fig)
    out = ROOT / "theory_code/results/dlr_hysu_audit.json"
    out.write_text(json.dumps(info, indent=2), encoding="utf8")
    print("wrote", out)


if __name__ == "__main__":
    main()
