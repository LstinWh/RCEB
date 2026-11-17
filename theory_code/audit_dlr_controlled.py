"""Synthetic continuous-abundance control using DLR measured spectral libraries."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.ndimage import gaussian_filter

from audit_released_reference import kkt
from round11_exactfcls import build_supports, exact_fcls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "W8_code"))
from w8_core import fcls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    raw = args.archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        libraries = {
            name: np.frombuffer(z.read("DLR_HySU/spectral_libraries/" + path),
                                dtype="<f8").reshape(6, 135).T / 10000
            for name, path in (
                ("field", "SVC_data/DLR_HySU_SVC.lib"),
                ("image", "HySpex_library/DLR_HySU_HS_library.lib"))
        }
    E = libraries["field"]
    output = {"archive_sha256": hashlib.sha256(raw).hexdigest(), "shape": [64, 64],
              "seeds": [11, 29, 47], "snr_db": 30,
              "description": "Synthetic truth; fit field library to mixtures generated from field or image library.",
              "runs": []}
    figure_data = None
    for seed in output["seeds"]:
        rng = np.random.default_rng(seed)
        fields = np.stack([gaussian_filter(rng.normal(size=(64, 64)), 5)
                           for _ in range(6)])
        fields = (fields - fields.mean(axis=(1, 2), keepdims=True)) / fields.std(
            axis=(1, 2), keepdims=True)
        weights = np.exp(fields - fields.max(axis=0, keepdims=True))
        truth = (weights / weights.sum(axis=0, keepdims=True)).reshape(6, -1)
        # Pair the noise draws across model conditions.
        unit_noise = rng.normal(size=(135, truth.shape[1]))
        for condition, generator in libraries.items():
            clean = generator @ truth
            sigma = np.linalg.norm(clean) / np.sqrt(clean.size) * 10 ** (-30 / 20)
            Y = clean + sigma * unit_noise
            A, residual, _ = exact_fcls(E.T @ E, E.T @ Y, (Y ** 2).sum(axis=0),
                                        build_supports(6), 6, Y.shape[1])
            if not np.isfinite(residual).all():
                raise RuntimeError("FCLS enumeration required fallback")
            estimates = {"heuristic": fcls(E, Y), "fcls": A}
            result = {"seed": seed, "generator": condition, "sigma": float(sigma),
                      "fallback_count": 0, "kkt": kkt(E, Y, A), "methods": {}}
            for name, estimate in estimates.items():
                result["methods"][name] = {
                    "abundance_rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))),
                    "reconstruction_rmse": float(np.sqrt(np.mean((E @ estimate - Y) ** 2)))}
            output["runs"].append(result)
            if seed == 11 and condition == "image":
                figure_data = (truth, estimates["heuristic"], A)
    out = ROOT / "theory_code/results/dlr_controlled_mixtures.json"
    out.write_text(json.dumps(output, indent=2), encoding="utf8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.1), layout="constrained")
    for row, (name, values) in enumerate(zip(("Synthetic truth", "Heuristic", "FCLS*"), figure_data)):
        for col, material in enumerate((0, 1, 2)):
            ax = axes[row, col]
            image = ax.imshow(values[material].reshape(64, 64), vmin=0, vmax=1,
                              cmap="viridis", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(("Bitumen", "Red metal", "Blue fabric")[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(name, fontsize=10)
    fig.colorbar(image, ax=axes, label="Abundance", shrink=0.8)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(ROOT / "paper" / f"fig7_dlr_controlled.{ext}", dpi=300)
    plt.close(fig)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
