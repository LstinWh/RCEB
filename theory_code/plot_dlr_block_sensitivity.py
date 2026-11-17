"""Plot DLR leave-one-spatial-block-out AUC differences from saved audit."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "theory_code/results/dlr_spatial_sensitivity.json"
OUT = ROOT / "paper/fig8_dlr_blocks"
MATERIALS = ("Bitumen", "Red metal", "Blue fabric", "Red fabric", "Green fabric")


def main():
    results = json.loads(DATA.read_text(encoding="utf8"))["libraries"]
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.3), layout="constrained")
    norm = TwoSlopeNorm(vmin=-0.08, vcenter=0, vmax=0.08)
    for ax, library in zip(axes, ("field", "image")):
        records = results[library]["8"]
        block_ids = sorted({x["block_id"] for m in MATERIALS for x in records[m]["drops"]})
        matrix = np.full((len(MATERIALS), len(block_ids)), np.nan)
        for i, material in enumerate(MATERIALS):
            values = {x["block_id"]: x["delta_auc"] for x in records[material]["drops"]}
            for j, block in enumerate(block_ids):
                matrix[i, j] = values.get(block, np.nan)
        image = ax.imshow(matrix, cmap="RdBu", norm=norm, interpolation="nearest", aspect="auto")
        for i in range(len(MATERIALS)):
            for j in range(len(block_ids)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i,j]:+.3f}", ha="center", va="center",
                            fontsize=5.5, color="white" if abs(matrix[i, j]) > 0.055 else "black")
        ax.set_yticks(range(len(MATERIALS)), MATERIALS, fontsize=8)
        ax.set_xticks(range(len(block_ids)), block_ids, fontsize=8)
        ax.tick_params(length=0)
        ax.set_title(f"{library.capitalize()} library", loc="left", fontsize=10)
        ax.set_xlabel("Removed spatial block ID", fontsize=8)
    fig.colorbar(image, ax=axes, label="AUC(FCLS*) - AUC(heuristic)", shrink=0.9)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT.with_suffix("." + ext), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
