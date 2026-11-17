"""Leave-one-spatial-block-out sensitivity of the DLR ROI AUC comparison."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.io import loadmat
import tifffile

from audit_dlr_hysu import MATERIALS, auc
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
        cube = tifffile.imread(io.BytesIO(z.read(
            "DLR_HySU/HySpex/full/DLR_HySU_HS_full.tif"))).astype(float) / 10000
        roi = loadmat(io.BytesIO(z.read(
            "DLR_HySU/HySpex/all_targets/TargetsROIs/Matlab/masks.mat")))["materials"].astype(bool)
        libraries = {
            name: np.frombuffer(z.read("DLR_HySU/spectral_libraries/" + path),
                                dtype="<f8").reshape(6, 135).T / 10000
            for name, path in (
                ("field", "SVC_data/DLR_HySU_SVC.lib"),
                ("image", "HySpex_library/DLR_HySU_HS_library.lib"))
        }
    h, w, bands = cube.shape
    Y = cube.reshape(-1, bands).T
    known = roi.any(axis=2).ravel()
    rows, cols = np.indices((h, w))
    output = {"archive_sha256": hashlib.sha256(raw).hexdigest(),
              "definition": "AUC(FCLS*) minus AUC(heuristic); remove each occupied grid block",
              "limitations": "Descriptive sensitivity only; blocks are not independent replicates.",
              "libraries": {}}
    for library, E in libraries.items():
        A, residual, _ = exact_fcls(E.T @ E, E.T @ Y, (Y ** 2).sum(axis=0),
                                    build_supports(6), 6, Y.shape[1])
        heuristic = fcls(E, Y)
        if not np.isfinite(residual).all():
            raise RuntimeError("FCLS support enumeration required fallback")
        library_out = {}
        for size in (4, 8):
            block_ids = (rows // size * ((w + size - 1) // size) + cols // size).ravel()
            records = {}
            for i, material in enumerate(MATERIALS):
                labels = roi[:, :, i].ravel()
                def difference(mask):
                    return auc(labels[mask], A[i, mask]) - auc(labels[mask], heuristic[i, mask])
                full = difference(known)
                drops = []
                for block in np.unique(block_ids[known]):
                    keep = known & (block_ids != block)
                    if labels[keep].any() and (~labels[keep]).any():
                        drops.append({"block_id": int(block),
                                      "removed_pixels": int((known & (block_ids == block)).sum()),
                                      "delta_auc": difference(keep)})
                values = [x["delta_auc"] for x in drops]
                records[material] = {"full_delta_auc": full, "n_blocks": len(values),
                                     "min_delta_auc": min(values), "max_delta_auc": max(values),
                                     "positive": sum(v > 0 for v in values),
                                     "negative": sum(v < 0 for v in values),
                                     "zero": sum(v == 0 for v in values), "drops": drops}
            library_out[str(size)] = records
        output["libraries"][library] = library_out
    out = ROOT / "theory_code/results/dlr_spatial_sensitivity.json"
    out.write_text(json.dumps(output, indent=2), encoding="utf8")
    print(json.dumps({lib: {size: {m: {k: v for k, v in rec.items() if k != "drops"}
                                   for m, rec in entries.items()}
                            for size, entries in sizes.items()}
                      for lib, sizes in output["libraries"].items()}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
