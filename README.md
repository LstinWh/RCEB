# RCEB: Reference-Consistency Evaluation for Hyperspectral Unmixing

Research code for examining how abundance references, a fixed-endmember
linear mixing model, and evaluation metrics affect the interpretation and
ranking of hyperspectral unmixing outputs. The repository accompanies a
manuscript in preparation; it is an evaluation audit, not a claim that an
unmixing algorithm achieves state-of-the-art accuracy.

The audit compares published abundance references with model-consistent
fully constrained least-squares (FCLS) targets, contrasts abundance RMSE
rankings with reconstruction rankings under a common endmember matrix, and
measures sensitivity to constructed changes in the scoring target. It also
includes an independent DLR HySU target-region check and controlled mixtures
generated from DLR spectra. Disagreement between metrics does not establish
that a reference or either metric is wrong.

## Code map

| File | Purpose |
| --- | --- |
| `theory_code/audit_released_reference.py` | Released-reference and common-model ranking audit, perturbation probes, KKT and SLSQP checks |
| `theory_code/reproduce_reference_audit.py` | Recount archived pairwise reversals; optionally regenerate the Samson abundance maps |
| `theory_code/round22_crossmodel.py` | Fixed estimator outputs and model-target interpolation experiments |
| `theory_code/round11_exactfcls.py`, `theory_code/xdomain_core.py` | FCLS support enumeration and solver implementations |
| `theory_code/audit_dlr_hysu.py` | DLR HySU target-region AUC and reconstruction audit |
| `theory_code/audit_dlr_spatial.py` | Descriptive leave-one-spatial-block-out AUC sensitivity |
| `theory_code/audit_dlr_controlled.py` | Controlled continuous-abundance mixtures using DLR spectral libraries |
| `theory_code/plot_dlr_block_sensitivity.py` | Plot the archived spatial sensitivity results |
| `W8_code/` | Shared estimators, reference heuristic and data loaders used by the audits |
| `theory_code/results/` | Archived JSON results used for the paper's focused audits |

Some file names reflect the development history. Several comparison methods
are implementations representing different mechanisms, not necessarily exact
reproductions of their original papers. See the manuscript for definitions,
settings and interpretation.

## Environment

The recorded environment is Python 3.13 with the versions in
`requirements.txt`. From the repository root:

```bash
python -m pip install -r requirements.txt
python theory_code/reproduce_reference_audit.py
python theory_code/plot_dlr_block_sensitivity.py
```

The first command reads the archived cross-model results and recounts ranking
reversals. It does **not** rerun the estimators. The second regenerates a plot
in `paper/` from the archived DLR spatial sensitivity results.

## Data and full audit commands

Data are not distributed here. Obtain the relevant datasets independently
and check their original terms of use. Place `Urban.mat` and
`end4_groundTruth.mat` in `data/`; put `samson.mat` and `jasperRidge.mat`
in a directory passed as `PATH_TO_DATA`. The loaders expect the specific
MAT-file structures used in the study; alternative dataset versions may not
be compatible.

```bash
python theory_code/reproduce_reference_audit.py --maps --data-dir PATH_TO_DATA
python theory_code/audit_released_reference.py --data-dir PATH_TO_DATA
```

The released-reference audit reruns nine scene-condition units and compares
endpoint metrics against the archive. It records data hashes, solver
diagnostics, sampled SLSQP comparisons, rankings and constructed reference
perturbations. The map command reruns a single Samson condition.

For the DLR analyses, obtain the original DLR HySU ZIP from the
[dataset provider](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-hysu).
The study used the archive with SHA-256
`2089ef22464e0fa7d6781245b7c431b4495ca2cc75d588403bc08a9fe5ebb086`.

```bash
python theory_code/audit_dlr_hysu.py --archive PATH_TO_DLR_HySU.zip
python theory_code/audit_dlr_spatial.py --archive PATH_TO_DLR_HySU.zip
python theory_code/audit_dlr_controlled.py --archive PATH_TO_DLR_HySU.zip
python theory_code/plot_dlr_block_sensitivity.py
```

The DLR masks label target regions, not continuous abundance ground truth;
the continuous-abundance mixtures are synthetic. Spatial block deletions are
descriptive and are not independent samples or confidence intervals.

**Output warning:** full audit commands overwrite their corresponding JSON
files in `theory_code/results/` and regenerate figures in `paper/`. Preserve
the archived files before rerunning if they are needed for comparison. The
commands above cover the focused audits, not every historical experiment in
the manuscript. Dataset provenance and exact version matching for all inputs
remain to be completed.

## Citation and license

The manuscript and a formal software citation are not yet available. Please
cite the original dataset papers when using their data.

The code in this repository is released under the [MIT License](LICENSE).
Dataset rights and terms are separate and are set by their respective owners.
