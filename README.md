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

## Citation and license

The manuscript and a formal software citation are not yet available. Please
cite the original dataset papers when using their data.

The code in this repository is released under the [MIT License](LICENSE).
Dataset rights and terms are separate and are set by their respective owners.
