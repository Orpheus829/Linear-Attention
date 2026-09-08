# Source and License Record

## Code dependencies (all permissive open source)

| Package | License | Used for |
|---|---|---|
| numpy | BSD-3-Clause | All numerical computation (task generation, aggregators) |
| matplotlib | PSF-based (BSD-compatible) | Static plots (`accuracy_and_cost.png`, notebook plots) |
| ipywidgets | BSD-3-Clause | Interactive sliders in `graph_memory_explainer.ipynb` |
| jupyter / nbconvert | BSD-3-Clause | Notebook execution and validation (development only, not a runtime dependency for the learner) |

No package here has a copyleft or non-commercial license. `requirements.txt`
pins none of these to a specific version deliberately, to keep Binder builds
resilient to minor version drift; if reproducibility of exact numbers matters,
freeze versions before final submission.

## Data

All data (keys, values, graph structure) is synthetically generated at
runtime by `task.py` using NumPy's `default_rng`, seeded for reproducibility.
No external dataset, scraped data, or third-party data file is used anywhere
in this artifact.

## Graphics

`accuracy_and_cost.png` and all notebook plots are generated directly by
`plot_results.py` / the notebook's own matplotlib calls — original to this
project, not sourced from elsewhere. No external images, icons, or fonts are
used; all text rendering uses matplotlib's and reportlab's built-in default
fonts.

## Text and figures reused from BDH-GPU's primary source

No text, equations, or figures from Kosowski et al. (2025) are reproduced
verbatim anywhere in this submission. All technical claims attributed to
that paper in `BDH_MODULE.md`, `README.md`, and `concept_summary.pdf` are
paraphrased and cited by section number; no figure from the paper is
copied or reformatted here.

## Summary

Everything in this repository — code, data, and graphics — was generated
originally for this project. The only external intellectual content used is
factual/technical claims from cited primary sources (see README.md §Primary
sources), attributed by citation, not reproduced verbatim.
