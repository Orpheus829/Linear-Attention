# AI Assistance Disclosure

## What was AI-assisted

This submission was built with substantial assistance from Claude (Anthropic)
as a coding and research assistant, used interactively throughout development:

- **Concept selection**: the team directed the choice of hybrid topic (Linear
  Attention + Sparse Non-Negative Activations + Local Neural Computation,
  applied to graph topology) after a discussion of the approved topic list;
  Claude proposed candidate combinations and trade-offs, the team chose the
  direction and specified the graph-topology pivot.
- **Code**: `task.py`, `aggregators.py`, `validate.py`, `experiment.py`,
  `plot_results.py`, and the notebook `graph_memory_explainer.ipynb` were
  written by Claude under the team's direction, including designing the
  associative-retrieval-on-a-graph task and the three aggregation mechanisms.
- **Debugging**: a real bug was found during development — the standard
  `elu(x)+1` linear-attention feature map produced a near-constant offset
  that masked the intended signal for this task's embedding scale. Claude
  diagnosed the issue numerically and replaced it with a plain ReLU feature
  map, which produced the intended graded degradation curve. Documented in
  `BUILD_LOG.md`.
- **Primary-source research**: Claude fetched and read the BDH primary paper
  (arXiv:2509.26507) directly to ground `BDH_MODULE.md` and
  `concept_summary.pdf`, including identifying a correction to an earlier,
  looser mapping between the toy aggregators and BDH-GPU's actual mechanism.
- **Writing**: `README.md`, `BDH_MODULE.md`, and `concept_summary.pdf` were
  drafted by Claude and reviewed/directed by the team.

## What the team is responsible for and can defend

Per the hackathon rules, the team must understand, trace, and defend every
major component regardless of how it was produced. Before submission, the
team should be able to, without assistance:

- Explain why `linear_attention`'s fast-weight matrix form is mathematically
  identical to the direct weighted-sum form (not just that `validate.py`
  confirms it, but *why*).
- Explain the superposition-catastrophe mechanism behind the load-vs-accuracy
  curve, and predict the direction of the effect for a setting not shown in
  the notebook.
- Explain, without notes, why BDH-GPU's actual mechanism corresponds to the
  *sparse* aggregator and not the plain *linear* one — this is the single
  most important technical distinction in the submission and the one most
  likely to be probed in a live defense.
- Explain what the `elu+1` bug was, why it happened, and why plain ReLU fixed
  it.
- Explain which claims in `BDH_MODULE.md` and `concept_summary.pdf` are
  live/reproduced versus cited-only, and why.

## What is not claimed

No part of this submission is presented as reproducing BDH-GPU itself, an
official BDH checkpoint, or BDH-GPU's own training results. The toy
substrate is explicitly labeled throughout as an independent, synthetic
illustration of a mechanism BDH-GPU is reported to use — not a
reimplementation of BDH-GPU, and not officially affiliated with or endorsed
by Pathway.
