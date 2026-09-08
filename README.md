# Sparse Codes Make Linear Attention Trustworthy: A Graph-Structured Explainer

## The one-sentence claim

On a graph, replacing dense pairwise attention with linear, per-edge message
passing lets memory scale with the number of edges instead of all node-pairs
— and constraining node activations to be sparse and non-negative on top of
that produces more selective, interference-resistant retrieval, mirroring
how BDH-GPU treats reasoning as local updates over a neuron-synapse graph
rather than global attention.

This is falsifiable: if a sparse, non-negative code did *not* reduce
crosstalk relative to a dense code at the same memory load, the claim would
be wrong. It isn't — see Results below.

## Intended learner and prerequisites

Someone comfortable with the basics of attention (softmax attention, what a
key/query/value is) and graph neural networks (nodes, edges, message
passing), who wants to understand *why* linear attention needs a sparsity
constraint to be trustworthy at scale — not just be told that it does.
No deep learning framework experience needed; the whole thing is pure NumPy.

## Learning objectives

After using this artifact, a learner should be able to:
1. State the one-sentence claim above and explain the mechanism behind it.
2. Move the load and sparsity sliders and predict, before running, which
   direction retrieval accuracy will move.
3. Explain why dense (softmax) attention never suffers this failure mode,
   while linear attention structurally does.
4. Explain where this shows up in BDH-GPU specifically — not just "some
   Transformer alternative" — and correctly distinguish BDH-GPU's actual
   mechanism from the older linear-attention literature it contrasts itself
   with.
5. Name at least one limitation of this explainer (see Limitations) and one
   open question about whether the result generalizes beyond this toy setup.

## Architecture: what's in this repo and what each piece does

| File | Role |
|---|---|
| `task.py` | Generates the synthetic bipartite graph task: write nodes (key, value) and query nodes connected by explicit edges, with a designated true target per query |
| `aggregators.py` | The three graph aggregation rules: `dense_attention` (softmax, recomputed from raw keys), `linear_attention` (fast-weight matrix, dense code), `sparse_linear_attention` (fast-weight matrix, sparse non-negative code) |
| `validate.py` | Correctness checks — run this first, always. Confirms the fast-weight matrix form is algebraically identical to a direct weighted sum, and that all three aggregators retrieve correctly with zero distractors |
| `experiment.py` | The memory-load sweep (L = 2 to 256) that produces the core result |
| `plot_results.py` | Generates `accuracy_and_cost.png` (3 panels: accuracy, fidelity, compute cost) from the sweep results |
| `dimension_scaling_check.py` | Tests whether linear attention's collapse point actually scales with embedding dimension `d`, as the theory predicts — produces `dimension_scaling.png` |
| `graph_memory_explainer.ipynb` | The interactive artifact — self-installing setup cell, sliders for load and sparsity, live retrieval panels, the BDH module woven in after the guided walkthrough, and a rigor section (Step 3) embedding the multi-seed/fidelity/dimension-scaling figures for the skeptical reader |
| `BDH_MODULE.md` | The full BDH-GPU grounding: primary-source citations, what's live vs. cited, and the correction distinguishing BDH-GPU's real mechanism from the classical linear-attention literature |
| `concept_summary.pdf` | The required one-page concept summary |
| `LICENSES.md` | Source/license record for all dependencies and data |
| `AI_DISCLOSURE.md` | What AI assistance was used and what the team is responsible for defending |
| `BUILD_LOG.md` | Phase-by-phase development history, including a real bug found and fixed — useful background for the live-defense round, not required reading for a first-time learner |

## What's live, precomputed, synthetic, or animated

| Component | Status |
|---|---|
| The three aggregators and their accuracy on any given load/sparsity setting | **Live** — recomputed from scratch every time a slider moves, on freshly sampled random graph instances |
| The fast-weight matrix ↔ direct weighted-sum equivalence | **Live, proven** — checked to float precision in `validate.py`, not asserted |
| Keys, values, graph structure | **Synthetic** — random unit vectors and a fully-connected bipartite graph (see Limitations) |
| BDH-GPU's ~5% activation sparsity, scaling-law results, Sudoku Extreme result, monosemantic synapses, scale-free connectivity | **Precomputed / cited only** — these are BDH-GPU's own reported findings from Kosowski et al. (2025), not reproduced live by this artifact. Explicitly labeled as such in `BDH_MODULE.md` and in the notebook |
| Nothing in this artifact is an animation standing in for real computation | — |

## How to reproduce

**Locally:**
```
pip install -r requirements.txt
python3 validate.py                  # confirms correctness before trusting anything else
python3 experiment.py                # reproduces the core sweep, 5 seeds per load (results.json)
python3 plot_results.py              # reproduces accuracy_and_cost.png (accuracy, fidelity, cost)
python3 dimension_scaling_check.py   # confirms the collapse point scales with d (dimension_scaling.png)
jupyter notebook graph_memory_explainer.ipynb
```

**Via Binder (no sign-in required):** push this repo to GitHub, then generate
a link at mybinder.org pointing to `graph_memory_explainer.ipynb`. Binder
builds the environment from `requirements.txt` automatically. First launch
takes ~1-2 minutes.

## Results

- **Dense attention** stays at 100% retrieval accuracy at every load tested
  (2 → 256, mean over 5 seeds, std = 0.000) — it recomputes from raw stored
  keys every query, so it never accumulates crosstalk.
- **Plain linear attention** collapses as load grows: accuracy falls from
  1.000 (L≤8) to 0.031±0.018 (L=256) — consistent with the 1/256 chance
  floor. This is the classical linear associative-memory interference
  (superposition catastrophe).
- **Sparse, non-negative linear attention** (~5% active, matching BDH-GPU's
  reported figure) holds up dramatically better at the *same* O(1)-per-query
  cost: 0.951±0.016 accuracy at L=256.
- **A more careful reading of fidelity vs. accuracy:** fidelity (raw cosine
  similarity to the true value) *decays with load for both linear and
  sparse* — sparse is not immune to magnitude decay. What sparse actually
  preserves is *relative ranking*: the true item's similarity score stays
  higher than every distractor's for far longer than it does under the
  dense-code linear variant, even as the absolute score shrinks. That
  distinction (accuracy = correct ranking survives; fidelity = absolute
  match strength) is worth being precise about rather than implying sparse
  "resists interference" in some absolute sense.
- **Dimension-scaling check (`dimension_scaling_check.py`):** the
  superposition-catastrophe explanation predicts that the collapse point
  should scale with the embedding dimension `d`, not sit at a fixed load.
  Testing d=32, 64, 128 and plotting accuracy against the ratio load/d
  confirms this: all three curves stay near 1.0 for load/d < ~0.25 and
  collapse by load/d > ~2, regardless of the absolute value of d. This
  upgrades the mechanism from "asserted" to "demonstrated."
- A follow-up sweep over sparsity level (in the notebook) shows accuracy is
  highest at 1-2% active and degrades as more units activate, with BDH-GPU's
  reported 5% sitting right at the edge of that decline — reported as a
  disclosed correlation, not a claimed causal link (BDH-GPU's number comes
  from training on real language; ours from a synthetic sweep).

See `accuracy_and_cost.png`, `dimension_scaling.png`, and `BDH_MODULE.md` for
full detail.

## Limitations (disclosed, not hidden)

- Keys/values are synthetic random unit vectors, not learned representations
  from a trained model. This isolates the capacity/interference property
  cleanly but doesn't show what happens with correlated, structured real
  embeddings — see "the most important open question" in `concept_summary.pdf`.
- The efficiency argument (dense O(load) per query vs. linear/sparse O(1)
  after one O(load) build) is a theoretical operation count, not a
  wall-clock benchmark.
- Each query in the toy task is connected to every write node
  (`connection_prob=1.0`); partial/irregular graph connectivity was not
  swept.
- BDH-GPU's reported scale-free connectivity and monosemantic-synapse
  effects are cited, not reproduced.
- The static figure (`accuracy_and_cost.png`) averages over 5 random seeds
  per load for statistical honesty; the notebook's live sliders use a single
  graph instance per setting (averaged over 150-300 queries within it) to
  keep interaction latency low. This is a disclosed and deliberate
  precision-vs-responsiveness trade-off, not an inconsistency between the
  two — the underlying pattern is the same either way, just noisier
  single-instance-to-single-instance in the notebook. The notebook's Step 3
  section embeds both rigorous figures directly so this trade-off is visible
  in-context, not just described in this README.

## Credits and licenses

See `LICENSES.md`.

## AI assistance disclosure

See `AI_DISCLOSURE.md`.

## Primary sources

1. Kosowski, Uznanski, Chorowski, Stamirowska, Bartoszkiewicz. "The Dragon
   Hatchling: The Missing Link between the Transformer and Models of the
   Brain." arXiv:2509.26507 (2025).
2. Katharopoulos et al. "Transformers are RNNs: Fast Autoregressive
   Transformers with Linear Attention." ICML 2020.
3. Haziza et al. (2025) — ReLU-driven sparse activation.
4. You et al. "Spark Transformer" (2025).
5. Buckman et al. (2024).
