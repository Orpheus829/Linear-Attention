# Graph Associative Retrieval: Dense vs. Linear vs. Sparse Attention

**Phase 1 deliverable** — the core substrate, validated and run for real. This is not
an animation; every number below comes from the scripts in this folder.

## The one-sentence claim

> On a graph, replacing dense pairwise attention with linear, per-edge message
> passing lets memory scale with the number of edges instead of all node-pairs —
> and constraining node activations to be sparse and non-negative on top of that
> produces more selective, interference-resistant retrieval, mirroring how BDH
> treats reasoning as local updates over a neuron-synapse graph rather than
> global attention.

This is falsifiable: if sparse+non-negative coding did *not* reduce crosstalk
relative to dense linear coding at the same memory load, the claim would be
wrong. It isn't — see results below.

## The task

A bipartite graph: **write nodes** each hold a (key, value) pair; **query
nodes** are connected by edges to a set of write nodes and must retrieve the
value of the one write node whose key matches the query's cue. "Memory load"
(`L`) = how many write nodes are packed into the graph. This is genuinely
graph-structured (explicit edge/neighbor lists), not a sequence in disguise.

Three aggregation (message-passing) rules over the *same* graph:

| Rule | Mechanism | Per-query cost |
|---|---|---|
| **Dense** | Softmax attention, recomputed from raw stored keys/values every query | O(L) every query |
| **Linear** | No softmax; feature map → one fixed d×d fast-weight matrix built once, queried in O(1) (Katharopoulos et al. 2020 formulation) | O(1) after O(L) build |
| **Sparse** | Same fast-weight mechanism, but keys/queries go through a sparse (~5% active) non-negative random-projection code before the matrix, matching BDH's reported neuron activation sparsity | O(1) after O(L) build |

## Results (see `accuracy_and_cost.png`)

- **Dense** stays at 100% accuracy at every load tested (2 → 256) — it never
  compresses memory, so it never accumulates crosstalk. Cost: grows linearly
  in both load and number of queries (has to rescan raw keys every time).
- **Linear** collapses as load grows: accuracy falls from 1.0 (L≤4) to 0.053
  (L=256) — barely above the 1/256 chance floor. This is the classical linear
  associative-memory interference (superposition catastrophe): packing 256
  items into a fixed 64-dimensional matrix causes them to bleed into each
  other. Even where accuracy still looks fine (L=8, acc=0.98), fidelity has
  already dropped to 0.34 — the retrieval is barely correct, not confidently
  correct.
- **Sparse** holds up dramatically better at the *same* O(1)-per-query cost:
  0.957 accuracy at L=256, where plain linear is at 0.053. Sparsifying the
  code (not changing the retrieval mechanism at all) recovers most of dense
  attention's robustness without giving up linear attention's efficiency.

## Validation (`validate.py`)

Before trusting any of the above:
1. The fast-weight matrix form of linear/sparse attention was checked
   against a direct weighted-sum computation — they match to float precision
   (~1e-16), confirming the "one matrix built once" framing is not just an
   approximation of the weighted-sum form, it's algebraically identical to it.
2. A trivial no-distractor case (L=1) gives ~1.0 cosine similarity for all
   three aggregators — sanity that nothing is broken before introducing load.

## An honest bug we hit and fixed (worth keeping in the writeup)

The textbook linear-attention feature map, `elu(x)+1` (Katharopoulos et al.
2020), was tried first and **failed immediately**, even at L=2. Diagnosis: for
small, zero-mean, unit-norm vectors in 64 dimensions, that feature map adds a
near-constant offset of magnitude ≈ d (measured: 65.2, vs. true signal
variation of only ≈1.2) to every key–query dot product — the actual
correlation is completely swamped by the offset. This is a real, known
pitfall of that feature map outside its usual operating regime (larger-scale
learned representations), not a property of linear attention itself. Fixed
by switching to a plain ReLU feature map (no additive offset), which produced
the graded, load-dependent degradation curve above. This is a genuine
"limitation encountered and resolved," not manufactured for the demo.

## Known limitations to disclose in the final submission

- Keys/values are synthetic random unit vectors, not learned representations
  from a trained model — this isolates the *capacity/interference* property
  cleanly, but doesn't show what happens when keys are structured/correlated
  (as real learned embeddings are).
- The efficiency comparison (`dense_cost` vs. `linear_cost`) is a theoretical
  operation count, not a wall-clock benchmark.
- The scale-free/heavy-tailed connectivity property BDH reports is **not**
  reproduced live in this substrate — it will be presented as a clearly
  labeled precomputed BDH result in the BDH module (Phase 3), not implied to
  come from this experiment.

## Files

- `task.py` — graph/task generator
- `aggregators.py` — the three aggregation rules
- `validate.py` — correctness checks (run this first, always)
- `experiment.py` — the load sweep (produces `results.json`, `costs.json`)
- `plot_results.py` — produces `accuracy_and_cost.png`

## Next (Phase 2 — done)

`graph_memory_explainer.ipynb` — the interactive notebook. Two sliders
(memory load, % active units), three live panels (accuracy bars, one
query's truth-vs-distractor similarity, sparse-code overlap between two
items). Verified to execute cleanly end-to-end via `jupyter nbconvert
--execute` before packaging — not just "it parses."

**To deploy on Binder (no sign-in required to run):**
1. Push this whole folder to a public GitHub repo (needs `task.py`,
   `aggregators.py`, `requirements.txt`, and the `.ipynb` all in the repo
   root, or the notebook's imports will fail).
2. Go to mybinder.org, paste the repo URL, set the path to
   `graph_memory_explainer.ipynb`, and generate the link. Binder builds the
   environment from `requirements.txt` automatically.
3. The generated `https://mybinder.org/v2/gh/<user>/<repo>/HEAD?filepath=graph_memory_explainer.ipynb`
   link is what goes in the submission as the public artifact URL. First
   launch takes ~1-2 minutes to build; subsequent launches are faster if the
   build is cached.

## Next (Phase 3 — done)

`BDH_MODULE.md` — the full BDH module writeup, grounded directly in the
primary source (Kosowski et al., arXiv:2509.26507), woven into the notebook
(not appended at the end) right after the guided interactive panel. Key
points:

- BDH-GPU's attention state is explicitly described by the paper as
  "associative memory (like KV-cache, but organized differently)" — the same
  fast-weight matrix our aggregators build and query, not an analogy we
  imposed.
- **Correction made explicit in the module:** our plain `linear_attention`
  represents the older, low-dimension Katharopoulos-style linear attention
  literature that the BDH paper explicitly contrasts itself with. BDH-GPU's
  actual mechanism is our `sparse_linear_attention` — linear attention in a
  large, sparse, positive neuronal dimension, matching the paper's reported
  ~5% activation sparsity (Section 4.1) exactly, not loosely.
- A live reference sweep (added to the notebook) shows our own toy accuracy
  is highest at 1-2% active and degrades as sparsity decreases, with BDH's
  reported 5% sitting right at the edge of that decline — reported honestly
  as a correlation between two independently-arrived-at numbers, not a
  claimed causal link.
- What's live (the fast-weight mechanism, the crosstalk/sparsity trade-off)
  vs. cited-only (BDH's reported scale-free connectivity, monosemantic
  synapses on real language) is stated explicitly, both in `BDH_MODULE.md`
  and in the notebook cell itself.
- 4 primary sources (2022–2026) identified for the required citation list:
  the BDH paper itself, Haziza et al. 2025, Spark Transformer (You et al.
  2025), and Buckman et al. 2024 — all surfaced from the BDH paper's own
  citations, not secondhand summaries.

## Next (Phase 4 — done)

`concept_summary.pdf` — one-page (788 words), covering: the design pressure
behind linear attention, what BDH-GPU changes technically and its trade-off,
a compact 3-way architecture comparison table (softmax / classical linear
attention / BDH-GPU), correctly-labeled evidence (BDH-GPU's scaling-law and
Sudoku Extreme results are flagged as developer-reported, not independently
reproduced — the Sudoku caveat is stated explicitly in Pathway's own repo),
the roles of BDH vs. BDH-CQ (BDH-CQ has no direct role here, stated plainly
rather than forcing a connection), and the most important open question
(does the interference-reduction argument survive on real, correlated,
trained codes rather than synthetic random ones).

## Next (Phase 5 — packaging, not started)

## Rigor upgrade (post-Phase 5, user-requested)

After reviewing the hero figure, three gaps were identified and closed:
1. **Multi-seed averaging.** `experiment.py` now runs 5 independent random
   graph instances per load and reports mean +/- std, instead of a single
   instance per load. Revealed real seed-to-seed variance previously hidden
   (e.g., linear attention at L=16: 0.918 +/- 0.083).
2. **Fidelity overlay.** `plot_results.py` now has 3 panels instead of 2,
   adding fidelity (raw cosine similarity to the true value) alongside
   accuracy. This surfaced a nuance worth being precise about: fidelity
   decays with load for BOTH linear and sparse attention -- sparse is not
   immune to magnitude decay. What sparse actually preserves is relative
   ranking (the true item stays the argmax far longer), not absolute match
   strength. Documented explicitly in README to avoid overclaiming.
3. **Dimension-scaling check.** New `dimension_scaling_check.py` tests
   whether linear attention's collapse point actually scales with embedding
   dimension d (as the superposition-catastrophe theory predicts), by
   plotting accuracy against load/d for d=32, 64, 128. Curves collapse onto
   a common shape as predicted (near-1.0 for load/d < ~0.25, collapsed by
   load/d > ~2, regardless of absolute d) -- upgrading this from an asserted
   mechanism to a demonstrated one.

Deliberately NOT added, to avoid diluting the one central claim: sweeping
partial graph connectivity, sweeping D_sparse independently of % active, or
wall-clock timing benchmarks. These remain disclosed limitations, not chart
lines.

Public artifact URL (Binder), public repo, this blog/summary as PDF (done),
full README (this file, needs final pass), license/source record for any
reused assets, AI-assistance disclosure.
