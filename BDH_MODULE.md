# BDH Module (Phase 3)

**Primary source:** Kosowski, Uznanski, Chorowski, Stamirowska, Bartoszkiewicz.
"The Dragon Hatchling: The Missing Link between the Transformer and Models of
the Brain." arXiv:2509.26507 (Sept 2025). Pathway. All specific claims below
are cited to this paper (section numbers given), not to secondhand summaries.

## Which system, and why it shows up here (not generically)

The concept module targets **BDH-GPU**, the tensor-friendly variant of Dragon
Hatchling (the graph-native "BDH" and BDH-GPU are proven formally equivalent
in the paper's Observation 4; BDH-GPU is what's actually trained and
benchmarked). BDH-GPU is built from exactly two mechanisms stacked per layer:
a "ReLU-lowrank" feed-forward block, and a **linear attention** block whose
state matrix is explicitly described as having "macro-interpretation as
associative memory (like KV-cache, but organized differently)" (Section 4.1).
That state matrix — called ρ or σ in the paper — is *precisely* the fast-weight
memory matrix `M` our three aggregators build and query. This is not an
analogy we're imposing; it's the mechanism the paper names directly.

## The correction that matters: BDH is the high-dimension, sparse case — not a variant of the low-dim case

Our `linear_attention` (plain, dense, d=64) is representative of the
**classical linear-attention literature** the paper explicitly distinguishes
itself from: "A much broader line of work on linear attention for the
Transformer, initiated by Katharopoulos et al. (2020) concerns applying
linear attention in **low dimension** after appropriate preparation of keys
and values... We use a **completely different approach** to achieve attention
in **high dimension**" (Section 4.3).

BDH-GPU's actual linear attention operates directly in the large neuronal
dimension `n` (tens of thousands in real models, vs. our toy `D=1024`), on
activation vectors that are constrained positive by a ReLU gate and are
reported empirically sparse: **"only ρ ≈ 5% of the n entries of vector x_t
are non-zero"** in a typical training run (Section 4.1). That is the exact
number our `sparse_linear_attention` variant was built to match — not a
loosely-inspired parallel, the same reported figure.

So the honest mapping is:
- **dense_attention** ↔ no direct BDH analogue; included as the "always
  correct, always expensive" reference point.
- **linear_attention** (our plain, dense-code version) ↔ the Katharopoulos-
  style linear attention literature BDH-GPU explicitly contrasts itself with.
- **sparse_linear_attention** (~5% active) ↔ **BDH-GPU's actual mechanism.**

## What's changing, concretely

Per the paper's own equations (Eq. 8, Fig. 3): at each layer, state
`ρ_{t,l} := ρ_{t-1,l} + LN(E·y_{t,l-1}) · x_{t,l}^T` — an outer-product
accumulation into a fixed-size matrix, updated once per token, read via a
linear (not softmax) attention operation. This is algebraically the same
"build M once via outer-product sums, read via matrix-vector product"
structure validated in `validate.py`. What's changing as a BDH model
processes tokens is exactly what changes in our toy: a fast-weight matrix,
not the model's trained parameters (E, Dx, Dy stay fixed during inference —
Section 3.2).

## Primary-sourced numbers cited (not reproduced live)

| Claim | Source | Live in our substrate? |
|---|---|---|
| ~5% activation sparsity in x_t | Section 4.1, Empirical Finding 1 | **Matched by design** (our sparsity slider defaults to 5%) — not independently re-derived from a trained BDH model |
| Monosemantic synapses, present even <100M-param models | Section 6.3 | Cited only — our toy's "code overlap" panel shows the *mechanism* (why sparsity reduces overlap/crosstalk) but does not reproduce concept-level monosemanticity in language |
| Neuron-neuron interaction graph shows high Newman modularity, heavy-tailed/power-law degree distribution | Section 5, Section 5.5 | **Not reproduced** — explicitly out of scope, disclosed in README limitations |
| BDH-GPU matches GPT2-class Transformer scaling laws, 10M–1B params | Section 4.2 | Cited only, not relevant to this concept's claim |

## Recent primary papers (2022–2026) for the required citation list

1. Kosowski et al., "The Dragon Hatchling..." (2025) — arXiv:2509.26507
2. Haziza et al. (2025) — first empirical demonstration of ReLU-driven sparse
   activation as a systematic mechanism (cited in BDH paper, Section 4.3)
3. You et al., "Spark Transformer" (2025) — achieves 8% neuron activation
   sparsity via top-k + soft thresholding, a different mechanism reaching a
   comparable sparsity regime — useful comparison point for our summary's
   "representative architectures" table
4. Buckman et al. (2024) — alternative approach to eliminating attention
   nonlinearity via tensor-product key preparation (cited in BDH paper,
   Section 4.3, as a contrasting design choice to BDH-GPU's high-dimension
   approach)

## Limitation to disclose

Our toy graph task uses synthetic random keys/values to isolate the
capacity/interference property cleanly. BDH-GPU's actual sparsity emerges
from training on language data, not from a design-time random projection —
we deliberately match the *reported number* (5%) to ground the toy in a real
finding, but we are not claiming our random-projection sparse code is
mechanistically identical to what BDH learns.

## What our own reference sweep actually showed

Running our own sweep over sparsity level at the hardest load tested (256):
accuracy is highest at very low activity (1-2% active, ~0.99-1.0) and
degrades steadily as more units activate (0.92 at 5%, 0.39 at 10%, 0.02 at
50%) — the sparse-coding/crosstalk argument holds directionally in our toy
setup. BDH's reported 5% operating point sits right at the edge of that
decline. **This is a correlation worth noting, not evidence of causation**:
we are not claiming BDH's designers chose ~5% *because* of this specific
capacity trade-off — their number emerged from training dynamics on real
language data, ours from a synthetic retrieval-capacity sweep. The honest
claim is narrower: our toy mechanism and BDH's reported operating point are
consistent with the same qualitative trade-off, which is itself a real,
useful pedagogical point.
