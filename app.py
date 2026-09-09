import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from task import build_graph_instance
from aggregators import (
    dense_attention,
    linear_attention,
    sparse_linear_attention,
    make_sparse_projection,
    _sparse_code,
)

# ---------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Graph Associative Memory Explainer",
    page_icon="🧠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

D_EMBED = 64
D_SPARSE = 1024
NUM_QUERIES = 150


# ---------------------------------------------------------------------
# Live experiment
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_condition(log_load: int, sparsity_pct: int):
    """Run exactly the live experiment used by the notebook."""
    L = 2 ** log_load
    k_active = max(1, int(D_SPARSE * sparsity_pct / 100))

    W = make_sparse_projection(D_EMBED, D_SPARSE, seed=7)

    g = build_graph_instance(
        num_write_nodes=L,
        d=D_EMBED,
        num_queries=NUM_QUERIES,
        connection_prob=1.0,
        seed=100 + L,
    )

    accs = {}
    example = {}

    fns = {
        "dense": lambda mask, cue: dense_attention(
            g.keys, g.values, mask, cue
        ),
        "linear": lambda mask, cue: linear_attention(
            g.keys, g.values, mask, cue
        ),
        "sparse": lambda mask, cue: sparse_linear_attention(
            g.keys, g.values, mask, cue, W, k_active
        ),
    }

    for name, fn in fns.items():
        outs = np.array(
            [fn(g.neighbor_idx[q], g.cues[q]) for q in range(NUM_QUERIES)]
        )

        sims = outs @ g.values.T
        preds = sims.argmax(axis=1)
        accs[name] = float((preds == g.targets).mean())

        true_sim = float(outs[0] @ g.values[g.targets[0]])
        distractor_idx = (g.targets[0] + 1) % L
        distractor_sim = float(outs[0] @ g.values[distractor_idx])
        example[name] = (true_sim, distractor_sim)

    return g, W, k_active, accs, example


# ---------------------------------------------------------------------
# Plot helpers (existing)
# ---------------------------------------------------------------------

def plot_live_results(accs, example, g, W, k_active, sparsity_pct, L):
    names = ["dense", "linear", "sparse"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.bar(names, [accs[n] for n in names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Retrieval accuracy")
    ax.set_title(f"Retrieval accuracy\nload = {L}, {sparsity_pct}% active")

    ax = axes[1]
    x = np.arange(3)
    width = 0.35

    ax.bar(
        x - width / 2,
        [example[n][0] for n in names],
        width,
        label="TRUE value",
    )
    ax.bar(
        x + width / 2,
        [example[n][1] for n in names],
        width,
        label="DISTRACTOR value",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.axhline(0, linewidth=0.5)
    ax.set_ylabel("Cosine similarity")
    ax.set_title("One query: truth vs. wrong answer")
    ax.legend(fontsize=8)

    ax = axes[2]
    code_a = _sparse_code(g.keys[0], W, k_active) > 0
    code_b = _sparse_code(g.keys[1], W, k_active) > 0

    overlap = int((code_a & code_b).sum())
    unique_a = int(code_a.sum()) - overlap
    unique_b = int(code_b.sum()) - overlap

    ax.bar(
        ["Shared", "Unique A", "Unique B"],
        [overlap, unique_a, unique_b],
    )
    ax.set_ylabel("Active units")
    ax.set_title("Two different sparse codes\n(shared units = crosstalk risk)")

    fig.tight_layout()
    return fig


def plot_sparsity_sweep(values_x, values_y):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(values_x, values_y, marker="o")
    ax.axvline(
        5,
        linestyle="--",
        linewidth=1,
        label="BDH-GPU reported operating point (~5%)",
    )
    ax.set_xlabel("% active units")
    ax.set_ylabel("Retrieval accuracy")
    ax.set_title("Sparse attention at load = 256")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_rigorous_results(results, costs):
    loads = results["loads"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    labels = {
        "dense": "Dense softmax",
        "linear": "Linear",
        "sparse": "Sparse + non-negative",
    }

    ax = axes[0]
    for name in ["dense", "linear", "sparse"]:
        means = [r["accuracy_mean"] for r in results[name]]
        stds = [r["accuracy_std"] for r in results[name]]
        ax.errorbar(loads, means, yerr=stds, marker="o", capsize=3, label=labels[name])

    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Memory load L")
    ax.set_ylabel("Accuracy")
    ax.set_title("Retrieval accuracy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    for name in ["dense", "linear", "sparse"]:
        means = [r["fidelity_mean"] for r in results[name]]
        stds = [r["fidelity_std"] for r in results[name]]
        ax.errorbar(loads, means, yerr=stds, marker="o", capsize=3, label=labels[name])

    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Memory load L")
    ax.set_ylabel("Cosine fidelity")
    ax.set_title("Fidelity to the true value")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(costs["loads"], costs["dense_cost"], marker="o", label="Dense: L x Q")
    ax.plot(costs["loads"], costs["linear_cost"], marker="o", label="Linear / sparse: L + Q")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Memory load L")
    ax.set_ylabel("Theoretical work")
    ax.set_title("The efficiency argument")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_dimension_scaling(data):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ratios = data["ratios"]

    for d in data["d_values"]:
        mean = data["linear"][str(d)]["mean"]
        std = data["linear"][str(d)]["std"]
        ax.errorbar(ratios, mean, yerr=std, marker="o", capsize=3, label=f"d = {d}")

    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Load / embedding dimension (L / d)")
    ax.set_ylabel("Linear-attention accuracy")
    ax.set_title("Collapse tracks load / dimension")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Plot helpers (NEW)
# ---------------------------------------------------------------------

def plot_toy_graph(num_write=6, num_query=2, seed=3):
    """A small, illustrative (not-to-scale) bipartite graph diagram. The
    real experiment fully connects every query to every write node
    (connection_prob=1.0); this shows that structure at a readable size,
    with each query's TRUE target edge highlighted distinctly from its
    distractor edges. Labeled explicitly as illustrative -- the actual
    sweep tests this same structure at L up to 256, just unreadable to draw."""
    g = build_graph_instance(num_write_nodes=num_write, d=8, num_queries=num_query,
                              connection_prob=1.0, seed=seed)

    fig, ax = plt.subplots(figsize=(7, 5))
    write_y = np.linspace(0.9, 0.1, num_write)
    query_y = np.linspace(0.75, 0.25, num_query)
    write_x, query_x = 0.15, 0.85

    for q in range(num_query):
        target = g.targets[q]
        for w in range(num_write):
            if w == target:
                continue
            ax.plot([query_x, write_x], [query_y[q], write_y[w]],
                    color="lightgray", linewidth=0.8, zorder=1)
    for q in range(num_query):
        target = g.targets[q]
        ax.plot([query_x, write_x], [query_y[q], write_y[target]],
                color="#d62728", linewidth=2.2, zorder=2,
                label="TRUE target edge" if q == 0 else None)

    ax.scatter([write_x] * num_write, write_y, s=500, color="#1f77b4", zorder=3)
    for i, y in enumerate(write_y):
        ax.text(write_x, y, f"W{i}", ha="center", va="center", color="white",
                fontsize=9, fontweight="bold", zorder=4)

    ax.scatter([query_x] * num_query, query_y, s=500, color="#2ca02c", zorder=3)
    for i, y in enumerate(query_y):
        ax.text(query_x, y, f"Q{i}", ha="center", va="center", color="white",
                fontsize=9, fontweight="bold", zorder=4)

    ax.text(write_x, 1.0, "Write nodes\n(key, value)", ha="center", fontsize=10)
    ax.text(query_x, 1.0, "Query nodes\n(cue)", ha="center", fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, 1.08)
    ax.axis("off")
    ax.legend(loc="lower center", fontsize=8)
    ax.set_title(f"Illustrative graph (L={num_write} shown here for readability;\n"
                 "the sweep tests this same fully-connected structure up to L=256)")
    fig.tight_layout()
    return fig


def plot_unit_selectivity(g, W, k_active, unit_idx):
    """Does one specific sparse unit fire selectively across the stored
    write-node keys, or does it fire for most of them (little crosstalk
    protection)? A live, computed illustration of the STRUCTURAL mechanism
    behind BDH's reported monosemantic synapses -- not a claim that this
    particular random-projection unit encodes any real concept the way a
    trained BDH synapse does."""
    codes = _sparse_code(g.keys, W, k_active) > 0
    activations = codes[:, unit_idx].astype(int)
    L = len(activations)
    frac = float(activations.mean())

    fig, ax = plt.subplots(figsize=(9, 3))
    colors = ["#2ca02c" if a else "#d3d3d3" for a in activations]
    ax.bar(range(L), activations, color=colors)
    ax.set_xlabel("Write-node key index (all L stored items)")
    ax.set_ylabel(f"Unit {unit_idx}\nactive?")
    ax.set_yticks([0, 1])
    ax.set_title(f"Does this one sparse unit fire selectively?\n"
                 f"({frac:.1%} of {L} stored keys activate it)")
    fig.tight_layout()
    return fig, frac


# ---------------------------------------------------------------------
# Load precomputed rigorous results
# ---------------------------------------------------------------------

@st.cache_data
def load_json(filename):
    with open(BASE_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.title("🧠 Graph Associative Memory Explainer")

st.markdown(
    """
### Sparse codes make linear attention more trustworthy

This interactive app uses the **actual `task.py` and `aggregators.py`
implementation** from the repository -- every chart below is a live
computation, not an animation.

The central experiment compares:

- **Dense attention** — recomputes attention from raw keys/values.
- **Linear attention** — compresses the neighborhood into a fixed-size
  fast-weight matrix.
- **Sparse linear attention** — performs the same fast-weight operation after
  projecting into a high-dimensional sparse, non-negative code.
"""
)

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🕸️ The Graph",
        "🎛️ Live Explorer",
        "🔬 Sparsity Sweep",
        "🧬 BDH Connection",
        "📊 Rigorous Results",
        "📖 How It Works",
    ]
)


# =====================================================================
# TAB 1 — The graph (NEW)
# =====================================================================

with tab1:
    st.subheader("Step 0 — this is a graph, not a sequence")

    st.markdown(
        """
Every query node is connected by an edge to every write node (shown here at
a readable scale of L=6; the actual sweep tests this same structure up to
L=256). Each query must retrieve the value of the ONE write node whose key
matches its cue — its **true target edge**, highlighted in red below.
Everything else is a distractor edge the retrieval mechanism has to ignore.
"""
    )

    fig = plot_toy_graph()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "This diagram is illustrative (small L, for readability) — it is not "
        "one of the graphs used in the accuracy experiments, which use "
        "L up to 256 with the same fully-connected structure."
    )


# =====================================================================
# TAB 2 — Live explorer (predict-first added)
# =====================================================================

with tab2:
    st.subheader("Step 1 — watch linear attention break, and sparsity fix it")

    col1, col2 = st.columns(2)

    with col1:
        log_load = st.slider(
            "log₂(memory load)", min_value=1, max_value=9, value=8, step=1,
            help="The number of stored write nodes is 2^log₂(load).",
        )

    with col2:
        sparsity_pct = st.slider(
            "% active units", min_value=1, max_value=50, value=5, step=1,
            help="Percentage of units retained in the sparse code.",
        )

    L = 2 ** log_load

    st.info(f"Current setting: **{L} stored items**, **{sparsity_pct}% active sparse units**.")

    # --- Predict-first interaction ---
    guess = st.radio(
        "Before you look: at this setting, will SPARSE accuracy be higher, "
        "lower, or about the same as LINEAR accuracy?",
        ["Higher", "Lower", "About the same"],
        horizontal=True,
        key=f"guess_{log_load}_{sparsity_pct}",
    )

    with st.spinner("Running the live experiment..."):
        g, W, k_active, accs, example = run_condition(log_load, sparsity_pct)

    diff = accs["sparse"] - accs["linear"]
    if abs(diff) < 0.05:
        actual = "About the same"
    elif diff > 0:
        actual = "Higher"
    else:
        actual = "Lower"

    if guess == actual:
        st.success(f"You guessed **{guess}** — that matches. Sparse vs. linear: {diff:+.1%}.")
    else:
        st.info(f"You guessed **{guess}**; actual is **{actual}** (sparse vs. linear: {diff:+.1%}).")

    c1, c2, c3 = st.columns(3)
    c1.metric("Dense accuracy", f"{accs['dense']:.1%}")
    c2.metric("Linear accuracy", f"{accs['linear']:.1%}")
    c3.metric("Sparse accuracy", f"{accs['sparse']:.1%}")

    fig = plot_live_results(accs, example, g, W, k_active, sparsity_pct, L)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        f"""
**What changed?**

The sparse code has **{k_active} active units out of {D_SPARSE}**.
The third panel measures how many active units two different stored keys
share. More shared units means more opportunity for crosstalk when the
fast-weight memory is read.
"""
    )


# =====================================================================
# TAB 3 — Sparsity sweep (unchanged)
# =====================================================================

with tab3:
    st.subheader("Step 2 — try to break the claim")

    st.markdown(
        """
At a fixed load of **256 items**, vary the sparsity level. The notebook's
original sweep tests 1%, 2%, 5%, 10%, 20%, 30%, and 50%.
"""
    )

    run_sweep = st.button("Run sparsity sweep", type="primary")

    if run_sweep:
        grid = [1, 2, 5, 10, 20, 30, 50]
        accuracies = []
        progress = st.progress(0)

        for i, pct in enumerate(grid):
            _, _, _, acc, _ = run_condition(8, pct)
            accuracies.append(acc["sparse"])
            progress.progress((i + 1) / len(grid))

        fig = plot_sparsity_sweep(grid, accuracies)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        best_idx = int(np.argmax(accuracies))
        st.success(
            f"Best result in this single-instance sweep: "
            f"**{grid[best_idx]}% active → {accuracies[best_idx]:.1%} accuracy**."
        )
        st.caption(
            "This is the notebook's fast, single-instance experiment, "
            "not the multi-seed rigorous sweep."
        )


# =====================================================================
# TAB 4 — BDH Connection (NEW)
# =====================================================================

with tab4:
    st.subheader("Where this lives in BDH-GPU (not bolted on)")

    st.markdown(
        """
**Primary source:** Kosowski, Uznanski, Chorowski, Stamirowska,
Bartoszkiewicz. *"The Dragon Hatchling: The Missing Link between the
Transformer and Models of the Brain."* arXiv:2509.26507 (2025).

BDH-GPU's attention block builds exactly the fast-weight matrix explored in
the Live Explorer tab. The paper describes its state matrix as having
*"macro-interpretation as associative memory (like KV-cache, but organized
differently)"* (Sec. 4.1) — updated by an outer-product accumulation every
token, then read via linear (not softmax) attention.

**A correction worth being explicit about:** the `linear` aggregator in this
app (dense code, d=64) is **not** what BDH-GPU does. It represents the older
Katharopoulos et al. (2020) line of linear-attention research, which operates
in **low dimension** after preparing keys/values. BDH-GPU does something the
paper calls *"a completely different approach"*: linear attention directly in
a **high** neuronal dimension `n`, on activations forced positive by a ReLU
gate. The paper reports these activations are empirically sparse — **"only
ρ ≈ 5% of the n entries... are non-zero"** (Sec. 4.1). That's the exact
number this app's sparsity slider defaults to.

So: **`sparse` is the aggregator that actually models BDH-GPU. `linear` is
the contrast case the paper itself draws.**
"""
    )

    st.divider()
    st.markdown("### Does one sparse unit fire selectively? (live, computed here)")
    st.markdown(
        """
BDH-GPU's paper reports *monosemantic synapses* — specific units that
consistently respond to one recognizable concept (Sec. 6.3). This app can't
reproduce that on real language data, but it can show the **structural**
reason sparse coding tends toward that property: at low activity levels, a
given random-projection unit only fires for a small fraction of stored
items; raise the activity level and the same unit starts firing much more
broadly, losing selectivity. Pick a sparsity level and see for yourself:
"""
    )

    demo_sparsity = st.slider(
        "% active units (for this demo)", min_value=1, max_value=50, value=5, step=1,
        key="mono_demo_slider",
    )
    k_active_demo = max(1, int(D_SPARSE * demo_sparsity / 100))
    W_demo = make_sparse_projection(D_EMBED, D_SPARSE, seed=7)
    g_demo = build_graph_instance(num_write_nodes=256, d=D_EMBED, num_queries=1,
                                    connection_prob=1.0, seed=100 + 256)
    cue_code = _sparse_code(g_demo.cues[0], W_demo, k_active_demo) > 0
    unit_idx = int(np.argmax(cue_code)) if cue_code.any() else 0

    fig, frac = plot_unit_selectivity(g_demo, W_demo, k_active_demo, unit_idx)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        f"At {demo_sparsity}% active, this unit fires for {frac:.1%} of 256 stored "
        "keys — compare low vs. high sparsity settings above to see selectivity "
        "change. This is a structural illustration, not a claim about what any "
        "specific trained BDH synapse encodes."
    )

    st.divider()
    st.markdown(
        """
### What's live vs. cited here

| Claim | Status |
|---|---|
| Fast-weight matrix mechanism, crosstalk vs. load | **Live** — every chart in this app |
| ~5% activation sparsity (Sec. 4.1) | **Matched by design** (slider default) — not independently re-derived from a trained BDH model |
| Monosemantic synapses (Sec. 6.3) | **Structural illustration only** (unit-selectivity demo above) — not reproduced on real language data |
| Scale-free/heavy-tailed connectivity (Sec. 5) | **Not reproduced** — cited only |
"""
    )


# =====================================================================
# TAB 5 — Rigorous precomputed results (unchanged)
# =====================================================================

with tab5:
    st.subheader("Step 3 — the rigorous version")

    st.markdown(
        """
These figures come from the repository's precomputed `results.json`,
`costs.json`, and `dimension_scaling_results.json`. They are not generated
by the fast interactive slider.
"""
    )

    results = load_json("results.json")
    costs = load_json("costs.json")
    dimension_data = load_json("dimension_scaling_results.json")

    fig = plot_rigorous_results(results, costs)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()

    fig = plot_dimension_scaling(dimension_data)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        """
### What the rigorous experiment shows

- Dense attention remains near-perfect over the tested loads.
- Plain linear attention degrades as the number of stored items grows.
- Sparse linear attention preserves retrieval accuracy much longer at the
  same fast-weight mechanism.
- Fidelity can decrease even while retrieval accuracy remains high: the
  important distinction is that sparse coding preserves the **ranking** of
  the true item better, rather than making the retrieved vector immune to
  interference.
- The dimension-scaling experiment tests whether the collapse tracks
  **load / embedding dimension**, rather than one particular absolute load.
"""
    )


# =====================================================================
# TAB 6 — Explanation (unchanged)
# =====================================================================

with tab6:
    st.subheader("The mechanism")

    st.markdown(
        """
### 1. The graph

There are two types of nodes:

**Write nodes**

Each stores a `(key, value)` pair.

**Query nodes**

Each has a cue and edges to write nodes. Its job is to retrieve the value
whose key matches the cue.

The current experiment uses a fully connected query → write graph
(`connection_prob = 1.0`) — see the Graph tab for a visual.

---

### 2. Dense attention

Dense attention directly compares the query against the stored keys:

`score = key · query`

followed by a temperature-scaled softmax.

Because the raw keys remain available, every query can recompute its
attention weights. The memory is not compressed into one fixed-size
representation.

---

### 3. Linear attention

Linear attention instead constructs a fast-weight memory:

`M = Σ φ(kⱼ) ⊗ vⱼ`

and reads it with the query.

That changes the computational structure:

- building the memory: approximately **O(L)**
- each subsequent query: approximately **O(1)**

But all stored items are superposed into the same fixed-size matrix.
As `L` grows relative to the feature dimension, their representations
overlap and retrieval develops crosstalk.

---

### 4. Sparse linear attention

The sparse version first maps the key/query into a much larger dimension,
then keeps only the top-k positive activations.

The notebook uses:

- original embedding dimension: **64**
- sparse dimension: **1024**
- default active fraction: **5%**
- therefore **51 active units** at the default setting

The idea is that sparse high-dimensional codes overlap less, so the same
fast-weight mechanism can retain better selective retrieval. See the BDH
Connection tab for why this specific mechanism is the one that models
BDH-GPU, not the plain linear variant.

---

### 5. What this does NOT prove

This is a synthetic capacity/interference experiment.

The keys and values are random unit vectors rather than representations
learned by a real language model. The theoretical cost comparison is also
an operation-count argument, not a wall-clock benchmark.

The repository explicitly distinguishes the live toy experiment from
BDH-GPU's reported graph/connectivity and monosemantic-synapse findings.
"""
    )

    st.caption(
        "Source implementation: task.py, aggregators.py, experiment.py, "
        "dimension_scaling_check.py, and graph_memory_explainer.ipynb."
    )