"""
Validation checks, run before trusting the sweep experiment.

1. Equivalence check: linear_attention's matrix form
       out = phi(q)^T M / (phi(q).z),  M = sum_j phi(k_j) outer v_j
   must exactly match the direct weighted-sum form
       out = sum_j w_j v_j / sum_j w_j,  w_j = phi(q).phi(k_j)
   These are algebraically identical; if they don't match numerically,
   there's a bug in the matrix implementation, not a modeling choice.

2. Same equivalence check for the sparse-coded variant.

3. Trivial sanity case: L=1 (no distractors at all) -> every aggregator
   must retrieve the true value almost exactly (cosine similarity ~ 1),
   since there is nothing to interfere with.
"""
import numpy as np
from task import build_graph_instance
from aggregators import (
    dense_attention,
    linear_attention,
    sparse_linear_attention,
    make_sparse_projection,
    _relu_feature,
    _sparse_code,
)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def check_linear_attention_equivalence():
    rng = np.random.default_rng(0)
    L, d = 6, 5
    keys = rng.normal(size=(L, d))
    values = rng.normal(size=(L, d))
    mask = np.array([True, True, False, True, False, True])
    cue = rng.normal(size=d)

    matrix_form = linear_attention(keys, values, mask, cue)

    phi_k = _relu_feature(keys)
    phi_q = _relu_feature(cue)
    w = np.array([mask[j] * (phi_q @ phi_k[j]) for j in range(L)])
    direct_form = (w[:, None] * values).sum(axis=0) / (w.sum() + 1e-6)

    diff = np.max(np.abs(matrix_form - direct_form))
    print(f"[linear_attention] max abs diff between matrix form and direct sum: {diff:.2e}")
    assert diff < 1e-8, "linear_attention matrix form does not match direct weighted sum!"


def check_sparse_linear_attention_equivalence():
    rng = np.random.default_rng(1)
    L, d, D, k_active = 6, 5, 40, 4
    keys = rng.normal(size=(L, d))
    values = rng.normal(size=(L, d))
    mask = np.array([True, False, True, True, False, True])
    cue = rng.normal(size=d)
    W = make_sparse_projection(d, D, seed=2)

    matrix_form = sparse_linear_attention(keys, values, mask, cue, W, k_active)

    phi_k = _sparse_code(keys, W, k_active)
    phi_q = _sparse_code(cue, W, k_active)
    w = np.array([mask[j] * (phi_q @ phi_k[j]) for j in range(L)])
    direct_form = (w[:, None] * values).sum(axis=0) / (w.sum() + 1e-6)

    diff = np.max(np.abs(matrix_form - direct_form))
    print(f"[sparse_linear_attention] max abs diff between matrix form and direct sum: {diff:.2e}")
    assert diff < 1e-8, "sparse_linear_attention matrix form does not match direct weighted sum!"


def check_no_distractor_case():
    g = build_graph_instance(num_write_nodes=1, d=16, num_queries=1, connection_prob=1.0, seed=3)
    W = make_sparse_projection(16, 256, seed=4)
    mask = g.neighbor_idx[0]
    cue = g.cues[0]
    target_val = g.values[g.targets[0]]

    out_dense = dense_attention(g.keys, g.values, mask, cue)
    out_linear = linear_attention(g.keys, g.values, mask, cue)
    out_sparse = sparse_linear_attention(g.keys, g.values, mask, cue, W, k_active=13)

    for name, out in [("dense", out_dense), ("linear", out_linear), ("sparse", out_sparse)]:
        sim = cosine(out, target_val)
        print(f"[no-distractor sanity] {name}: cosine similarity to true value = {sim:.4f}")
        assert sim > 0.95, f"{name} failed trivial no-distractor retrieval (sim={sim:.4f})"


if __name__ == "__main__":
    check_linear_attention_equivalence()
    check_sparse_linear_attention_equivalence()
    check_no_distractor_case()
    print("\nAll validation checks passed.")
