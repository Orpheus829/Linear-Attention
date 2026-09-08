"""
Three graph aggregation (message-passing) rules over the same
write-node -> query-node bipartite graph.

1. dense_attention:
   Standard scaled dot-product softmax attention (GAT-style). Recomputes
   weights fresh, from the RAW stored (key, value) pairs, for every query.
   Cost per query ~ O(|neighbors|). Never compresses memory into a fixed
   object, so it doesn't suffer capacity interference -- but it must
   re-scan every neighbor's raw key/value for every single query.

2. linear_attention:
   Removes the softmax (Katharopoulos et al. 2020 "linear attention as an
   RNN" formulation). A nonnegative feature map phi(x) = elu(x) + 1 is
   required because linear attention's normalization breaks down with
   possibly-negative weights. This is mathematically equivalent to
   building one fixed-size fast-weight memory matrix
       M = sum_j phi(k_j) outer v_j
   once per neighborhood, then answering a query in O(1) via
       out = phi(q)^T M / (phi(q) . z)
   That incremental-update property is the actual efficiency story: for
   many queries against the same memory, dense attention pays O(L) again
   per query, linear attention pays O(L) once then O(1) per query. The
   trade-off: compressing L stored pairs into one fixed d x d matrix means
   the stored keys' feature vectors start to overlap (they are only
   approximately orthogonal), so retrievals pick up crosstalk from OTHER
   stored items as L grows relative to d. This is the classical linear
   associative-memory interference (superposition catastrophe).

3. sparse_linear_attention:
   Same fast-weight formulation as (2), but the feature map projects into
   a much higher dimension D >> d and keeps only the top-k activations
   (ReLU + top-k), giving a sparse, non-negative code -- the property BDH
   reports for its own neuron activations (~5% active, non-negative).
   Sparse high-dimensional codes are far closer to mutually orthogonal
   than dense low-dimensional ones (classical sparse-coding / sparse
   distributed memory capacity argument), so the SAME incremental
   fast-weight mechanism as (2) should show less crosstalk / higher
   effective capacity, without giving up the O(1)-per-query efficiency.
"""
import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def dense_attention(keys, values, neighbor_mask, cue, temperature=0.05):
    """
    keys: (L, d), values: (L, d), neighbor_mask: (L,) bool, cue: (d,)
    Returns: (d,) retrieved vector.

    NOTE on temperature: the usual Transformer convention divides by
    sqrt(d) because raw (unnormalized) query/key dot products grow with
    d. Here keys/queries are explicitly UNIT-NORMALIZED (a design choice
    for this task), so raw dot products are bounded cosine similarities
    in [-1, 1] -- the true match sits near +1, unrelated keys sit near
    0 with a noise floor that shrinks as ~1/sqrt(d). Reusing the sqrt(d)
    convention on top of already-bounded scores under-sharpens the
    softmax and lets accumulated noise from many distractors drown out a
    single true signal. A small fixed temperature (default 0.05) sharpens
    the softmax onto the cosine scale instead. This is a real, common
    practical knob (temperature-scaled cosine attention / kNN-style
    retrieval), not a fudge to make the demo work -- see the growth-vs-L
    stress test in validate.py, which confirms dense attention stays
    robust up to the largest loads tested precisely BECAUSE it can be
    sharpened this way, per-query, from the raw keys -- a degree of
    freedom linear/sparse attention structurally lack (their crosstalk is
    baked into a shared matrix before any query arrives).
    """
    scores = keys @ cue / temperature            # (L,)
    scores = np.where(neighbor_mask, scores, -1e9)  # mask out non-neighbors
    weights = _softmax(scores)                    # (L,)
    return weights @ values                       # (d,)


def _relu_feature(x):
    return np.maximum(x, 0.0)


def linear_attention(keys, values, neighbor_mask, cue, eps=1e-6):
    """
    Fast-weight / linear-attention formulation. Mathematically equivalent
    to building M = sum_j phi(k_j) outer v_j over neighbors and reading
    out = phi(cue)^T M / (phi(cue) . z). Implemented via the explicit
    matrix form here so the equivalence can be checked directly against
    the weighted-sum form in validate.py.

    Feature map: plain ReLU, phi(x) = max(x, 0). NOTE: the textbook
    Katharopoulos et al. (2020) choice elu(x)+1 was tried first here and
    rejected -- for small, zero-mean, unit-norm vectors in d=64 dims it
    adds a near-constant offset of magnitude ~d to every dot product
    (measured: offset ~64 vs. true signal variation ~1.2), which drowns
    the actual key-query correlation almost entirely regardless of load.
    That is a real, documented pitfall of that feature map outside its
    usual (larger-magnitude, learned-representation) operating regime --
    not a property of linear attention itself. Plain ReLU has no additive
    offset, stays non-negative, and preserves the correlation structure
    that the crosstalk argument below depends on.
    """
    phi_k = _relu_feature(keys)          # (L, d)
    phi_q = _relu_feature(cue)           # (d,)
    phi_k = phi_k * neighbor_mask[:, None]  # zero out non-neighbor rows

    M = phi_k.T @ values                  # (d, d) fast-weight memory matrix
    z = phi_k.sum(axis=0)                 # (d,) normalizer accumulator

    numerator = phi_q @ M                 # (d,)
    denominator = phi_q @ z + eps
    return numerator / denominator


def make_sparse_projection(d, D, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=1.0 / np.sqrt(d), size=(d, D))


def _sparse_code(x, W, k_active):
    """x: (..., d), W: (d, D) -> sparse non-negative code (..., D) with
    exactly k_active nonzero entries per row (ReLU + top-k)."""
    proj = np.maximum(x @ W, 0.0)  # ReLU, (..., D)
    if proj.ndim == 1:
        proj = proj[None, :]
        squeeze = True
    else:
        squeeze = False
    D = proj.shape[-1]
    k_active = min(k_active, D)
    idx = np.argpartition(proj, -k_active, axis=-1)[..., -k_active:]
    mask = np.zeros_like(proj)
    np.put_along_axis(mask, idx, 1.0, axis=-1)
    out = proj * mask
    return out[0] if squeeze else out


def sparse_linear_attention(keys, values, neighbor_mask, cue, W, k_active, eps=1e-6):
    """
    Same fast-weight mechanism as linear_attention, but keys/cue are first
    mapped through a sparse non-negative random-projection code.
    """
    phi_k = _sparse_code(keys, W, k_active)          # (L, D)
    phi_q = _sparse_code(cue, W, k_active)            # (D,)
    phi_k = phi_k * neighbor_mask[:, None]

    M = phi_k.T @ values                              # (D, d)
    z = phi_k.sum(axis=0)                             # (D,)

    numerator = phi_q @ M                             # (d,)
    denominator = phi_q @ z + eps
    return numerator / denominator
