"""
Graph-structured associative retrieval task.

Setup
-----
A bipartite graph with two node types:
  - WRITE nodes: each holds a (key, value) pair written into the graph's memory.
  - QUERY nodes: each is connected by edges to a subset of write nodes (its
    graph neighborhood) and must retrieve the value of the ONE write node
    among its neighbors whose key matches its cue.

This is deliberately graph-structured, not a disguised sequence: a query's
neighborhood is an explicit edge list into the write-node set, and the
"load" (memory difficulty) is controlled by how many write nodes sit in
that neighborhood.

Everything here is analytic / deterministic-given-seed. No training loop
is needed for the core experiment: we are measuring capacity/interference
properties of three different graph message-passing (aggregation) rules,
not learning a task.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class GraphInstance:
    keys: np.ndarray        # (L, d) unit-norm key vectors, one per write node
    values: np.ndarray      # (L, d) unit-norm value vectors, one per write node
    neighbor_idx: np.ndarray  # (num_query, L) boolean mask: which write nodes each query is connected to
    cues: np.ndarray        # (num_query, d) query cue vectors
    targets: np.ndarray     # (num_query,) int, index into [0, L) of the true target write node


def _unit_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def build_graph_instance(
    num_write_nodes: int,
    d: int,
    num_queries: int = 200,
    connection_prob: float = 1.0,
    cue_noise_sigma: float = 0.05,
    seed: int = 0,
) -> GraphInstance:
    """
    Build one random graph instance.

    num_write_nodes (L): total items written into the graph -- this is the
        "memory load" variable we sweep in the experiment.
    d: embedding dimension of keys/values (fixed across the sweep).
    connection_prob: probability that a given write node is an edge-neighbor
        of a given query node (the true target is always forced to be a
        neighbor). 1.0 reproduces "each query sees all write nodes"; lower
        values make the graph an irregular, partially-connected graph.
    """
    rng = np.random.default_rng(seed)
    L = num_write_nodes

    keys = _unit_rows(rng.normal(size=(L, d)))
    values = _unit_rows(rng.normal(size=(L, d)))

    targets = rng.integers(low=0, high=L, size=num_queries)

    # Build the bipartite edge structure: neighbor_idx[q, j] = True if
    # write node j is connected to query q.
    neighbor_idx = rng.random((num_queries, L)) < connection_prob
    neighbor_idx[np.arange(num_queries), targets] = True  # target is always a neighbor

    # Cue = the true target's key plus small noise, re-normalized to unit length.
    noise = rng.normal(scale=cue_noise_sigma, size=(num_queries, d))
    cues = _unit_rows(keys[targets] + noise)

    return GraphInstance(keys=keys, values=values, neighbor_idx=neighbor_idx, cues=cues, targets=targets)


if __name__ == "__main__":
    g = build_graph_instance(num_write_nodes=8, d=4, num_queries=3, seed=1)
    print("keys shape:", g.keys.shape)
    print("values shape:", g.values.shape)
    print("neighbor_idx (query x write):\n", g.neighbor_idx.astype(int))
    print("targets:", g.targets)
