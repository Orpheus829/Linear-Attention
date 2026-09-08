"""
Core experiment: does the predicted pattern actually hold?

  - dense_attention:  stays near-perfect regardless of load (recomputes
                       from raw keys/values every time -- no compression).
  - linear_attention: degrades as load grows (dense low-dim code packed
                       into a fixed-size matrix -> crosstalk).
  - sparse_linear_attention: degrades much less than linear_attention at
                       the same load (sparse high-dim code -> less crosstalk),
                       while keeping the same O(1)-per-query mechanism.

For each load L, we sample many random graph instances/queries and report
mean retrieval accuracy (does the output vector's nearest stored value,
by cosine similarity, equal the true target?) and mean cosine fidelity to
the true value.
"""
import numpy as np
import json
from task import build_graph_instance
from aggregators import dense_attention, linear_attention, sparse_linear_attention, make_sparse_projection

D_EMBED = 64          # embedding dimension d, held fixed across the sweep
D_SPARSE = 1024       # sparse code dimension D >> d
SPARSITY_FRAC = 0.05  # ~5%, matching BDH's reported active-unit fraction
K_ACTIVE = max(1, int(D_SPARSE * SPARSITY_FRAC))
LOADS = [2, 4, 8, 16, 32, 64, 128, 256]
NUM_QUERIES_PER_LOAD = 300
SEED = 42


NUM_SEEDS_PER_LOAD = 5  # independent random graph instances per load, for error bars


def retrieval_accuracy_and_fidelity(out_vectors, values, targets):
    # out_vectors: (Q, d), values: (L, d) [shared across queries in this instance], targets: (Q,)
    sims = out_vectors @ values.T  # (Q, L) cosine-ish sim (values are unit-norm)
    preds = np.argmax(sims, axis=1)
    acc = float(np.mean(preds == targets))
    fidelity = float(np.mean([out_vectors[i] @ values[targets[i]] for i in range(len(targets))]))
    return acc, fidelity


def run_sweep():
    W = make_sparse_projection(D_EMBED, D_SPARSE, seed=SEED + 1)
    results = {"loads": LOADS, "num_seeds": NUM_SEEDS_PER_LOAD, "dense": [], "linear": [], "sparse": []}

    for L in LOADS:
        per_method_acc = {"dense": [], "linear": [], "sparse": []}
        per_method_fid = {"dense": [], "linear": [], "sparse": []}

        for seed_i in range(NUM_SEEDS_PER_LOAD):
            g = build_graph_instance(
                num_write_nodes=L,
                d=D_EMBED,
                num_queries=NUM_QUERIES_PER_LOAD,
                connection_prob=1.0,
                seed=SEED + L * 1000 + seed_i,  # distinct graph instance per seed
            )

            out_dense = np.zeros((NUM_QUERIES_PER_LOAD, D_EMBED))
            out_linear = np.zeros((NUM_QUERIES_PER_LOAD, D_EMBED))
            out_sparse = np.zeros((NUM_QUERIES_PER_LOAD, D_EMBED))

            for q in range(NUM_QUERIES_PER_LOAD):
                mask = g.neighbor_idx[q]
                cue = g.cues[q]
                out_dense[q] = dense_attention(g.keys, g.values, mask, cue)
                out_linear[q] = linear_attention(g.keys, g.values, mask, cue)
                out_sparse[q] = sparse_linear_attention(g.keys, g.values, mask, cue, W, K_ACTIVE)

            for name, out in [("dense", out_dense), ("linear", out_linear), ("sparse", out_sparse)]:
                acc, fid = retrieval_accuracy_and_fidelity(out, g.values, g.targets)
                per_method_acc[name].append(acc)
                per_method_fid[name].append(fid)

        for name in ["dense", "linear", "sparse"]:
            acc_mean = float(np.mean(per_method_acc[name]))
            acc_std = float(np.std(per_method_acc[name]))
            fid_mean = float(np.mean(per_method_fid[name]))
            fid_std = float(np.std(per_method_fid[name]))
            results[name].append({
                "load": L,
                "accuracy_mean": acc_mean, "accuracy_std": acc_std,
                "fidelity_mean": fid_mean, "fidelity_std": fid_std,
                "accuracy_per_seed": per_method_acc[name],
                "fidelity_per_seed": per_method_fid[name],
            })
            print(f"L={L:4d}  {name:8s}  accuracy={acc_mean:.3f}+/-{acc_std:.3f}  "
                  f"fidelity={fid_mean:.3f}+/-{fid_std:.3f}  (n={NUM_SEEDS_PER_LOAD} seeds)")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def theoretical_cost(loads, num_queries=NUM_QUERIES_PER_LOAD):
    """
    Per the incremental-update argument: dense attention must re-scan all L
    raw (key,value) pairs for EVERY query -> total work ~ L * Q.
    Linear/sparse attention build a fixed-size memory once (~L) then answer
    each query in O(1) -> total work ~ L + Q.
    Returns a dict of load -> (dense_cost, linear_cost) for plotting.
    """
    out = {"loads": loads, "dense_cost": [], "linear_cost": []}
    for L in loads:
        out["dense_cost"].append(L * num_queries)
        out["linear_cost"].append(L + num_queries)
    return out


if __name__ == "__main__":
    results = run_sweep()
    costs = theoretical_cost(LOADS)
    with open("costs.json", "w") as f:
        json.dump(costs, f, indent=2)
    print("\nSaved results.json and costs.json")
