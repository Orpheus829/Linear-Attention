"""
The central claim rests on: linear attention's capacity is tied to the
embedding dimension d (it compresses everything into a fixed d x d matrix).
That predicts something specific and checkable: if we plot accuracy against
the RATIO load/d instead of raw load, curves for different values of d
should collapse onto roughly the same curve -- because what matters isn't
the absolute number of stored items, it's how many are packed in relative
to the matrix's capacity.

This is deliberately run for dense and linear only. Sparse's effective
capacity is governed by D_SPARSE (the sparse code's dimension), not d --
mixing it into this specific check (which isolates the d-dependence of the
DENSE fast-weight matrix) would muddy what's being tested. Sparse's own
capacity behavior is covered by the sparsity-level sweep in the notebook.
"""
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from task import build_graph_instance
from aggregators import dense_attention, linear_attention

NUM_QUERIES = 200
SEEDS_PER_POINT = 3
RATIOS = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]  # load / d
D_VALUES = [32, 64, 128]


def accuracy_at(d, L, seed):
    g = build_graph_instance(num_write_nodes=L, d=d, num_queries=NUM_QUERIES, connection_prob=1.0, seed=seed)
    out_dense = np.array([dense_attention(g.keys, g.values, g.neighbor_idx[q], g.cues[q]) for q in range(NUM_QUERIES)])
    out_linear = np.array([linear_attention(g.keys, g.values, g.neighbor_idx[q], g.cues[q]) for q in range(NUM_QUERIES)])
    acc_dense = float((out_dense @ g.values.T).argmax(axis=1).__eq__(g.targets).mean())
    acc_linear = float((out_linear @ g.values.T).argmax(axis=1).__eq__(g.targets).mean())
    return acc_dense, acc_linear


def run():
    results = {"ratios": RATIOS, "d_values": D_VALUES, "dense": {}, "linear": {}}
    for d in D_VALUES:
        dense_means, dense_stds, linear_means, linear_stds = [], [], [], []
        for ratio in RATIOS:
            L = max(2, int(round(ratio * d)))
            accs_d, accs_l = [], []
            for s in range(SEEDS_PER_POINT):
                seed = 7000 + d * 100 + int(ratio * 1000) + s
                ad, al = accuracy_at(d, L, seed)
                accs_d.append(ad)
                accs_l.append(al)
            dense_means.append(float(np.mean(accs_d))); dense_stds.append(float(np.std(accs_d)))
            linear_means.append(float(np.mean(accs_l))); linear_stds.append(float(np.std(accs_l)))
            print(f"d={d:4d}  L/d={ratio:6.3f}  L={L:4d}  "
                  f"dense={np.mean(accs_d):.3f}  linear={np.mean(accs_l):.3f}")
        results["dense"][d] = {"mean": dense_means, "std": dense_stds}
        results["linear"][d] = {"mean": linear_means, "std": linear_stds}

    with open("dimension_scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {32: "#9467bd", 64: "#d62728", 128: "#8c564b"}
    for d in D_VALUES:
        ax.errorbar(RATIOS, results["linear"][d]["mean"], yerr=results["linear"][d]["std"],
                     marker="o", capsize=3, label=f"linear attention, d={d}", color=colors[d])
    # Dense reference: should stay ~1.0 regardless of d or ratio -- plot once, dashed, gray.
    dense_all = [results["dense"][d]["mean"] for d in D_VALUES]
    dense_avg = np.mean(dense_all, axis=0)
    ax.plot(RATIOS, dense_avg, linestyle="--", color="gray", marker="s",
            label="dense attention (any d -- reference)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Memory load / embedding dimension  (L / d)")
    ax.set_ylabel("Retrieval accuracy")
    ax.set_title("Does linear attention's collapse point scale with d?\n(curves should roughly line up if capacity ~ d)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("dimension_scaling.png", dpi=150)
    print("Saved dimension_scaling.png")


if __name__ == "__main__":
    results = run()
    plot(results)
