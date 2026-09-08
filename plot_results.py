import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results.json") as f:
    results = json.load(f)
with open("costs.json") as f:
    costs = json.load(f)

loads = results["loads"]
n_seeds = results.get("num_seeds", 1)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

colors = {"dense": "#1f77b4", "linear": "#d62728", "sparse": "#2ca02c"}
labels = {
    "dense": "Dense softmax attention",
    "linear": "Linear attention (dense fast-weight matrix)",
    "sparse": "Sparse+non-neg linear attention (~5% active, BDH-style)",
}

# --- Panel 1: retrieval accuracy, with error bars across seeds ---
ax = axes[0]
for name in ["dense", "linear", "sparse"]:
    means = [r["accuracy_mean"] for r in results[name]]
    stds = [r["accuracy_std"] for r in results[name]]
    ax.errorbar(loads, means, yerr=stds, marker="o", capsize=3, label=labels[name], color=colors[name])
ax.set_xscale("log", base=2)
ax.set_xlabel("Memory load L (number of stored write-node items)")
ax.set_ylabel("Retrieval accuracy")
ax.set_title(f"Does the right item come back?\n(mean +/- std over {n_seeds} random seeds)")
ax.set_ylim(-0.05, 1.08)
ax.legend(fontsize=7.5, loc="lower left")
ax.grid(alpha=0.3)

# --- Panel 2: fidelity, with error bars -- the early-warning signal ---
ax = axes[1]
for name in ["dense", "linear", "sparse"]:
    means = [r["fidelity_mean"] for r in results[name]]
    stds = [r["fidelity_std"] for r in results[name]]
    ax.errorbar(loads, means, yerr=stds, marker="o", capsize=3, label=labels[name], color=colors[name])
ax.set_xscale("log", base=2)
ax.set_xlabel("Memory load L")
ax.set_ylabel("Fidelity (cosine similarity to true value)")
ax.set_title("How CONFIDENTLY correct is the retrieval?\n(drops long before accuracy does, for linear)")
ax.set_ylim(-0.05, 1.08)
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(alpha=0.3)

# --- Panel 3: theoretical per-query compute cost (unchanged) ---
ax = axes[2]
ax.plot(costs["loads"], costs["dense_cost"], marker="o", color=colors["dense"],
         label="Dense: rescans all L items per query\n(cost ~ L x num_queries)")
ax.plot(costs["loads"], costs["linear_cost"], marker="o", color="#7f7f7f",
         label="Linear / sparse: build memory once,\nO(1) per query after (cost ~ L + num_queries)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Memory load L")
ax.set_ylabel("Total work for 300 queries (arbitrary units)")
ax.set_title("The efficiency argument for going linear")
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("accuracy_and_cost.png", dpi=150)
print("Saved accuracy_and_cost.png (3 panels: accuracy, fidelity, cost)")
