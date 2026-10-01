#!/usr/bin/env python3
import csv, pickle, sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from deap import gp
from GeneticAlgo import train_board_graph  # registers DEAP pickle types
from GeneticAlgo.train_board_graph import FEATURE_NAMES

source, output, best = map(Path, sys.argv[1:4])
with source.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise ValueError(f"{source} has no data rows")
data = {key: [float(row[key]) for row in rows] for key in rows[0]}
x = data["generation"]
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True, constrained_layout=True)
for label, key in (("Best", "best_fitness"), ("Average", "average_fitness"), ("Worst", "worst_fitness")):
    axes[0].plot(x, data[key], label=label)
if "validation_fitness" in data:
    axes[0].plot(
        x, data["validation_fitness"], label="Champion validation",
        color="#111827", linewidth=2.5, linestyle="--",
    )
if "best_validation_fitness_so_far" in data:
    axes[0].plot(
        x, data["best_validation_fitness_so_far"],
        label="Best validation so far", color="#16a34a", linewidth=2.5,
    )
axes[0].set(title="Fitness", ylabel="Fitness score"); axes[0].legend()
for axis, avg, champion, title in ((axes[1], "average_tree_depth", "best_tree_depth", "Tree Depth"), (axes[2], "average_node_count", "best_node_count", "Node Count")):
    axis.plot(x, data[avg], label="Average"); axis.plot(x, data[champion], label="Best")
    axis.set_title(title); axis.legend()
axes[2].set_xlabel("Generation")
metrics = output / "training_metrics.png"
fig.savefig(metrics, dpi=200, bbox_inches="tight"); plt.close(fig)
print(metrics.resolve())

with best.open("rb") as handle:
    tree = pickle.load(handle)
nodes, edges, labels = gp.graph(tree)
children = {node: [] for node in nodes}
for parent, child in edges: children[parent].append(child)
positions, leaf = {}, [0]
def place(node, depth):
    if children[node]:
        xs = [place(child, depth + 1) for child in children[node]]
        x = sum(xs) / len(xs)
    else:
        x = leaf[0]; leaf[0] += 1
    positions[node] = (x, -depth)
    return x
place(0, 0)
terminal_colors = dict(zip(FEATURE_NAMES, plt.get_cmap("tab20").colors))
terminal_codes = {
    terminal: f"T{index + 1}"
    for index, terminal in enumerate(FEATURE_NAMES)
}
leaf_count = max(1, leaf[0])
fig_width = max(16.0, min(36.0, leaf_count * 1.8))
fig_height = max(10.0, (tree.height + 1) * 1.45 + 4.0)
fig, axis = plt.subplots(figsize=(fig_width, fig_height))
for parent, child in edges:
    axis.plot(
        (positions[parent][0], positions[child][0]),
        (positions[parent][1], positions[child][1]),
        color="#718096",
        linewidth=1.8,
        zorder=1,
    )
for node in nodes:
    label = str(labels[node])
    if label in terminal_colors:
        display_label = terminal_codes[label]
        facecolor = terminal_colors[label]
        text_color = "white"
    else:
        display_label = label
        is_constant = not children[node]
        facecolor = "#64748b" if is_constant else "#1d4ed8"
        text_color = "white"
    axis.annotate(
        display_label,
        positions[node],
        ha="center",
        va="center",
        color=text_color,
        fontsize=11,
        fontweight="bold",
        zorder=3,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": facecolor,
            "edgecolor": "#172033",
            "linewidth": 1.2,
        },
    )
legend_items = [
    Patch(facecolor="#1d4ed8", edgecolor="#172033", label="Function node"),
    Patch(facecolor="#64748b", edgecolor="#172033", label="Numeric constant"),
] + [
    Patch(
        facecolor=terminal_colors[terminal],
        edgecolor="#172033",
        label=f"{terminal_codes[terminal]}  {terminal}",
    )
    for terminal in FEATURE_NAMES
]
fig.legend(
    handles=legend_items,
    title="Node color map",
    loc="lower center",
    bbox_to_anchor=(0.5, 0.015),
    ncol=3,
    frameon=True,
    fontsize=10,
    title_fontsize=11,
)
axis.set_title(
    f"Best Shared Move-Scoring Tree — {len(tree)} nodes, "
    f"depth {tree.height + 1}",
    fontsize=16,
    pad=24,
)
axis.margins(x=0.08, y=0.12)
axis.set_axis_off()
fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.24)
tree_path = output / "best_tree.png"
fig.savefig(tree_path, dpi=180, bbox_inches="tight"); plt.close(fig)
print(tree_path.resolve())
