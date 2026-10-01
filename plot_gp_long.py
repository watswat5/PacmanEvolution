import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from deap import gp

from GeneticAlgo.gp_core import ACTIONS, load_individual


SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("gp_long.csv")
OUTPUT_DIRECTORY = Path(sys.argv[2]) if len(sys.argv) > 2 else SOURCE.parent
BEST_INDIVIDUAL = Path(sys.argv[3]) if len(sys.argv) > 3 else None
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
OUTPUT = OUTPUT_DIRECTORY / "training_metrics.png"


def read_numeric_csv(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return {name: [float(row[name]) for row in rows] for name in rows[0]}


data = read_numeric_csv(SOURCE)
generation = data["generation"]

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True, constrained_layout=True)
fig.suptitle("Genetic Programming Training Metrics", fontsize=18, fontweight="bold")

fitness_colors = {"Best": "#1f77b4", "Average": "#ff7f0e", "Worst": "#7f7f7f"}
for label, column in (("Best", "best_fitness"), ("Average", "average_fitness"), ("Worst", "worst_fitness")):
    axes[0].plot(generation, data[column], label=label, color=fitness_colors[label], linewidth=2)
axes[0].axhline(0, color="#333333", linewidth=0.8, alpha=0.6)
axes[0].set_title("Fitness")
axes[0].set_ylabel("Fitness score")
axes[0].legend(ncol=3, frameon=True)

axes[1].plot(generation, data["average_tree_depth"], label="Average tree", color="#2ca02c", linewidth=2)
axes[1].plot(generation, data["best_tree_depth"], label="Best tree", color="#9467bd", linewidth=1.8, alpha=0.9)
axes[1].set_title("Tree Depth")
axes[1].set_ylabel("Depth")
axes[1].legend(ncol=2, frameon=True)

axes[2].plot(generation, data["average_node_count"], label="Average tree", color="#17becf", linewidth=2)
axes[2].plot(generation, data["best_node_count"], label="Best tree", color="#d62728", linewidth=1.6, alpha=0.85)
axes[2].set_title("Node Count")
axes[2].set_xlabel("Generation")
axes[2].set_ylabel("Nodes")
axes[2].legend(ncol=2, frameon=True)

for axis in axes:
    axis.margins(x=0)
    axis.grid(True, alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
print(OUTPUT.resolve())
plt.close(fig)

fig, axes = plt.subplots(
    2, 1, figsize=(13, 9), sharex=True, constrained_layout=True
)
fig.suptitle("Action-Tree Complexity", fontsize=18, fontweight="bold")
colors = {
    "up": "#1f77b4",
    "down": "#ff7f0e",
    "left": "#2ca02c",
    "right": "#d62728",
}
for action, color in colors.items():
    axes[0].plot(
        generation, data[f"average_{action}_tree_depth"],
        label=action.title(), color=color, linewidth=2,
    )
    axes[1].plot(
        generation, data[f"average_{action}_node_count"],
        label=action.title(), color=color, linewidth=2,
    )
axes[0].set(title="Population-Average Depth", ylabel="Depth")
axes[1].set(title="Population-Average Node Count", xlabel="Generation", ylabel="Nodes")
for axis in axes:
    axis.legend(ncol=4, frameon=True)
    axis.margins(x=0)
    axis.spines[["top", "right"]].set_visible(False)
action_output = OUTPUT_DIRECTORY / "action_tree_complexity.png"
fig.savefig(action_output, dpi=200, bbox_inches="tight", facecolor="white")
print(action_output.resolve())


def tree_layout(tree):
    """Return deterministic hierarchical positions for a DEAP prefix tree."""
    nodes, edges, labels = gp.graph(tree)
    children = {node: [] for node in nodes}
    for parent, child in edges:
        children[parent].append(child)
    positions = {}
    next_leaf = [0]

    def place(node, depth):
        if not children[node]:
            x_position = next_leaf[0]
            next_leaf[0] += 1
        else:
            child_positions = [place(child, depth + 1) for child in children[node]]
            x_position = sum(child_positions) / len(child_positions)
        positions[node] = (x_position, -depth)
        return x_position

    place(0, 0)
    return nodes, edges, labels, positions


def render_best_action_trees(path, individual):
    """Render all four action trees as one readable 2x2 report."""
    figure, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    figure.suptitle(
        "Best Overall Individual — Action Trees",
        fontsize=20,
        fontweight="bold",
    )
    action_colors = {
        "up": "#1f77b4",
        "down": "#ff7f0e",
        "left": "#2ca02c",
        "right": "#d62728",
    }
    for axis, action, tree in zip(axes.flat, ACTIONS, individual):
        nodes, edges, labels, positions = tree_layout(tree)
        color = action_colors[action]
        for parent, child in edges:
            x_values = (positions[parent][0], positions[child][0])
            y_values = (positions[parent][1], positions[child][1])
            axis.plot(x_values, y_values, color="#7f8c8d", linewidth=1.2, zorder=1)
        x_values = [positions[node][0] for node in nodes]
        y_values = [positions[node][1] for node in nodes]
        axis.scatter(
            x_values,
            y_values,
            s=520,
            color=color,
            edgecolor="white",
            linewidth=1.4,
            zorder=2,
        )
        font_size = max(6, 10 - len(nodes) // 20)
        for node in nodes:
            axis.annotate(
                str(labels[node]),
                positions[node],
                ha="center",
                va="center",
                fontsize=font_size,
                color="white",
                fontweight="bold",
                zorder=3,
            )
        axis.set_title(
            f"{action.title()} — {len(tree)} nodes, depth {tree.height + 1}",
            fontsize=14,
            fontweight="bold",
        )
        axis.margins(x=0.08, y=0.15)
        axis.set_axis_off()
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if BEST_INDIVIDUAL is not None:
    tree_output = OUTPUT_DIRECTORY / "best_action_trees.png"
    render_best_action_trees(tree_output, load_individual(BEST_INDIVIDUAL))
    print(tree_output.resolve())
