from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from algorithms import AlgorithmResult
from mdp import Action, GridWorldMDP, State
from utils import ARROWS

CELL_FACE_COLORS = {
    "E": "#f7f7f7",
    "S": "#d9ecff",
    "G": "#cdeccf",
    "B": "#f3d1c8",
    "W": "#333333",
}


def _base_grid_figure(mdp: GridWorldMDP, title: str, figsize_scale: float = 1.0):
    fig_width = max(6.0, mdp.cols * 1.15 * figsize_scale)
    fig_height = max(5.0, mdp.rows * 1.05 * figsize_scale)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, mdp.cols)
    ax.set_ylim(mdp.rows, 0)
    ax.set_xticks(np.arange(0, mdp.cols + 1, 1))
    ax.set_yticks(np.arange(0, mdp.rows + 1, 1))
    ax.grid(color="#888888", linewidth=0.8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=14, pad=14)
    return fig, ax


def save_policy_grid(
    mdp: GridWorldMDP,
    policy: Mapping[State, Action],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = _base_grid_figure(mdp, title)

    for row in range(mdp.rows):
        for col in range(mdp.cols):
            cell_type = mdp.grid[row][col]
            rect = plt.Rectangle(
                (col, row),
                1,
                1,
                facecolor=CELL_FACE_COLORS[cell_type],
                edgecolor="none",
            )
            ax.add_patch(rect)
            if cell_type == "W":
                ax.text(
                    col + 0.5,
                    row + 0.5,
                    "W",
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="white",
                    weight="bold",
                )
            else:
                state = (row, col)
                arrow = ARROWS[policy[state]]
                ax.text(
                    col + 0.5,
                    row + 0.54,
                    arrow,
                    ha="center",
                    va="center",
                    fontsize=24,
                    color="#1d3557",
                    weight="bold",
                )
                ax.text(
                    col + 0.12,
                    row + 0.18,
                    cell_type,
                    ha="left",
                    va="top",
                    fontsize=9,
                    color="#333333",
                    weight="bold",
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_utility_heatmap(
    mdp: GridWorldMDP,
    utilities: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    utility_grid = mdp.utility_grid(utilities)
    masked = np.ma.masked_invalid(utility_grid)

    fig, ax = plt.subplots(figsize=(max(6.0, mdp.cols * 1.15), max(5.0, mdp.rows * 1.05)))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color=CELL_FACE_COLORS["W"])
    image = ax.imshow(masked, cmap=cmap, origin="upper")
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticks(np.arange(mdp.cols))
    ax.set_yticks(np.arange(mdp.rows))
    ax.set_xticklabels([str(col + 1) for col in range(mdp.cols)])
    ax.set_yticklabels([str(row + 1) for row in range(mdp.rows)])
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(np.arange(-0.5, mdp.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mdp.rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(mdp.rows):
        for col in range(mdp.cols):
            cell_type = mdp.grid[row][col]
            if cell_type == "W":
                ax.text(col, row, "W", ha="center", va="center", color="white", weight="bold")
                continue
            state = (row, col)
            utility = utilities[mdp.state_to_index[state]]
            label = f"{utility:.2f}\n{cell_type}"
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                weight="bold" if cell_type in {"G", "B", "S"} else None,
            )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("Utility")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_convergence_plot_all_states(
    mdp: GridWorldMDP,
    history: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = np.arange(history.shape[0])
    for state_index, state in enumerate(mdp.states):
        ax.plot(
            iterations,
            history[:, state_index],
            linewidth=1.0,
            alpha=0.55,
        )
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Utility estimate")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_convergence_plot_subset(
    mdp: GridWorldMDP,
    history: np.ndarray,
    selected_states: Sequence[State],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = np.arange(history.shape[0])
    for state in selected_states:
        state_index = mdp.state_to_index[state]
        ax.plot(
            iterations,
            history[:, state_index],
            linewidth=2.0,
            label=mdp.state_label(state),
        )
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Utility estimate")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_comparative_convergence(
    mdp: GridWorldMDP,
    value_result: AlgorithmResult,
    policy_result: AlgorithmResult,
    selected_states: Sequence[State],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    vi_iterations = np.arange(value_result.utility_history.shape[0])
    for state in selected_states:
        state_index = mdp.state_to_index[state]
        axes[0].plot(
            vi_iterations,
            value_result.utility_history[:, state_index],
            linewidth=2.0,
            label=mdp.state_label(state),
        )
    axes[0].set_title(
        f"Value Iteration\niterations={value_result.iteration_count}, "
        f"runtime={value_result.runtime_seconds:.4f}s"
    )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Utility estimate")
    axes[0].grid(alpha=0.3)

    pi_iterations = np.arange(policy_result.utility_history.shape[0])
    for state in selected_states:
        state_index = mdp.state_to_index[state]
        axes[1].plot(
            pi_iterations,
            policy_result.utility_history[:, state_index],
            linewidth=2.0,
            label=mdp.state_label(state),
        )
    axes[1].set_title(
        f"Policy Iteration\niterations={policy_result.iteration_count}, "
        f"runtime={policy_result.runtime_seconds:.4f}s"
    )
    axes[1].set_xlabel("Iteration")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
