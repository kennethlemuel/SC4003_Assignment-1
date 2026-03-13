from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from algorithms import AlgorithmResult
from mdp import Action, GridWorldMDP, State

ARROWS = {
    "up": "^",
    "right": ">",
    "down": "v",
    "left": "<",
}


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_utilities_grid(mdp: GridWorldMDP, utilities: np.ndarray) -> str:
    rows: List[str] = []
    for row in range(mdp.rows):
        tokens = []
        for col in range(mdp.cols):
            if mdp.grid[row][col] == "W":
                tokens.append("  WALL  ")
                continue
            value = utilities[mdp.state_to_index[(row, col)]]
            tokens.append(f"{value:8.3f}")
        rows.append(" ".join(tokens))
    return "\n".join(rows)


def format_policy_grid(mdp: GridWorldMDP, policy: Mapping[State, Action]) -> str:
    rows: List[str] = []
    for row in range(mdp.rows):
        tokens = []
        for col in range(mdp.cols):
            if mdp.grid[row][col] == "W":
                tokens.append("WALL")
            else:
                tokens.append(f"  {ARROWS[policy[(row, col)]]} ")
        rows.append(" ".join(tokens))
    return "\n".join(rows)


def policies_match(
    policy_a: Mapping[State, Action], policy_b: Mapping[State, Action]
) -> bool:
    return all(policy_a[state] == policy_b[state] for state in policy_a)


def differing_policy_states(
    mdp: GridWorldMDP,
    policy_a: Mapping[State, Action],
    policy_b: Mapping[State, Action],
) -> List[str]:
    differences = []
    for state in mdp.states:
        if policy_a[state] != policy_b[state]:
            differences.append(
                f"{mdp.state_label(state)}: {policy_a[state]} vs {policy_b[state]}"
            )
    return differences


def run_validation_checks(
    mdp: GridWorldMDP,
    value_result: AlgorithmResult,
    policy_result: AlgorithmResult,
    utility_tolerance: float = 1e-5,
) -> List[str]:
    messages: List[str] = []

    transition_messages = mdp.validate_transition_probabilities()
    if transition_messages:
        messages.extend(transition_messages)
    else:
        messages.append("All transition probabilities sum to 1.0.")

    wall_count = sum(cell == "W" for row in mdp.grid for cell in row)
    if mdp.num_states + wall_count == mdp.rows * mdp.cols:
        messages.append("Wall cells are excluded from the state space correctly.")
    else:
        messages.append("Warning: State space size is inconsistent with wall count.")

    if policies_match(value_result.policy, policy_result.policy):
        messages.append("Value iteration and policy iteration produced the same policy.")
    else:
        differences = differing_policy_states(mdp, value_result.policy, policy_result.policy)
        messages.append(
            "Warning: Policies differ. Likely causes are tie states or insufficient "
            f"convergence. Differing states: {', '.join(differences[:10])}"
        )

    max_difference = np.max(np.abs(value_result.utilities - policy_result.utilities))
    if max_difference <= utility_tolerance:
        messages.append(
            f"Final utility vectors are numerically close (max abs diff = {max_difference:.6e})."
        )
    else:
        messages.append(
            "Warning: Final utilities are not sufficiently close "
            f"(max abs diff = {max_difference:.6e})."
        )

    return messages


def export_utility_history_csv(
    path: Path,
    mdp: GridWorldMDP,
    history: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        header = ["iteration"] + [mdp.state_label(state) for state in mdp.states]
        writer.writerow(header)
        for iteration_index, utilities in enumerate(history):
            writer.writerow([iteration_index, *utilities.tolist()])


def export_final_utilities_csv(
    path: Path,
    mdp: GridWorldMDP,
    utilities: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["row", "col", "state", "cell_type", "reward", "utility"])
        for state in mdp.states:
            row, col = state
            writer.writerow(
                [
                    row + 1,
                    col + 1,
                    mdp.state_label(state),
                    mdp.state_types[state],
                    mdp.reward_by_state[state],
                    utilities[mdp.state_to_index[state]],
                ]
            )


def export_policy_csv(
    path: Path,
    mdp: GridWorldMDP,
    policy: Mapping[State, Action],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["row", "col", "state", "action", "arrow"])
        for state in mdp.states:
            row, col = state
            action = policy[state]
            writer.writerow([row + 1, col + 1, mdp.state_label(state), action, ARROWS[action]])


def export_summary_metrics_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def representative_states_for_plot(
    mdp: GridWorldMDP, max_states: int = 6
) -> List[State]:
    return mdp.representative_states[:max_states]
