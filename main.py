from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from algorithms import policy_iteration, value_iteration
from environments import create_part1_environment, create_part2_environment
from mdp import GridWorldMDP
from utils import (
    ensure_directory,
    export_final_utilities_csv,
    export_policy_csv,
    export_summary_metrics_csv,
    export_utility_history_csv,
    format_policy_grid,
    format_utilities_grid,
    representative_states_for_plot,
    run_validation_checks,
)
from visualization import (
    save_comparative_convergence,
    save_convergence_plot_all_states,
    save_convergence_plot_subset,
    save_policy_grid,
    save_utility_heatmap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run value iteration and policy iteration on assignment gridworlds."
    )
    parser.add_argument(
        "--part",
        choices=("1", "2", "both"),
        default="both",
        help="Run Part 1 only, Part 2 only, or both environments.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Value iteration tolerance parameter.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20_000,
        help="Maximum iterations for value iteration and policy iteration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for figures and CSV exports.",
    )
    return parser.parse_args()


def selected_environments(part_flag: str) -> List[GridWorldMDP]:
    if part_flag == "1":
        return [create_part1_environment()]
    if part_flag == "2":
        return [create_part2_environment()]
    return [create_part1_environment(), create_part2_environment()]


def print_environment_header(mdp: GridWorldMDP) -> None:
    print("=" * 90)
    print(f"Environment: {mdp.name}")
    print(f"Grid size: {mdp.rows} x {mdp.cols}")
    print(f"Number of non-wall states: {mdp.num_states}")
    print(f"Discount factor gamma: {mdp.gamma}")
    print("=" * 90)


def export_environment_results(
    mdp: GridWorldMDP,
    environment_output_dir: Path,
    value_result,
    policy_result,
) -> None:
    selected_states = representative_states_for_plot(mdp)

    export_utility_history_csv(
        environment_output_dir / f"{mdp.name}_value_iteration_history.csv",
        mdp,
        value_result.utility_history,
    )
    export_utility_history_csv(
        environment_output_dir / f"{mdp.name}_policy_iteration_history.csv",
        mdp,
        policy_result.utility_history,
    )
    export_final_utilities_csv(
        environment_output_dir / f"{mdp.name}_value_iteration_utilities.csv",
        mdp,
        value_result.utilities,
    )
    export_final_utilities_csv(
        environment_output_dir / f"{mdp.name}_policy_iteration_utilities.csv",
        mdp,
        policy_result.utilities,
    )
    export_policy_csv(
        environment_output_dir / f"{mdp.name}_value_iteration_policy.csv",
        mdp,
        value_result.policy,
    )
    export_policy_csv(
        environment_output_dir / f"{mdp.name}_policy_iteration_policy.csv",
        mdp,
        policy_result.policy,
    )

    save_policy_grid(
        mdp,
        value_result.policy,
        environment_output_dir / f"{mdp.name}_value_iteration_policy.png",
        title=f"{mdp.name}: optimal policy from value iteration",
    )
    save_policy_grid(
        mdp,
        policy_result.policy,
        environment_output_dir / f"{mdp.name}_policy_iteration_policy.png",
        title=f"{mdp.name}: optimal policy from policy iteration",
    )
    save_utility_heatmap(
        mdp,
        value_result.utilities,
        environment_output_dir / f"{mdp.name}_value_iteration_utilities.png",
        title=f"{mdp.name}: utilities from value iteration",
    )
    save_utility_heatmap(
        mdp,
        policy_result.utilities,
        environment_output_dir / f"{mdp.name}_policy_iteration_utilities.png",
        title=f"{mdp.name}: utilities from policy iteration",
    )
    save_convergence_plot_all_states(
        mdp,
        value_result.utility_history,
        environment_output_dir / f"{mdp.name}_value_iteration_convergence_all_states.png",
        title=f"{mdp.name}: value iteration convergence for all non-wall states",
    )
    save_convergence_plot_subset(
        mdp,
        value_result.utility_history,
        selected_states,
        environment_output_dir / f"{mdp.name}_value_iteration_convergence_selected_states.png",
        title=f"{mdp.name}: value iteration convergence for representative states",
    )
    save_convergence_plot_all_states(
        mdp,
        policy_result.utility_history,
        environment_output_dir / f"{mdp.name}_policy_iteration_convergence_all_states.png",
        title=f"{mdp.name}: policy iteration convergence for all non-wall states",
    )
    save_convergence_plot_subset(
        mdp,
        policy_result.utility_history,
        selected_states,
        environment_output_dir / f"{mdp.name}_policy_iteration_convergence_selected_states.png",
        title=f"{mdp.name}: policy iteration convergence for representative states",
    )
    save_comparative_convergence(
        mdp,
        value_result,
        policy_result,
        selected_states,
        environment_output_dir / f"{mdp.name}_comparative_convergence.png",
        title=f"{mdp.name}: value iteration vs policy iteration",
    )


def run_single_environment(
    mdp: GridWorldMDP,
    output_root: Path,
    epsilon: float,
    max_iterations: int,
) -> List[Dict[str, object]]:
    environment_output_dir = ensure_directory(output_root / mdp.name)
    print_environment_header(mdp)

    value_result = value_iteration(
        mdp=mdp,
        epsilon=epsilon,
        max_iterations=max_iterations,
    )
    policy_result = policy_iteration(
        mdp=mdp,
        max_iterations=max_iterations,
    )

    print("[Value Iteration]")
    print(f"Converged: {value_result.converged}")
    print(f"Iterations: {value_result.iteration_count}")
    print(f"Runtime: {value_result.runtime_seconds:.6f} seconds")
    print("Utility grid:")
    print(format_utilities_grid(mdp, value_result.utilities))
    print("Policy grid:")
    print(format_policy_grid(mdp, value_result.policy))
    print()

    print("[Policy Iteration]")
    print(f"Converged: {policy_result.converged}")
    print(f"Iterations: {policy_result.iteration_count}")
    print(f"Runtime: {policy_result.runtime_seconds:.6f} seconds")
    print("Utility grid:")
    print(format_utilities_grid(mdp, policy_result.utilities))
    print("Policy grid:")
    print(format_policy_grid(mdp, policy_result.policy))
    print()

    validation_messages = run_validation_checks(mdp, value_result, policy_result)
    print("[Validation]")
    for message in validation_messages:
        print(f"- {message}")
    print()

    export_environment_results(
        mdp=mdp,
        environment_output_dir=environment_output_dir,
        value_result=value_result,
        policy_result=policy_result,
    )

    max_utility_difference = float(
        np.max(np.abs(value_result.utilities - policy_result.utilities))
    )
    policy_match = all(
        value_result.policy[state] == policy_result.policy[state] for state in mdp.states
    )

    return [
        {
            "environment": mdp.name,
            "algorithm": "value_iteration",
            "num_states": mdp.num_states,
            "iterations": value_result.iteration_count,
            "runtime_seconds": value_result.runtime_seconds,
            "converged": value_result.converged,
            "policy_match_between_methods": policy_match,
            "max_utility_difference_between_methods": max_utility_difference,
        },
        {
            "environment": mdp.name,
            "algorithm": "policy_iteration",
            "num_states": mdp.num_states,
            "iterations": policy_result.iteration_count,
            "runtime_seconds": policy_result.runtime_seconds,
            "converged": policy_result.converged,
            "policy_match_between_methods": policy_match,
            "max_utility_difference_between_methods": max_utility_difference,
        },
    ]


def main() -> None:
    args = parse_args()
    output_root = ensure_directory(args.output_dir)

    summary_rows: List[Dict[str, object]] = []
    for environment in selected_environments(args.part):
        summary_rows.extend(
            run_single_environment(
                mdp=environment,
                output_root=output_root,
                epsilon=args.epsilon,
                max_iterations=args.max_iterations,
            )
        )

    export_summary_metrics_csv(output_root / "summary_metrics.csv", summary_rows)
    print(f"Saved figures and CSV exports under: {output_root.resolve()}")


if __name__ == "__main__":
    main()
