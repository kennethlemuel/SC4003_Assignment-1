from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Mapping

import numpy as np

from mdp import ACTIONS, Action, GridWorldMDP, State


@dataclass
class AlgorithmResult:
    utilities: np.ndarray
    policy: Dict[State, Action]
    utility_history: np.ndarray
    iteration_count: int
    runtime_seconds: float
    converged: bool


def value_iteration(
    mdp: GridWorldMDP,
    epsilon: float = 1e-6,
    max_iterations: int = 20_000,
) -> AlgorithmResult:
    """Run Bellman optimality updates until the residual is below tolerance."""

    utilities = np.zeros(mdp.num_states, dtype=float)
    history = [utilities.copy()]
    start_time = perf_counter()

    if mdp.gamma == 0.0:
        threshold = epsilon
    else:
        threshold = epsilon * (1.0 - mdp.gamma) / mdp.gamma

    converged = False
    iteration_count = 0

    for iteration in range(1, max_iterations + 1):
        new_utilities = np.zeros_like(utilities)
        for state in mdp.states:
            state_index = mdp.state_to_index[state]
            best_action_value = max(
                mdp.expected_next_utility(utilities, state, action)
                for action in ACTIONS
            )
            new_utilities[state_index] = (
                mdp.reward_vector[state_index] + mdp.gamma * best_action_value
            )

        history.append(new_utilities.copy())
        delta = np.max(np.abs(new_utilities - utilities))
        utilities = new_utilities
        iteration_count = iteration

        if delta < threshold:
            converged = True
            break

    runtime_seconds = perf_counter() - start_time
    policy = mdp.derive_policy(utilities)

    return AlgorithmResult(
        utilities=utilities,
        policy=policy,
        utility_history=np.vstack(history),
        iteration_count=iteration_count,
        runtime_seconds=runtime_seconds,
        converged=converged,
    )


def evaluate_policy_exact(
    mdp: GridWorldMDP, policy: Mapping[State, Action]
) -> np.ndarray:
    """Evaluate a fixed policy using exact linear-system solution."""

    policy_transition_matrix = mdp.build_policy_transition_matrix(policy)
    identity = np.eye(mdp.num_states, dtype=float)
    system_matrix = identity - mdp.gamma * policy_transition_matrix

    try:
        return np.linalg.solve(system_matrix, mdp.reward_vector)
    except np.linalg.LinAlgError:
        # Fallback only for numerical safety. The assignment requires exact
        # evaluation via linear equations, so solve() is always attempted first.
        return np.linalg.lstsq(system_matrix, mdp.reward_vector, rcond=None)[0]


def policy_iteration(
    mdp: GridWorldMDP,
    max_iterations: int = 10_000,
) -> AlgorithmResult:
    """Run policy iteration with exact policy evaluation."""

    policy: Dict[State, Action] = {state: ACTIONS[0] for state in mdp.states}
    history = [np.zeros(mdp.num_states, dtype=float)]
    converged = False
    iteration_count = 0
    start_time = perf_counter()

    for iteration in range(1, max_iterations + 1):
        utilities = evaluate_policy_exact(mdp, policy)
        history.append(utilities.copy())

        policy_stable = True
        improved_policy: Dict[State, Action] = {}
        for state in mdp.states:
            best_action = mdp.best_action(utilities, state)
            improved_policy[state] = best_action
            if best_action != policy[state]:
                policy_stable = False

        policy = improved_policy
        iteration_count = iteration
        if policy_stable:
            converged = True
            break

    runtime_seconds = perf_counter() - start_time

    return AlgorithmResult(
        utilities=utilities,
        policy=policy,
        utility_history=np.vstack(history),
        iteration_count=iteration_count,
        runtime_seconds=runtime_seconds,
        converged=converged,
    )
