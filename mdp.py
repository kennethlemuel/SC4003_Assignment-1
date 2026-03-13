from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

State = Tuple[int, int]
Action = str

ACTIONS: Tuple[Action, ...] = ("up", "right", "down", "left")
ACTION_DELTAS: Mapping[Action, Tuple[int, int]] = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
PERPENDICULAR_ACTIONS: Mapping[Action, Tuple[Action, Action]] = {
    "up": ("left", "right"),
    "right": ("up", "down"),
    "down": ("left", "right"),
    "left": ("up", "down"),
}


@dataclass(frozen=True)
class TransitionOutcome:
    next_state: State
    probability: float


class GridWorldMDP:
    """Gridworld MDP with stochastic actions and state-based rewards.

    Coordinate convention:
    - States are stored as `(row, col)` tuples.
    - Both row and col are zero-indexed internally.
    - `(0, 0)` is the top-left grid cell.
    """

    def __init__(
        self,
        grid: Sequence[Sequence[str]],
        gamma: float,
        name: str,
        rewards: Mapping[str, float] | None = None,
        representative_states: Sequence[State] | None = None,
    ) -> None:
        self.name = name
        self.grid = tuple(tuple(cell for cell in row) for row in grid)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.rows else 0
        self.gamma = gamma
        self.rewards = {
            "E": -0.05,
            "S": -0.05,
            "G": 1.0,
            "B": -1.0,
            "W": 0.0,
        }
        if rewards is not None:
            self.rewards.update(rewards)

        self.states: List[State] = []
        self.state_types: Dict[State, str] = {}
        self.reward_by_state: Dict[State, float] = {}
        self.start_state: State | None = None

        for row in range(self.rows):
            for col in range(self.cols):
                cell_type = self.grid[row][col]
                if cell_type == "W":
                    continue
                state = (row, col)
                self.states.append(state)
                self.state_types[state] = cell_type
                self.reward_by_state[state] = self.rewards[cell_type]
                if cell_type == "S":
                    self.start_state = state

        self.state_to_index: Dict[State, int] = {
            state: index for index, state in enumerate(self.states)
        }
        self.index_to_state: Dict[int, State] = {
            index: state for state, index in self.state_to_index.items()
        }
        self.num_states = len(self.states)
        self.reward_vector = np.array(
            [self.reward_by_state[state] for state in self.states],
            dtype=float,
        )
        self.transitions = self._build_transition_model()
        self.transition_matrices = self._build_transition_matrices()
        self.representative_states = self._sanitize_representative_states(
            representative_states
        )

    def _sanitize_representative_states(
        self, representative_states: Sequence[State] | None
    ) -> List[State]:
        if representative_states:
            valid_states = [state for state in representative_states if state in self.state_to_index]
            if valid_states:
                return valid_states

        candidates: List[State] = []
        if self.start_state is not None:
            candidates.append(self.start_state)
        if self.states:
            candidates.extend(
                [
                    self.states[0],
                    self.states[len(self.states) // 3],
                    self.states[(2 * len(self.states)) // 3],
                    self.states[-1],
                ]
            )

        unique_candidates: List[State] = []
        seen = set()
        for state in candidates:
            if state in self.state_to_index and state not in seen:
                seen.add(state)
                unique_candidates.append(state)
        return unique_candidates[:6]

    def _build_transition_model(
        self,
    ) -> Dict[State, Dict[Action, Tuple[TransitionOutcome, ...]]]:
        transitions: Dict[State, Dict[Action, Tuple[TransitionOutcome, ...]]] = {}
        for state in self.states:
            transitions[state] = {}
            for action in ACTIONS:
                probability_by_state: Dict[State, float] = {}
                action_candidates = (
                    (action, 0.8),
                    (PERPENDICULAR_ACTIONS[action][0], 0.1),
                    (PERPENDICULAR_ACTIONS[action][1], 0.1),
                )
                for candidate_action, probability in action_candidates:
                    next_state = self.move(state, candidate_action)
                    probability_by_state[next_state] = (
                        probability_by_state.get(next_state, 0.0) + probability
                    )
                ordered = tuple(
                    TransitionOutcome(next_state=next_state, probability=probability)
                    for next_state, probability in sorted(
                        probability_by_state.items(),
                        key=lambda item: self.state_to_index[item[0]],
                    )
                )
                transitions[state][action] = ordered
        return transitions

    def _build_transition_matrices(self) -> Dict[Action, np.ndarray]:
        matrices: Dict[Action, np.ndarray] = {}
        for action in ACTIONS:
            matrix = np.zeros((self.num_states, self.num_states), dtype=float)
            for state in self.states:
                row = self.state_to_index[state]
                for outcome in self.transitions[state][action]:
                    col = self.state_to_index[outcome.next_state]
                    matrix[row, col] = outcome.probability
            matrices[action] = matrix
        return matrices

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_wall(self, row: int, col: int) -> bool:
        return not self.in_bounds(row, col) or self.grid[row][col] == "W"

    def move(self, state: State, action: Action) -> State:
        row, col = state
        delta_row, delta_col = ACTION_DELTAS[action]
        next_row = row + delta_row
        next_col = col + delta_col
        if self.is_wall(next_row, next_col):
            return state
        return (next_row, next_col)

    def expected_next_utility(
        self, utilities: np.ndarray, state: State, action: Action
    ) -> float:
        return sum(
            outcome.probability * utilities[self.state_to_index[outcome.next_state]]
            for outcome in self.transitions[state][action]
        )

    def best_action(
        self, utilities: np.ndarray, state: State, tie_tolerance: float = 1e-12
    ) -> Action:
        best_action = ACTIONS[0]
        best_value = -np.inf
        for action in ACTIONS:
            value = self.expected_next_utility(utilities, state, action)
            if value > best_value + tie_tolerance:
                best_value = value
                best_action = action
        return best_action

    def derive_policy(self, utilities: np.ndarray) -> Dict[State, Action]:
        return {state: self.best_action(utilities, state) for state in self.states}

    def build_policy_transition_matrix(
        self, policy: Mapping[State, Action]
    ) -> np.ndarray:
        matrix = np.zeros((self.num_states, self.num_states), dtype=float)
        for state in self.states:
            row = self.state_to_index[state]
            action = policy[state]
            for outcome in self.transitions[state][action]:
                col = self.state_to_index[outcome.next_state]
                matrix[row, col] = outcome.probability
        return matrix

    def validate_transition_probabilities(self, tolerance: float = 1e-12) -> List[str]:
        messages: List[str] = []
        for state in self.states:
            for action in ACTIONS:
                total_probability = sum(
                    outcome.probability for outcome in self.transitions[state][action]
                )
                if not np.isclose(total_probability, 1.0, atol=tolerance):
                    messages.append(
                        f"Transition probabilities for state {self.state_label(state)} "
                        f"and action {action} sum to {total_probability:.12f}."
                    )
        return messages

    def utility_grid(self, utilities: np.ndarray) -> np.ndarray:
        grid = np.full((self.rows, self.cols), np.nan, dtype=float)
        for state in self.states:
            row, col = state
            grid[row, col] = utilities[self.state_to_index[state]]
        return grid

    def state_label(self, state: State) -> str:
        row, col = state
        return f"r{row + 1}c{col + 1}"

    def all_non_wall_cells(self) -> Iterable[State]:
        return iter(self.states)
