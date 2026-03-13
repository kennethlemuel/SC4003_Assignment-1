from __future__ import annotations

from mdp import GridWorldMDP


def create_part1_environment() -> GridWorldMDP:
    grid = [
        ["G", "W", "G", "E", "E", "G"],
        ["E", "B", "E", "G", "W", "B"],
        ["E", "E", "B", "E", "G", "E"],
        ["E", "E", "S", "B", "E", "G"],
        ["E", "W", "W", "W", "B", "E"],
        ["E", "E", "E", "E", "E", "E"],
    ]
    representative_states = [
        (3, 2),  # Start cell
        (0, 0),  # Top-left green
        (1, 3),  # Interior green
        (1, 5),  # Brown cell
        (5, 0),  # Bottom-left empty
        (5, 5),  # Bottom-right empty
    ]
    return GridWorldMDP(
        grid=grid,
        gamma=0.99,
        name="part1_maze",
        representative_states=representative_states,
    )


def create_part2_environment() -> GridWorldMDP:
    grid = [
        ["E", "E", "E", "W", "G", "E", "E", "W", "G", "E"],
        ["W", "W", "E", "W", "B", "W", "E", "W", "B", "E"],
        ["E", "E", "E", "E", "E", "W", "E", "E", "E", "E"],
        ["E", "W", "W", "W", "E", "W", "W", "W", "B", "E"],
        ["S", "E", "G", "W", "E", "E", "E", "W", "E", "G"],
        ["E", "W", "B", "W", "W", "W", "E", "W", "E", "W"],
        ["E", "W", "E", "E", "E", "W", "E", "E", "E", "E"],
        ["E", "W", "E", "W", "G", "W", "B", "W", "W", "E"],
        ["E", "E", "E", "W", "E", "E", "E", "E", "W", "E"],
        ["B", "W", "E", "E", "E", "W", "G", "E", "E", "E"],
    ]
    representative_states = [
        (4, 0),  # Start cell
        (0, 4),  # Northern green
        (4, 2),  # Central green
        (5, 2),  # Risky brown on corridor
        (7, 4),  # Southern green with narrow access
        (9, 6),  # Bottom green region
    ]
    return GridWorldMDP(
        grid=grid,
        gamma=0.99,
        name="part2_complex_maze",
        representative_states=representative_states,
    )
