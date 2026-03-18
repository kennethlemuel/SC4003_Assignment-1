from __future__ import annotations

from mdp import GridWorldMDP


def create_part1_environment() -> GridWorldMDP:
    """this returns the exact 6x6 assignment maze for Part 1."""
    grid = [
        ["G", "W", "G", "E", "E", "G"],
        ["E", "B", "E", "G", "W", "B"],
        ["E", "E", "B", "E", "G", "E"],
        ["E", "E", "S", "B", "E", "G"],
        ["E", "W", "W", "W", "B", "E"],
        ["E", "E", "E", "E", "E", "E"],
    ]
    representative_states = [
        (3, 2),  #the start cell shown as S in the maze.
        (0, 0),  #top-left green reward cell.
        (1, 3),  #interior green reward cell.
        (1, 5),  #brown penalty cell.
        (5, 0),  #bottom-left empty cell.
        (5, 5),  #bottom-right empty cell.
    ]
    return GridWorldMDP(
        grid=grid,
        gamma=0.99,
        name="part1_maze",
        representative_states=representative_states,
    )


def create_part2_environment() -> GridWorldMDP:
    """this returns a larger custom made maze with bottlenecks and reward-risk trade-offs."""
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
        (4, 0),  #the start cell.
        (0, 4),  #Northern green reward region.
        (4, 2),  #Central green reward region.
        (5, 2),  #a brown cell on a risky corridor.
        (7, 4),  #Southern green with narrow access.
        (9, 6),  #the bottom green reward region.
    ]
    return GridWorldMDP(
        grid=grid,
        gamma=0.99,
        name="part2_complex_maze",
        representative_states=representative_states,
    )
