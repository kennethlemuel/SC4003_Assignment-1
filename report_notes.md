# NTU Intelligent Agents Assignment Notes

## Problem formulation

This assignment models the maze as a discounted Markov Decision Process (MDP). The agent occupies one non-wall grid cell at a time and repeatedly chooses one of four actions: `up`, `right`, `down`, or `left`. The world is stochastic, so the intended move succeeds with probability `0.8`, while the agent slips to one of the two perpendicular directions with probability `0.1` each. If a move would leave the grid or enter a wall cell, the agent remains in the current state.

The objective is to compute an optimal stationary policy that maximizes expected discounted return:

\[
U(s) = R(s) + \gamma \max_a \sum_{s'} P(s' \mid s, a) U(s')
\]

for value iteration, and to solve the equivalent control problem using policy iteration with exact policy evaluation.

## MDP definition

### States

- Each non-wall cell is a valid state.
- Internal coordinates are stored as `(row, col)` with zero-based indexing.
- `(0, 0)` is the top-left cell.
- In the report, it is usually clearer to refer to cells using one-based row and column labels, which is how the CSV exports and plot labels are presented (for example `r4c3` means row 4, column 3).

### Actions

- `up`
- `right`
- `down`
- `left`

The deterministic tie-breaking order for action selection is exactly the same in both algorithms:

1. `up`
2. `right`
3. `down`
4. `left`

This makes the output reproducible even when multiple actions have equal value.

### Transition model

For every state-action pair, the transition model is built explicitly and reused by both algorithms. Probability mass is distributed as:

- intended direction: `0.8`
- first perpendicular direction: `0.1`
- second perpendicular direction: `0.1`

If multiple stochastic outcomes map to the same next state because of a wall or boundary, the implementation combines those probabilities into a single transition entry. The code also validates that every state-action distribution sums to `1.0`.

### Reward model

Rewards are state-based:

- empty white cell `E`: `-0.05`
- start cell `S`: `-0.05`
- green cell `G`: `+1`
- brown cell `B`: `-1`

The start cell is treated as a normal traversable white cell with reward `-0.05`. This is consistent with the assignment statement because there are no terminal states, so `S` only identifies a reference location and does not imply any special transition or reward dynamics.

### Discount factor

- `gamma = 0.99`

## Important assumptions

- There are **no terminal states**.
- Green and brown cells are **not** absorbing.
- Rewards are attached to the current state, not to the action.
- Hitting a wall or the grid boundary results in the agent staying in the same state.
- Because `gamma < 1`, utilities remain finite even though the process continues forever.

## Why there are no terminal states

This assignment explicitly states that there are no terminal states. Therefore, even after the agent enters a green or brown cell, it continues acting and collecting future discounted rewards. This changes the interpretation of the optimal policy compared with the standard textbook gridworld: the agent is not trying to reach a terminal cell once and stop. Instead, it is trying to position itself in regions of the maze that give strong long-run discounted return under stochastic movement.

## Value iteration

Value iteration applies the Bellman optimality update to every state:

\[
U_{k+1}(s) = R(s) + \gamma \max_a \sum_{s'} P(s' \mid s, a) U_k(s')
\]

The implementation:

- initializes utilities to zero
- updates all state utilities synchronously
- stores the full utility vector after every iteration
- stops when the Bellman residual falls below the tolerance-based threshold
- derives the final greedy policy using the fixed tie-breaking order

The history arrays are exported to CSV and plotted so that utility trajectories can be shown in a report in the same spirit as AIMA Figure 17.5(a).

## Policy iteration

Policy iteration alternates between:

1. exact policy evaluation
2. policy improvement

### Exact policy evaluation

For a fixed policy `pi`, the utility vector is obtained by solving the linear system:

\[
(I - \gamma P_\pi) U = R
\]

where:

- `P_pi` is the transition matrix induced by the current policy
- `U` is the utility vector
- `R` is the reward vector

The implementation constructs `P_pi` explicitly and then uses `numpy.linalg.solve` as the primary method. A least-squares fallback is included only as a numerical safeguard if the linear solver raises an exception. This is preferable to approximate iterative policy evaluation in this assignment because:

- it matches the mathematical policy-evaluation equation exactly
- it removes ambiguity about how many sweeps of evaluation are “enough”
- it gives a cleaner and fairer comparison against value iteration
- it makes policy iteration stable and reproducible for report-quality results

### Policy improvement

After exact evaluation, the algorithm performs one-step lookahead over all four actions in every state and greedily updates the policy. The process terminates only when the policy no longer changes.

## Part 1 environment

The Part 1 maze is exactly the `6 x 6` environment given in the assignment brief. Wall cells are excluded from the state space, giving a smaller effective state set than the raw grid size.

Outputs produced for Part 1:

- optimal policy from value iteration
- optimal policy from policy iteration
- final utilities for all non-wall states
- utility heatmaps
- convergence plots for all states
- cleaner convergence plots for representative states
- a comparative convergence figure
- CSV exports for utility histories, final utilities, policies, and summary metrics

## Part 1 results summary

In the generated run for this workspace, Part 1 produced:

- `31` non-wall states
- value iteration: `1833` iterations
- policy iteration: `5` improvement cycles
- identical optimal policies from both methods
- maximum absolute utility difference between methods of about `1e-6`

The exact numerical results are also exported into `outputs/summary_metrics.csv` and the per-environment CSV files. The main interpretation is:

- value iteration requires many more iterations because it improves utilities gradually through Bellman backups
- policy iteration converges in far fewer improvement cycles because each cycle evaluates the current policy exactly

These claims are verified automatically by the validation checks printed to the console.

## Part 2 environment

The custom Part 2 maze is intentionally more difficult than Part 1. It is larger and introduces:

- a larger state space
- more walls
- bottlenecks
- longer corridors
- separated reward regions
- brown penalty cells placed near narrow passages
- positive regions that are attractive but risky to access under stochastic slipping

This makes the policy design problem more interesting because the agent must balance long-run reward against the chance of drifting into penalties or becoming trapped in low-value corridors.

## Part 2 results summary

In the generated run for this workspace, Part 2 produced:

- `67` non-wall states
- value iteration: `1821` iterations
- policy iteration: `7` improvement cycles
- identical optimal policies from both methods
- maximum absolute utility difference between methods of about `1e-6`

The exact observed runtimes are written to `outputs/summary_metrics.csv`, and the main qualitative findings to discuss in the report are:

- increasing the number of states usually increases runtime for both algorithms
- topology matters, not just size: bottlenecks, wall density, and risky routes can slow the propagation of useful value information
- value iteration is more sensitive to slow utility propagation across long corridors
- policy iteration often needs fewer outer iterations, but each iteration requires solving a linear system whose cost grows with the number of states
- the learned policy remains reasonable when it stabilizes and routes toward high long-run reward regions while avoiding brown cells unless the risk is justified

## Comparison between methods

For this assignment, the comparison should focus on both iteration count and computational cost:

- **Value iteration** is conceptually simple and easy to visualize, but it can require many iterations when `gamma` is high and the environment is large.
- **Policy iteration** usually requires fewer policy-improvement cycles, but each cycle is heavier because of the linear solve.
- With exact policy evaluation, policy iteration gives a cleaner benchmark than approximate evaluation.
- Because both algorithms use the same transition model, reward vector, and tie-breaking rule, their outputs can be compared directly.

## Interpreting convergence behaviour

The convergence plots should be interpreted in terms of both magnitude and policy stability:

- Utility curves flatten as the algorithms converge.
- States near influential reward regions often stabilize earlier than distant states.
- In more complex mazes, information can propagate more slowly through narrow corridors or around walls.
- A policy should be considered as “still learning the right policy” only if:
  - the policy has stabilized
  - utilities have converged numerically
  - the action choices qualitatively make sense under stochastic motion

Operationally, this assignment defines “learning the right policy” as producing a stable policy whose utility estimates have converged and whose routing behaviour is consistent with long-run stochastic reward trade-offs.

## Mapping outputs into the final report PDF

A clean report structure is:

1. problem formulation and MDP definition
2. transition/reward modelling details
3. value iteration implementation
4. policy iteration with exact linear-system evaluation
5. Part 1 results
6. Part 2 results
7. comparative discussion
8. conclusion

Suggested figures and tables:

- use the policy grid image and utility heatmap for each environment
- use the selected-state convergence plots in the main body
- move the all-state convergence plots to the appendix if they are visually dense
- build a small table from `outputs/summary_metrics.csv` for states, iterations, and runtime
- quote the validation checks briefly to show that both methods agree
