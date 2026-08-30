# Connect4RL

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![MCTS](https://img.shields.io/badge/Search-MCTS%20%2B%20UCB1-6C63FF)
![Status](https://img.shields.io/badge/status-academic%20project-2E8B57)

A tournament framework for Connect Four decision policies, featuring a time-bounded Monte Carlo Tree Search agent with UCB1 and tactical safety checks.

## Why this project

Connect Four is small enough to reason about, but large enough to expose the trade-off between planning depth, exploration and strict decision-time budgets. This repository provides a reusable environment for loading policies dynamically, running head-to-head matches and recording every game as JSON.

## Final policy

`groups/Final/policy.py` implements the `Head` agent:

1. **Immediate tactics**: take a winning move when available.
2. **One-step safety**: avoid actions that allow an immediate opponent win whenever a safe alternative exists.
3. **Selection**: traverse expanded nodes using UCB1.
4. **Expansion**: add one unexplored legal action.
5. **Simulation**: finish the game with random rollouts.
6. **Backpropagation**: update visits and alternating-player rewards.
7. **Final decision**: return the most visited action, breaking ties by mean value.

## Search budget

| Parameter | Default | Purpose |
| --- | ---: | --- |
| Exploration constant | `sqrt(2)` | Balance exploitation and exploration |
| Per-turn budget | `2.0 s` | Normal planning time per action |
| Global budget | `58.0 s` | Maximum accumulated search time |
| Minimum reserve | `0.05 s` | Preserve enough time to return a legal action |

## Project structure

```text
.
|-- main.py                  # Discovers policies and starts a tournament
|-- tournament.py            # Match execution and JSON result storage
|-- connect4/
|   |-- connect_state.py     # Board state, transitions and winner detection
|   |-- environment_state.py
|   |-- policy.py            # Policy interface
|   `-- utils.py             # Dynamic policy discovery
`-- groups/
    |-- Final/policy.py      # MCTS + tactical safeguards
    `-- Random/policy.py     # Random baseline
```

## Quick start

```bash
git clone https://github.com/rafaelsava/Connect4RL.git
cd Connect4RL
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

The runner discovers every class derived from `connect4.policy.Policy` under `groups/`, builds the bracket and prints the champion. Match histories are written to `versus/` as JSON.

## Add your own policy

1. Create `groups/<PolicyName>/policy.py`.
2. Implement a class derived from `Policy`.
3. Provide `mount(action_timeout=None)` and `act(board)`.
4. Return a legal column index from `0` to `6`.
5. Run `python main.py`; discovery is automatic.

The board uses `0` for empty cells, `-1` for red and `1` for yellow.

## Reproducibility notes

- Rollouts and bracket order use randomness, so repeated tournaments can differ.
- The current implementation does not expose a global random seed.
- Runtime depends on the number of policies, game duration and configured action limits.
- Tournament results should be reported across multiple runs rather than from a single bracket.

## Possible extensions

- Seeded experiment runs and aggregate win-rate reporting.
- First-Visit Monte Carlo and TBOPI baselines under the same interface.
- Unit tests for terminal-state detection and tactical filters.
- Parallel tournament execution and performance profiling.

## Author

[Rafael Salcedo](https://github.com/rafaelsava) · Computer Engineering · AI and reinforcement-learning projects.
