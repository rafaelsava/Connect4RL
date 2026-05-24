import argparse
import csv
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4.connect_state import ConnectState
from groups.Rafa.policy import RafaRootUCBPolicy
from groups.Rafa.train_self_play import train_one_game


def state_key(board: np.ndarray, player: int) -> tuple[int, ...]:
    return tuple((board * player).astype(np.int8).ravel().tolist())


def lookup(
    q_table: dict[tuple[int, ...], dict[int, list[float | int]]],
    key: tuple[int, ...],
    action: int,
) -> tuple[float, int]:
    q, n = q_table.get(key, {}).get(action, [0.0, 0])
    return float(q), int(n)


def wins_immediately(state: ConnectState, player: int, col: int) -> bool:
    return state.is_applicable(col) and state.transition(col).get_winner() == player


def learned_action(
    q_table: dict[tuple[int, ...], dict[int, list[float | int]]],
    state: ConnectState,
    rng: np.random.Generator,
) -> int:
    actions = [c for c in state.get_free_cols() if state.is_applicable(c)]
    if len(actions) == 1:
        return actions[0]

    for action in actions:
        if wins_immediately(state, state.player, action):
            return action

    opponent_state = ConnectState(board=state.board, player=-state.player)
    for action in actions:
        if wins_immediately(opponent_state, -state.player, action):
            return action

    key = state_key(state.board, state.player)
    center_order = {3: 0, 2: 1, 4: 1, 1: 2, 5: 2, 0: 3, 6: 3}
    scored = []
    for action in actions:
        q, visits = lookup(q_table, key, action)
        prior = -0.01 * center_order[action] if visits == 0 else 0.0
        scored.append((q + prior, action))

    best = max(score for score, _ in scored)
    best_actions = [action for score, action in scored if score == best]
    return int(rng.choice(best_actions))


def random_action(state: ConnectState, rng: np.random.Generator) -> int:
    return int(rng.choice(state.get_free_cols()))


def play_game(red_action, yellow_action, rng: np.random.Generator) -> int:
    state = ConnectState()
    while not state.is_final():
        if state.player == -1:
            action = red_action(state, rng)
        else:
            action = yellow_action(state, rng)
        state = state.transition(int(action))
    return int(state.get_winner())


def evaluate_vs_random(
    q_table: dict[tuple[int, ...], dict[int, list[float | int]]],
    games_per_color: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    def learned(state: ConnectState, local_rng: np.random.Generator) -> int:
        return learned_action(q_table, state, local_rng)

    results: dict[str, float | int] = {}
    for color_name, color in [("red", -1), ("yellow", 1)]:
        wins = losses = draws = 0
        for _ in range(games_per_color):
            if color == -1:
                winner = play_game(learned, random_action, rng)
            else:
                winner = play_game(random_action, learned, rng)

            if winner == color:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1

        prefix = f"random_{color_name}"
        total = max(1, games_per_color)
        results[f"{prefix}_wins"] = wins
        results[f"{prefix}_losses"] = losses
        results[f"{prefix}_draws"] = draws
        results[f"{prefix}_win_rate"] = wins / total
        results[f"{prefix}_loss_rate"] = losses / total
    return results


def evaluate_vs_initial(
    current_table: dict[tuple[int, ...], dict[int, list[float | int]]],
    initial_table: dict[tuple[int, ...], dict[int, list[float | int]]],
    games_per_side: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    def current(state: ConnectState, local_rng: np.random.Generator) -> int:
        return learned_action(current_table, state, local_rng)

    def initial(state: ConnectState, local_rng: np.random.Generator) -> int:
        return learned_action(initial_table, state, local_rng)

    wins = losses = draws = 0
    for _ in range(games_per_side):
        winner = play_game(current, initial, rng)
        if winner == -1:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1

        winner = play_game(initial, current, rng)
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1

    total = max(1, games_per_side * 2)
    return {
        "vs_initial_wins": wins,
        "vs_initial_losses": losses,
        "vs_initial_draws": draws,
        "vs_initial_win_rate": wins / total,
        "vs_initial_loss_rate": losses / total,
    }


def write_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows: list[dict[str, float | int]], output_dir: Path) -> None:
    checkpoints = [int(row["trained_games"]) for row in rows]

    plt.figure(figsize=(8, 4.5))
    plt.plot(checkpoints, [row["random_red_win_rate"] for row in rows], marker="o", label="Rafa rojo vs random")
    plt.plot(checkpoints, [row["random_yellow_win_rate"] for row in rows], marker="o", label="Rafa amarillo vs random")
    plt.plot(checkpoints, [row["vs_initial_win_rate"] for row in rows], marker="o", label="Rafa actual vs Rafa inicial")
    plt.xlabel("Partidas de self-play entrenadas")
    plt.ylabel("Tasa de victoria")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "win_rates.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(checkpoints, [row["q_states"] for row in rows], marker="o", label="Estados con Q-values")
    plt.plot(checkpoints, [row["q_entries"] for row in rows], marker="o", label="Pares (estado, accion)")
    plt.xlabel("Partidas de self-play entrenadas")
    plt.ylabel("Tamano de memoria")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "qtable_growth.png", dpi=180)
    plt.close()


def parse_checkpoints(raw: str) -> list[int]:
    checkpoints = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not checkpoints or checkpoints[0] != 0:
        checkpoints.insert(0, 0)
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera analitica de entrenamiento self-play para el agente Rafa."
    )
    parser.add_argument(
        "--checkpoints",
        default="0,100,500,1000,2500,5000",
        help="Partidas acumuladas donde se evalua, separadas por coma.",
    )
    parser.add_argument("--eval-games", type=int, default=100, help="Juegos vs random por color.")
    parser.add_argument("--self-games", type=int, default=50, help="Juegos por lado contra la version inicial.")
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--ucb-c", type=float, default=1.2)
    parser.add_argument("--inner-rollouts", type=int, default=24)
    parser.add_argument("--budget-mode", choices=["rollouts", "time"], default="rollouts")
    parser.add_argument("--training-total-time", type=float, default=60.0)
    parser.add_argument("--max-turn-time", type=float, default=5.0)
    parser.add_argument(
        "--selection",
        choices=["ucb_rollout", "epsilon_greedy"],
        default="ucb_rollout",
    )
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("analytics"))
    parser.add_argument("--fresh", action="store_true", help="Reinicia la Q-table de analitica.")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    qtable_path = output_dir / "analysis_q_values.pkl"

    if args.fresh and qtable_path.exists():
        qtable_path.unlink()

    rng = np.random.default_rng(args.seed)
    agent = RafaRootUCBPolicy(qtable_path=qtable_path, auto_save=False)
    initial_table = {key: {a: value[:] for a, value in row.items()} for key, row in agent.q_table.items()}

    checkpoints = parse_checkpoints(args.checkpoints)
    rows: list[dict[str, float | int]] = []
    trained_games = 0

    for checkpoint in checkpoints:
        while trained_games < checkpoint:
            train_one_game(
                agent,
                rng,
                args.epsilon,
                args.ucb_c,
                args.inner_rollouts,
                args.budget_mode,
                args.training_total_time,
                args.max_turn_time,
                args.selection,
                args.alpha,
                args.gamma,
            )
            trained_games += 1

        agent.save_q_values(force=True)
        q_entries = sum(len(actions) for actions in agent.q_table.values())
        row: dict[str, float | int] = {
            "trained_games": trained_games,
            "q_states": len(agent.q_table),
            "q_entries": q_entries,
        }
        row.update(evaluate_vs_random(agent.q_table, args.eval_games, rng))
        row.update(evaluate_vs_initial(agent.q_table, initial_table, args.self_games, rng))
        rows.append(row)
        print(
            f"{trained_games:>6} games | "
            f"random rojo={row['random_red_win_rate']:.2f}, "
            f"amarillo={row['random_yellow_win_rate']:.2f}, "
            f"vs inicial={row['vs_initial_win_rate']:.2f}, "
            f"estados={row['q_states']}"
        )

    csv_path = output_dir / "training_metrics.csv"
    write_csv(rows, csv_path)
    plot_metrics(rows, output_dir)

    latest_qtable = output_dir / "latest_analysis_q_values.pkl"
    shutil.copy2(qtable_path, latest_qtable)
    print(f"\nMetricas: {csv_path}")
    print(f"Grafica de desempeno: {output_dir / 'win_rates.png'}")
    print(f"Grafica de memoria: {output_dir / 'qtable_growth.png'}")
    print(f"Q-table entrenada de analitica: {latest_qtable}")


if __name__ == "__main__":
    main()
