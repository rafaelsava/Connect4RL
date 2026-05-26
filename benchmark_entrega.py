"""Benchmark externo para la validacion de las politicas de Connect-4.

Ejecuta partidas reales cargando exclusivamente los archivos de politica del
grupo y exporta datos crudos por partida junto con un resumen para el notebook.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import time
from pathlib import Path

from connect4.connect_state import ConnectState
from connect4.policy import Policy


POLICY_PATHS = {
    "Fermin_MCTS": Path("groups/Fermin_MCTS/policy.py"),
    "Fermin_MCTS_Time&C": Path("groups/Fermin_MCTS_Time&C/policy.py"),
    "Random": Path("groups/Random/policy.py"),
}

MATCHUPS = [
    ("Base vs Random", "Fermin_MCTS", "Random"),
    ("Mejorado vs Random", "Fermin_MCTS_Time&C", "Random"),
    ("Base vs Mejorado", "Fermin_MCTS", "Fermin_MCTS_Time&C"),
    ("Mejorado vs Mejorado", "Fermin_MCTS_Time&C", "Fermin_MCTS_Time&C"),
]

DEFAULT_BUDGETS = [0.10, 0.20, 0.35, 0.50]


def load_policy_class(label: str, path: Path) -> type[Policy]:
    module_name = f"benchmark_{label.replace('&', 'and')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No fue posible importar {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = [
        cls
        for cls in vars(module).values()
        if isinstance(cls, type)
        and issubclass(cls, Policy)
        and cls is not Policy
        and cls.__module__ == module.__name__
    ]
    if len(candidates) != 1:
        raise ValueError(f"Se esperaba una unica Policy concreta en {path}: {candidates}")
    return candidates[0]


def build_agent(policy_class: type[Policy], turn_limit: float) -> Policy:
    agent = policy_class()
    if hasattr(agent, "TURN_TIME_LIMIT"):
        agent.TURN_TIME_LIMIT = turn_limit
    try:
        agent.mount(turn_limit)
    except TypeError:
        agent.mount()
    return agent


def play_game(
    policy_classes: dict[str, type[Policy]],
    policy_a: str,
    policy_b: str,
    turn_limit: float,
    game_index: int,
) -> dict[str, object]:
    a_starts = game_index % 2 == 0
    random.seed(1200 + game_index)
    agent_a = build_agent(policy_classes[policy_a], turn_limit)
    agent_b = build_agent(policy_classes[policy_b], turn_limit)
    state = ConnectState()
    elapsed = {"a": 0.0, "b": 0.0}
    moves = {"a": 0, "b": 0}

    while not state.is_final():
        is_role_a = (state.player == -1 and a_starts) or (
            state.player == 1 and not a_starts
        )
        role = "a" if is_role_a else "b"
        agent = agent_a if is_role_a else agent_b
        start = time.perf_counter()
        action = int(agent.act(state.board))
        elapsed[role] += time.perf_counter() - start
        moves[role] += 1
        state = state.transition(action)

    winner = state.get_winner()
    if winner == 0:
        winner_role = "draw"
    else:
        first_won = winner == -1
        winner_role = "a" if first_won == a_starts else "b"

    return {
        "game_index": game_index,
        "a_starts": a_starts,
        "winner_role": winner_role,
        "moves_total": moves["a"] + moves["b"],
        "moves_a": moves["a"],
        "moves_b": moves["b"],
        "elapsed_a_s": elapsed["a"],
        "elapsed_b_s": elapsed["b"],
        "avg_move_ms_a": 1000 * elapsed["a"] / max(1, moves["a"]),
        "avg_move_ms_b": 1000 * elapsed["b"] / max(1, moves["b"]),
    }


def summarize(
    matchup: str,
    policy_a: str,
    policy_b: str,
    turn_limit: float,
    games: list[dict[str, object]],
) -> dict[str, object]:
    wins_a = sum(game["winner_role"] == "a" for game in games)
    wins_b = sum(game["winner_role"] == "b" for game in games)
    draws = sum(game["winner_role"] == "draw" for game in games)
    moves_a = sum(int(game["moves_a"]) for game in games)
    moves_b = sum(int(game["moves_b"]) for game in games)
    elapsed_a = sum(float(game["elapsed_a_s"]) for game in games)
    elapsed_b = sum(float(game["elapsed_b_s"]) for game in games)
    n_games = len(games)
    return {
        "matchup": matchup,
        "agent_a": policy_a,
        "agent_b": policy_b,
        "time_limit_per_turn": turn_limit,
        "games": n_games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a": wins_a / n_games,
        "win_rate_b": wins_b / n_games,
        "draw_rate": draws / n_games,
        "avg_move_ms_a": 1000 * elapsed_a / max(1, moves_a),
        "avg_move_ms_b": 1000 * elapsed_b / max(1, moves_b),
        "source": "benchmark real externo con politicas del repositorio",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    budgets: list[float],
    games_per_condition: int,
    summary_path: Path,
    games_path: Path,
) -> None:
    policy_classes = {
        name: load_policy_class(name, path) for name, path in POLICY_PATHS.items()
    }
    all_games: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for matchup, policy_a, policy_b in MATCHUPS:
        for budget in budgets:
            condition_games: list[dict[str, object]] = []
            condition_start = time.perf_counter()
            for game_index in range(games_per_condition):
                game = play_game(
                    policy_classes, policy_a, policy_b, budget, game_index
                )
                game.update(
                    {
                        "matchup": matchup,
                        "agent_a": policy_a,
                        "agent_b": policy_b,
                        "time_limit_per_turn": budget,
                    }
                )
                condition_games.append(game)
            elapsed = time.perf_counter() - condition_start
            all_games.extend(condition_games)
            summaries.append(
                summarize(matchup, policy_a, policy_b, budget, condition_games)
            )
            write_csv(games_path, all_games)
            write_csv(summary_path, summaries)
            latest = summaries[-1]
            print(
                f"{matchup:24} t={budget:>4.2f}s | "
                f"A={latest['wins_a']} B={latest['wins_b']} "
                f"E={latest['draws']} | {elapsed:.1f}s",
                flush=True,
            )

    print(f"\nDatos por partida: {games_path}")
    print(f"Resumen para notebook: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10, help="Partidas por condicion.")
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=DEFAULT_BUDGETS,
        help="Presupuestos TURN_TIME_LIMIT a evaluar.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("versus/benchmark_entrega.csv"),
    )
    parser.add_argument(
        "--games-path",
        type=Path,
        default=Path("versus/benchmark_entrega_partidas.csv"),
    )
    args = parser.parse_args()
    run_benchmark(args.budgets, args.games, args.summary_path, args.games_path)


if __name__ == "__main__":
    main()
