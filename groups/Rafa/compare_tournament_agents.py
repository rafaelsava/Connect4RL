import argparse
import csv
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4.connect_state import ConnectState
from connect4.policy import Policy
from connect4.utils import find_importable_classes
from groups.Rafa.policy import (
    RafaBasePolicy,
    RafaImprovedPolicy,
    RafaNoMemoryPolicy,
    RafaQPolicy,
)


def make_policy(policy_cls: type[Policy], total_time: float) -> Policy:
    try:
        return policy_cls(total_time=total_time)
    except TypeError:
        return policy_cls()


def mount_policy(policy: Policy, total_time: float) -> None:
    try:
        policy.mount(total_time=total_time)
    except TypeError:
        policy.mount()


def call_action(policy: Policy, state: ConnectState) -> tuple[int, float]:
    start = time.perf_counter()
    action = int(policy.act(state.board))
    return action, time.perf_counter() - start


def play_game(
    red_name: str,
    red_cls: type[Policy],
    yellow_name: str,
    yellow_cls: type[Policy],
    total_time: float,
) -> dict[str, object]:
    red_policy = make_policy(red_cls, total_time)
    yellow_policy = make_policy(yellow_cls, total_time)
    mount_policy(red_policy, total_time)
    mount_policy(yellow_policy, total_time)

    policies = {-1: red_policy, 1: yellow_policy}
    names = {-1: red_name, 1: yellow_name}
    time_used = {-1: 0.0, 1: 0.0}
    moves: list[dict[str, object]] = []
    invalid_by: str | None = None
    state = ConnectState()

    while not state.is_final():
        player = state.player
        policy = policies[player]
        try:
            action, elapsed = call_action(policy, state)
        except Exception as exc:
            invalid_by = names[player]
            moves.append(
                {
                    "player": names[player],
                    "error": repr(exc),
                    "board": state.board.copy().tolist(),
                }
            )
            break

        time_used[player] += elapsed
        moves.append(
            {
                "player": names[player],
                "color": "red" if player == -1 else "yellow",
                "action": action,
                "elapsed": elapsed,
                "board": state.board.copy().tolist(),
            }
        )

        if not state.is_applicable(action):
            invalid_by = names[player]
            break

        state = state.transition(action)

    if invalid_by is not None:
        winner_name = yellow_name if invalid_by == red_name else red_name
        winner_color = 1 if invalid_by == red_name else -1
    else:
        winner_color = int(state.get_winner())
        winner_name = "draw" if winner_color == 0 else names[winner_color]

    return {
        "red": red_name,
        "yellow": yellow_name,
        "winner": winner_name,
        "winner_color": winner_color,
        "moves_count": len(moves),
        "red_time": time_used[-1],
        "yellow_time": time_used[1],
        "red_over_time": time_used[-1] > total_time,
        "yellow_over_time": time_used[1] > total_time,
        "invalid_by": invalid_by,
        "moves": moves,
    }


def init_scoreboard(agent_names: list[str]) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "points": 0.0,
            "red_games": 0,
            "red_wins": 0,
            "yellow_games": 0,
            "yellow_wins": 0,
            "over_time_games": 0,
            "invalid_games": 0,
        }
        for name in agent_names
    }


def update_scoreboard(
    scoreboard: dict[str, dict[str, float | int]],
    result: dict[str, object],
) -> None:
    red = str(result["red"])
    yellow = str(result["yellow"])
    winner = str(result["winner"])

    scoreboard[red]["games"] += 1
    scoreboard[yellow]["games"] += 1
    scoreboard[red]["red_games"] += 1
    scoreboard[yellow]["yellow_games"] += 1

    if result["red_over_time"]:
        scoreboard[red]["over_time_games"] += 1
    if result["yellow_over_time"]:
        scoreboard[yellow]["over_time_games"] += 1

    invalid_by = result["invalid_by"]
    if invalid_by:
        scoreboard[str(invalid_by)]["invalid_games"] += 1

    if winner == "draw":
        scoreboard[red]["draws"] += 1
        scoreboard[yellow]["draws"] += 1
        scoreboard[red]["points"] += 0.5
        scoreboard[yellow]["points"] += 0.5
    elif winner == red:
        scoreboard[red]["wins"] += 1
        scoreboard[yellow]["losses"] += 1
        scoreboard[red]["points"] += 1.0
        scoreboard[red]["red_wins"] += 1
    else:
        scoreboard[yellow]["wins"] += 1
        scoreboard[red]["losses"] += 1
        scoreboard[yellow]["points"] += 1.0
        scoreboard[yellow]["yellow_wins"] += 1


def write_csv(
    rows: list[dict[str, object]],
    path: Path,
    append: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    write_header = not append or not path.exists() or path.stat().st_size == 0
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara agentes en formato round-robin estilo torneo."
    )
    parser.add_argument("--games-per-match", type=int, default=10)
    parser.add_argument("--total-time", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--include", nargs="*", default=None, help="Agentes a incluir.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Agentes a excluir.")
    parser.add_argument("--target", default=None, help="Agente fijo para jugar solo contra opponents.")
    parser.add_argument("--opponents", nargs="*", default=None, help="Oponentes del target.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("analytics") / "tournament_style",
    )
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribe CSV en vez de agregar filas.")
    parser.add_argument("--save-games-json", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    agents = find_importable_classes("groups", Policy)
    agents["RafaNoQ"] = RafaNoMemoryPolicy
    agents["RafaBase"] = RafaBasePolicy
    agents["RafaQ"] = RafaQPolicy
    agents["RafaImproved"] = RafaImprovedPolicy

    if args.target:
        requested = [args.target] + (args.opponents or [])
        missing = [name for name in requested if name not in agents]
        if missing:
            raise ValueError(f"No se encontraron agentes: {', '.join(missing)}")
        agents = {name: agents[name] for name in requested if name in agents}
    elif args.include:
        agents = {name: agents[name] for name in args.include if name in agents}
    if args.exclude:
        agents = {name: cls for name, cls in agents.items() if name not in args.exclude}

    agent_names = sorted(agents)
    if len(agent_names) < 2:
        raise ValueError("Se necesitan al menos dos agentes para comparar.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Agentes:", ", ".join(agent_names))
    if args.target:
        print(f"Modo target: {args.target} vs {', '.join(args.opponents or [])}")
    print(f"Partidas por pareja: {args.games_per_match}")
    print(f"Tiempo por agente/partida: {args.total_time}s")
    print(f"Salida: {output_dir}\n")

    scoreboard = init_scoreboard(agent_names)
    match_rows: list[dict[str, object]] = []
    game_summaries: list[dict[str, object]] = []
    game_details: list[dict[str, object]] = []

    if args.target:
        pairs = [(args.target, opponent) for opponent in (args.opponents or [])]
    else:
        pairs = list(combinations(agent_names, 2))
    total_games = len(pairs) * args.games_per_match
    completed_games = 0
    start = time.perf_counter()

    for pair_index, (a_name, b_name) in enumerate(pairs, start=1):
        a_wins = b_wins = draws = 0
        print(f"[{pair_index}/{len(pairs)}] {a_name} vs {b_name}")

        for game_index in range(args.games_per_match):
            if game_index % 2 == 0:
                red_name, yellow_name = a_name, b_name
            else:
                red_name, yellow_name = b_name, a_name

            result = play_game(
                red_name,
                agents[red_name],
                yellow_name,
                agents[yellow_name],
                args.total_time,
            )
            update_scoreboard(scoreboard, result)

            winner = str(result["winner"])
            if winner == a_name:
                a_wins += 1
            elif winner == b_name:
                b_wins += 1
            else:
                draws += 1

            completed_games += 1
            elapsed = time.perf_counter() - start
            rate = completed_games / max(elapsed, 1e-9)
            eta = (total_games - completed_games) / max(rate, 1e-9)

            game_summary = {
                "run_started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "game": completed_games,
                "match": f"{a_name} vs {b_name}",
                "red": result["red"],
                "yellow": result["yellow"],
                "winner": result["winner"],
                "winner_color": result["winner_color"],
                "moves_count": result["moves_count"],
                "red_time": round(float(result["red_time"]), 4),
                "yellow_time": round(float(result["yellow_time"]), 4),
                "red_over_time": result["red_over_time"],
                "yellow_over_time": result["yellow_over_time"],
                "invalid_by": result["invalid_by"] or "",
            }
            game_summaries.append(game_summary)

            if args.save_games_json:
                game_details.append(result)

            print(
                f"  game {game_index + 1}/{args.games_per_match} | "
                f"winner={winner} | score {a_name}:{a_wins} {b_name}:{b_wins} D:{draws} | "
                f"ETA {int(eta)}s",
                flush=True,
            )

        match_rows.append(
            {
                "run_started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "agent_a": a_name,
                "agent_b": b_name,
                "a_wins": a_wins,
                "b_wins": b_wins,
                "draws": draws,
                "games": args.games_per_match,
            }
        )
        print()

    ranking_rows = []
    for name, stats in scoreboard.items():
        games = max(1, int(stats["games"]))
        ranking_rows.append(
            {
                "agent": name,
                "run_started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "points": stats["points"],
                "games": stats["games"],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "draws": stats["draws"],
                "win_rate": round(float(stats["wins"]) / games, 4),
                "points_per_game": round(float(stats["points"]) / games, 4),
                "red_wins": stats["red_wins"],
                "red_games": stats["red_games"],
                "yellow_wins": stats["yellow_wins"],
                "yellow_games": stats["yellow_games"],
                "over_time_games": stats["over_time_games"],
                "invalid_games": stats["invalid_games"],
            }
        )
    ranking_rows.sort(key=lambda row: (row["points"], row["wins"]), reverse=True)

    append = not args.overwrite
    write_csv(ranking_rows, output_dir / "ranking.csv", append=append)
    write_csv(match_rows, output_dir / "matches.csv", append=append)
    write_csv(game_summaries, output_dir / "games.csv", append=append)

    if args.save_games_json:
        with (output_dir / "games_detailed.json").open("w", encoding="utf-8") as f:
            json.dump(game_details, f, indent=2)

    print("Ranking final")
    for i, row in enumerate(ranking_rows, start=1):
        print(
            f"{i}. {row['agent']}: {row['points']} pts | "
            f"W:{row['wins']} L:{row['losses']} D:{row['draws']} | "
            f"PPG:{row['points_per_game']}"
        )

    print(f"\nCSV ranking: {output_dir / 'ranking.csv'}")
    print(f"CSV matches: {output_dir / 'matches.csv'}")
    print(f"CSV games: {output_dir / 'games.csv'}")


if __name__ == "__main__":
    main()
