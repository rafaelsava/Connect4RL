import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4.connect_state import ConnectState
from groups.Rafa.policy import RafaBasePolicy, RafaImprovedPolicy, RafaQPolicy, RafaRootUCBPolicy
from groups.Rafa.train_self_play import train_one_game


POLICIES = {
    "RafaBase": RafaBasePolicy,
    "RafaQ": RafaQPolicy,
    "RafaImproved": RafaImprovedPolicy,
}


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def random_action(state: ConnectState, rng: np.random.Generator) -> int:
    return int(rng.choice(state.get_free_cols()))


def policy_action(policy, state: ConnectState, _: np.random.Generator) -> int:
    return int(policy.act(state.board))


def play_game(red_action, yellow_action, rng: np.random.Generator) -> tuple[int, int]:
    state = ConnectState()
    moves = 0
    while not state.is_final():
        action = red_action(state, rng) if state.player == -1 else yellow_action(state, rng)
        if not state.is_applicable(action):
            return (1 if state.player == -1 else -1), moves + 1
        state = state.transition(int(action))
        moves += 1
    return int(state.get_winner()), moves


def make_policy(name: str, total_time: float):
    return POLICIES[name](total_time=total_time)


def evaluate_vs_random(
    versions: list[str],
    games_per_color: int,
    total_time: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for version in versions:
        for color_name, color in [("red", -1), ("yellow", 1)]:
            wins = losses = draws = total_moves = 0
            for _ in range(games_per_color):
                policy = make_policy(version, total_time)
                policy.mount(total_time=total_time)
                if color == -1:
                    winner, moves = play_game(lambda s, r: policy_action(policy, s, r), random_action, rng)
                else:
                    winner, moves = play_game(random_action, lambda s, r: policy_action(policy, s, r), rng)

                total_moves += moves
                if winner == color:
                    wins += 1
                elif winner == 0:
                    draws += 1
                else:
                    losses += 1

            games = max(1, games_per_color)
            rows.append(
                {
                    "version": version,
                    "color": color_name,
                    "opponent": "Random",
                    "total_time": total_time,
                    "games": games_per_color,
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "win_rate": wins / games,
                    "loss_rate": losses / games,
                    "draw_rate": draws / games,
                    "avg_moves": total_moves / games,
                }
            )
            print(
                f"Random | {version} como {color_name}: "
                f"W:{wins} L:{losses} D:{draws}",
                flush=True,
            )
    return rows


def evaluate_self_play(
    versions: list[str],
    games: int,
    total_time: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for version in versions:
        red_wins = yellow_wins = draws = total_moves = 0
        for _ in range(games):
            red_policy = make_policy(version, total_time)
            yellow_policy = make_policy(version, total_time)
            red_policy.mount(total_time=total_time)
            yellow_policy.mount(total_time=total_time)
            winner, moves = play_game(
                lambda s, r: policy_action(red_policy, s, r),
                lambda s, r: policy_action(yellow_policy, s, r),
                rng,
            )
            total_moves += moves
            if winner == -1:
                red_wins += 1
            elif winner == 1:
                yellow_wins += 1
            else:
                draws += 1

        total = max(1, games)
        rows.append(
            {
                "version": version,
                "total_time": total_time,
                "games": games,
                "red_wins": red_wins,
                "yellow_wins": yellow_wins,
                "draws": draws,
                "red_win_rate": red_wins / total,
                "yellow_win_rate": yellow_wins / total,
                "draw_rate": draws / total,
                "avg_moves": total_moves / total,
            }
        )
        print(
            f"Self-play | {version}: rojo={red_wins} amarillo={yellow_wins} empates={draws}",
            flush=True,
        )
    return rows


def parse_checkpoints(raw: str) -> list[int]:
    checkpoints = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not checkpoints or checkpoints[0] != 0:
        checkpoints.insert(0, 0)
    return checkpoints


def count_q_entries(q_table: dict) -> int:
    return sum(len(actions) for actions in q_table.values())


def evaluate_training_checkpoint(
    agent: RafaRootUCBPolicy,
    trained_games: int,
    eval_games_per_color: int,
    eval_total_time: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    color_metrics: dict[str, dict[str, float | int]] = {}
    for color_name, color in [("red", -1), ("yellow", 1)]:
        wins = losses = draws = 0
        for _ in range(eval_games_per_color):
            agent.mount(total_time=eval_total_time)
            if color == -1:
                winner, _ = play_game(lambda s, r: policy_action(agent, s, r), random_action, rng)
            else:
                winner, _ = play_game(random_action, lambda s, r: policy_action(agent, s, r), rng)
            if winner == color:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1
        total = max(1, eval_games_per_color)
        color_metrics[color_name] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / total,
            "loss_rate": losses / total,
            "draw_rate": draws / total,
        }

    return {
        "trained_games": trained_games,
        "q_states": len(agent.q_table),
        "q_entries": count_q_entries(agent.q_table),
        "random_red_win_rate": color_metrics["red"]["win_rate"],
        "random_red_loss_rate": color_metrics["red"]["loss_rate"],
        "random_yellow_win_rate": color_metrics["yellow"]["win_rate"],
        "random_yellow_loss_rate": color_metrics["yellow"]["loss_rate"],
    }


def evaluate_training_curve(args, rng: np.random.Generator) -> list[dict[str, object]]:
    checkpoints = parse_checkpoints(args.training_checkpoints)
    rows: list[dict[str, object]] = []
    trained_games = 0

    with tempfile.TemporaryDirectory(prefix="rafa_training_curve_") as tmp_dir:
        qtable_path = Path(tmp_dir) / "curve_q_values.pkl"
        agent = RafaRootUCBPolicy(
            total_time=args.eval_total_time,
            qtable_path=qtable_path,
            auto_save=False,
            global_prior_visits=args.global_prior_visits,
            rollout_policy="heuristic",
            final_selection="robust",
        )

        for checkpoint in checkpoints:
            while trained_games < checkpoint:
                train_one_game(
                    agent,
                    rng,
                    args.epsilon,
                    args.ucb_c,
                    args.inner_rollouts,
                    "rollouts",
                    60.0,
                    5.0,
                    "ucb_rollout",
                    args.alpha,
                    args.gamma,
                )
                trained_games += 1

            row = evaluate_training_checkpoint(
                agent,
                trained_games,
                args.training_eval_games,
                args.eval_total_time,
                rng,
            )
            rows.append(row)
            print(
                f"Training curve | {trained_games} games: "
                f"red={row['random_red_win_rate']:.2f}, "
                f"yellow={row['random_yellow_win_rate']:.2f}, "
                f"states={row['q_states']}",
                flush=True,
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera analitica local exigida por reto.pdf: random, self-play y curva offline."
    )
    parser.add_argument("--versions", nargs="+", choices=list(POLICIES), default=["RafaBase", "RafaQ", "RafaImproved"])
    parser.add_argument("--games-vs-random", type=int, default=10)
    parser.add_argument("--self-play-games", type=int, default=10)
    parser.add_argument("--total-time", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("analytics") / "reto_requirements")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--skip-self-play", action="store_true")
    parser.add_argument("--skip-training-curve", action="store_true")

    parser.add_argument("--training-checkpoints", default="0,25,50,100,250")
    parser.add_argument("--training-eval-games", type=int, default=5)
    parser.add_argument("--eval-total-time", type=float, default=0.3)
    parser.add_argument("--inner-rollouts", type=int, default=8)
    parser.add_argument("--global-prior-visits", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--ucb-c", type=float, default=1.2)
    parser.add_argument("--alpha", type=float, default=0.04)
    parser.add_argument("--gamma", type=float, default=0.995)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_random:
        random_rows = evaluate_vs_random(args.versions, args.games_vs_random, args.total_time, rng)
        write_csv(random_rows, output_dir / "random_vs_versions.csv")

    if not args.skip_self_play:
        self_rows = evaluate_self_play(args.versions, args.self_play_games, args.total_time, rng)
        write_csv(self_rows, output_dir / "self_play_versions.csv")

    if not args.skip_training_curve:
        curve_rows = evaluate_training_curve(args, rng)
        write_csv(curve_rows, output_dir / "training_curve.csv")

    print(f"\nCSV output: {output_dir}")


if __name__ == "__main__":
    main()
