import argparse
import csv
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


def policy_action(policy: RafaRootUCBPolicy, state: ConnectState, _: np.random.Generator) -> int:
    return int(policy.act(state.board))


def evaluate_vs_random(
    policy: RafaRootUCBPolicy,
    variant: str,
    games_per_color: int,
    rng: np.random.Generator,
    log_every: int,
) -> dict[str, float | int]:
    results: dict[str, float | int] = {}

    for color_name, color in [("red", -1), ("yellow", 1)]:
        wins = losses = draws = 0
        for game in range(1, games_per_color + 1):
            policy.mount()
            if color == -1:
                winner = play_game(lambda s, r: policy_action(policy, s, r), random_action, rng)
            else:
                winner = play_game(random_action, lambda s, r: policy_action(policy, s, r), rng)

            if winner == color:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1

            if game % log_every == 0 or game == games_per_color:
                print(
                    f"  {variant} vs random ({color_name}) "
                    f"{game}/{games_per_color} | W:{wins} L:{losses} D:{draws}",
                    flush=True,
                )

        total = max(1, games_per_color)
        results[f"{color_name}_wins"] = wins
        results[f"{color_name}_losses"] = losses
        results[f"{color_name}_draws"] = draws
        results[f"{color_name}_win_rate"] = wins / total
        results[f"{color_name}_loss_rate"] = losses / total
        results[f"{color_name}_draw_rate"] = draws / total

    return results


def evaluate_head_to_head(
    memory_policy: RafaRootUCBPolicy,
    no_memory_policy: RafaRootUCBPolicy,
    games_per_side: int,
    rng: np.random.Generator,
    log_every: int,
) -> dict[str, float | int]:
    memory_wins = no_memory_wins = draws = 0

    completed = 0
    total_games = games_per_side * 2
    for _ in range(games_per_side):
        memory_policy.mount()
        no_memory_policy.mount()
        winner = play_game(
            lambda s, r: policy_action(memory_policy, s, r),
            lambda s, r: policy_action(no_memory_policy, s, r),
            rng,
        )
        if winner == -1:
            memory_wins += 1
        elif winner == 1:
            no_memory_wins += 1
        else:
            draws += 1
        completed += 1
        if completed % log_every == 0 or completed == total_games:
            print(
                f"  Head-to-head {completed}/{total_games} | "
                f"Q:{memory_wins} NoQ:{no_memory_wins} D:{draws}",
                flush=True,
            )

        memory_policy.mount()
        no_memory_policy.mount()
        winner = play_game(
            lambda s, r: policy_action(no_memory_policy, s, r),
            lambda s, r: policy_action(memory_policy, s, r),
            rng,
        )
        if winner == 1:
            memory_wins += 1
        elif winner == -1:
            no_memory_wins += 1
        else:
            draws += 1
        completed += 1
        if completed % log_every == 0 or completed == total_games:
            print(
                f"  Head-to-head {completed}/{total_games} | "
                f"Q:{memory_wins} NoQ:{no_memory_wins} D:{draws}",
                flush=True,
            )

    total = max(1, games_per_side * 2)
    return {
        "memory_wins": memory_wins,
        "no_memory_wins": no_memory_wins,
        "draws": draws,
        "memory_win_rate": memory_wins / total,
        "no_memory_win_rate": no_memory_wins / total,
        "draw_rate": draws / total,
    }


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_vs_random(rows: list[dict[str, float | int | str]], path: Path) -> None:
    labels = [str(row["variant"]) for row in rows]
    red_rates = [float(row["red_win_rate"]) for row in rows]
    yellow_rates = [float(row["yellow_win_rate"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width / 2, red_rates, width, label="Como rojo")
    plt.bar(x + width / 2, yellow_rates, width, label="Como amarillo")
    plt.xticks(x, labels)
    plt.ylim(0, 1.02)
    plt.ylabel("Win rate vs random")
    plt.title("Impacto de guardar Q-values globales")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_head_to_head(row: dict[str, float | int], path: Path) -> None:
    labels = ["Con Q global", "Sin Q global", "Empates"]
    values = [
        float(row["memory_win_rate"]),
        float(row["no_memory_win_rate"]),
        float(row["draw_rate"]),
    ]

    plt.figure(figsize=(6.5, 4.5))
    plt.bar(labels, values, color=["#2f6f4e", "#b54c3f", "#8c8c8c"])
    plt.ylim(0, 1.02)
    plt.ylabel("Proporcion de partidas")
    plt.title("Rafa con memoria vs Rafa sin memoria")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara Rafa con Q-values persistentes contra Rafa sin memoria global."
    )
    parser.add_argument(
        "--games-vs-random",
        type=int,
        default=0,
        help="Partidas por color contra random. 0 lo omite.",
    )
    parser.add_argument("--games-head-to-head", type=int, default=25, help="Partidas por lado entre variantes.")
    parser.add_argument("--total-time", type=float, default=2.0, help="Tiempo por partida para cada policy.")
    parser.add_argument("--global-prior-visits", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=5, help="Muestra progreso cada N partidas.")
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--qtable", type=Path, default=RafaRootUCBPolicy.DEFAULT_QTABLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("analytics"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    memory_policy = RafaRootUCBPolicy(
        total_time=args.total_time,
        qtable_path=args.qtable,
        auto_save=False,
        global_prior_visits=args.global_prior_visits,
    )
    no_memory_policy = RafaRootUCBPolicy(
        total_time=args.total_time,
        qtable_path=args.qtable,
        auto_save=False,
        global_prior_visits=0,
    )

    print("Comparando Rafa con memoria vs sin memoria")
    print(f"Q-table: {memory_policy.qtable_path}")
    print(f"Estados cargados: {len(memory_policy.q_table)}")
    print(f"Tiempo por policy/partida: {args.total_time}s\n")

    rows: list[dict[str, float | int | str]] = []
    if args.games_vs_random > 0:
        for variant, policy in [
            ("Con Q global", memory_policy),
            ("Sin Q global", no_memory_policy),
        ]:
            metrics = evaluate_vs_random(
                policy, variant, args.games_vs_random, rng, args.log_every
            )
            row: dict[str, float | int | str] = {"variant": variant}
            row.update(metrics)
            rows.append(row)
            print(
                f"{variant}: "
                f"rojo={row['red_win_rate']:.2f}, "
                f"amarillo={row['yellow_win_rate']:.2f}, "
                f"perdidas rojo={row['red_loss_rate']:.2f}, "
                f"perdidas amarillo={row['yellow_loss_rate']:.2f}"
            )
        print()
    else:
        print("Omitiendo evaluacion contra random; ambas variantes suelen ganar siempre.\n")

    h2h = evaluate_head_to_head(
        memory_policy,
        no_memory_policy,
        args.games_head_to_head,
        rng,
        args.log_every,
    )
    print(
        "\nHead-to-head: "
        f"con memoria={h2h['memory_win_rate']:.2f}, "
        f"sin memoria={h2h['no_memory_win_rate']:.2f}, "
        f"empates={h2h['draw_rate']:.2f}"
    )

    if rows:
        write_csv(rows, output_dir / "memory_vs_nomemory_random.csv")
        plot_vs_random(rows, output_dir / "memory_vs_nomemory_random.png")
    write_csv([h2h], output_dir / "memory_vs_nomemory_head_to_head.csv")
    plot_head_to_head(h2h, output_dir / "memory_vs_nomemory_head_to_head.png")

    print(f"CSV head-to-head: {output_dir / 'memory_vs_nomemory_head_to_head.csv'}")
    print(f"Grafica head-to-head: {output_dir / 'memory_vs_nomemory_head_to_head.png'}")
    if rows:
        print(f"CSV random: {output_dir / 'memory_vs_nomemory_random.csv'}")
        print(f"Grafica random: {output_dir / 'memory_vs_nomemory_random.png'}")


if __name__ == "__main__":
    main()
