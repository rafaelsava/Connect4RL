import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4.connect_state import ConnectState
from groups.Rafa.policy import RafaBasePolicy, RafaImprovedPolicy, RafaQPolicy


POLICIES = {
    "RafaBase": RafaBasePolicy,
    "RafaQ": RafaQPolicy,
    "RafaImproved": RafaImprovedPolicy,
}


def make_policy(name: str, total_time: float, prior_visits: int):
    cls = POLICIES[name]
    if name == "RafaBase":
        return cls(total_time=total_time)
    return cls(total_time=total_time, global_prior_visits=prior_visits)


def play_game(red_policy, yellow_policy) -> int:
    red_policy.mount()
    yellow_policy.mount()
    state = ConnectState()
    while not state.is_final():
        policy = red_policy if state.player == -1 else yellow_policy
        action = int(policy.act(state.board))
        if not state.is_applicable(action):
            return 1 if state.player == -1 else -1
        state = state.transition(action)
    return int(state.get_winner())


def evaluate_pair(
    agent_name: str,
    baseline_name: str,
    total_time: float,
    prior_visits: int,
    games_per_side: int,
) -> dict[str, float | int | str]:
    agent_wins = baseline_wins = draws = 0

    for _ in range(games_per_side):
        agent = make_policy(agent_name, total_time, prior_visits)
        baseline = make_policy(baseline_name, total_time, prior_visits)
        winner = play_game(agent, baseline)
        if winner == -1:
            agent_wins += 1
        elif winner == 1:
            baseline_wins += 1
        else:
            draws += 1

        agent = make_policy(agent_name, total_time, prior_visits)
        baseline = make_policy(baseline_name, total_time, prior_visits)
        winner = play_game(baseline, agent)
        if winner == 1:
            agent_wins += 1
        elif winner == -1:
            baseline_wins += 1
        else:
            draws += 1

    total_games = max(1, games_per_side * 2)
    return {
        "agent": agent_name,
        "baseline": baseline_name,
        "total_time": total_time,
        "global_prior_visits": prior_visits,
        "games": total_games,
        "agent_wins": agent_wins,
        "baseline_wins": baseline_wins,
        "draws": draws,
        "agent_win_rate": agent_wins / total_games,
        "baseline_win_rate": baseline_wins / total_games,
        "draw_rate": draws / total_games,
        "points_per_game": (agent_wins + 0.5 * draws) / total_games,
    }


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_time_sweep(df: pd.DataFrame, output: Path) -> None:
    for baseline in sorted(df["baseline"].unique()):
        subset = df[df["baseline"] == baseline]
        plt.figure(figsize=(8, 4.8))
        for agent in sorted(subset["agent"].unique()):
            data = subset[subset["agent"] == agent].sort_values("total_time")
            plt.plot(
                data["total_time"],
                data["points_per_game"],
                marker="o",
                label=agent,
            )
        plt.xscale("log")
        plt.ylim(0, 1.02)
        plt.xlabel("Tiempo por agente/partida (s)")
        plt.ylabel("Puntos por partida")
        plt.title(f"Sensibilidad al tiempo vs {baseline}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output / f"time_sweep_vs_{baseline}.png", dpi=180)
        plt.close()


def plot_prior_sweep(df: pd.DataFrame, output: Path) -> None:
    subset = df[df["agent"].isin(["RafaQ", "RafaImproved"])].copy()
    if subset.empty:
        return

    for total_time in sorted(subset["total_time"].unique()):
        data_time = subset[subset["total_time"] == total_time]
        plt.figure(figsize=(8, 4.8))
        for agent in sorted(data_time["agent"].unique()):
            data = data_time[data_time["agent"] == agent].sort_values("global_prior_visits")
            plt.plot(
                data["global_prior_visits"],
                data["points_per_game"],
                marker="o",
                label=agent,
            )
        plt.ylim(0, 1.02)
        plt.xlabel("global_prior_visits")
        plt.ylabel("Puntos por partida")
        plt.title(f"Sensibilidad al peso de memoria, tiempo={total_time}s")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output / f"prior_sweep_time_{total_time}.png", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep de hiperparametros comparando RafaBase, RafaQ y RafaImproved."
    )
    parser.add_argument("--times", nargs="+", type=float, default=[0.1, 0.3, 1.0, 2.0, 10.0])
    parser.add_argument("--prior-visits", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--games-per-side", type=int, default=10)
    parser.add_argument("--baseline", choices=["RafaBase", "RafaQ"], default="RafaBase")
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=list(POLICIES),
        default=["RafaBase", "RafaQ", "RafaImproved"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("analytics") / "version_sweep",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    jobs = []
    for total_time in args.times:
        for prior in args.prior_visits:
            for agent in args.agents:
                if agent == args.baseline:
                    continue
                if agent == "RafaBase" and prior != args.prior_visits[0]:
                    continue
                jobs.append((agent, args.baseline, total_time, prior))

    print(f"Jobs: {len(jobs)} | games/job: {args.games_per_side * 2}")
    for idx, (agent, baseline, total_time, prior) in enumerate(jobs, start=1):
        row = evaluate_pair(agent, baseline, total_time, prior, args.games_per_side)
        rows.append(row)
        print(
            f"[{idx}/{len(jobs)}] {agent} vs {baseline} | "
            f"time={total_time}s prior={prior} | ppg={row['points_per_game']:.2f}",
            flush=True,
        )

    csv_path = output_dir / "version_sweep.csv"
    write_csv(rows, csv_path)
    df = pd.DataFrame(rows)
    plot_time_sweep(df, output_dir)
    plot_prior_sweep(df, output_dir)

    print(f"\nCSV: {csv_path}")
    print(f"Graficas: {output_dir}")


if __name__ == "__main__":
    main()
