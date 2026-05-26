"""Self-play independiente del agente principal Head.

Ejecuta exclusivamente groups/Fermin_MCTS/policy.py contra si mismo y genera
una grafica de resultados segun el orden de inicio.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns

from connect4.connect_state import ConnectState
from connect4.policy import Policy


POLICY_PATH = Path("groups/Fermin_MCTS/policy.py")
DEFAULT_GAMES = 10
DEFAULT_TURN_LIMITS = [0.10, 0.20, 0.35, 0.50]
DEFAULT_GAMES_PATH = Path("versus/self_play_head_games.csv")
DEFAULT_SUMMARY_PATH = Path("versus/self_play_head_summary.csv")
DEFAULT_FIGURE_PATH = Path("Figura self-play Head.png")


def load_head_class() -> type[Policy]:
    spec = importlib.util.spec_from_file_location("self_play_head_policy", POLICY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"No fue posible importar {POLICY_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    head = getattr(module, "Head", None)
    if not isinstance(head, type) or not issubclass(head, Policy):
        raise ValueError(f"{POLICY_PATH} no expone una clase Head compatible con Policy.")
    return head


def build_head(head_class: type[Policy], turn_limit: float) -> Policy:
    agent = head_class()
    agent.mount(action_timeout=turn_limit)
    return agent


def play_game(
    head_class: type[Policy], game_index: int, turn_limit: float, seed: int
) -> dict[str, object]:
    a_starts = game_index % 2 == 0
    random.seed(seed + game_index)
    head_a = build_head(head_class, turn_limit)
    head_b = build_head(head_class, turn_limit)
    state = ConnectState()
    elapsed = {"A": 0.0, "B": 0.0}
    moves = {"A": 0, "B": 0}

    while not state.is_final():
        a_turn = (state.player == -1 and a_starts) or (
            state.player == 1 and not a_starts
        )
        role = "A" if a_turn else "B"
        agent = head_a if a_turn else head_b
        start = time.perf_counter()
        action = int(agent.act(state.board))
        elapsed[role] += time.perf_counter() - start
        moves[role] += 1
        state = state.transition(action)

    if state.get_winner() == 0:
        winner_role = "Empate"
        outcome = "Empate"
    else:
        first_won = state.get_winner() == -1
        winner_role = "A" if first_won == a_starts else "B"
        starter_role = "A" if a_starts else "B"
        outcome = (
            "Gana quien inicia"
            if winner_role == starter_role
            else "Gana quien juega segundo"
        )

    return {
        "game": game_index + 1,
        "turn_time_limit_s": turn_limit,
        "starts": "A" if a_starts else "B",
        "winner_role": winner_role,
        "outcome": outcome,
        "moves": moves["A"] + moves["B"],
        "elapsed_a_s": elapsed["A"],
        "elapsed_b_s": elapsed["B"],
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_figure(games: pd.DataFrame, figure_path: Path) -> pd.DataFrame:
    order = ["Gana quien inicia", "Gana quien juega segundo", "Empate"]
    summary = (
        games.groupby(["turn_time_limit_s", "outcome"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
    )
    summary["games"] = summary.sum(axis=1)
    proportions = summary[order].div(summary["games"], axis=0)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
    proportions.plot(
        kind="bar",
        stacked=True,
        color=["#31688e", "#35b779", "#bdbdbd"],
        width=0.68,
        ax=ax,
    )
    fig.suptitle("Self-play de Head (MCTS base)", fontsize=19, weight="bold")
    ax.set_title(
        f"{int(summary['games'].iloc[0])} partidas reales por presupuesto; inicio alternado",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel("TURN_TIME_LIMIT (segundos)")
    ax.set_ylabel("Proporcion de partidas")
    ax.set_xticklabels([f"{value:.2f}" for value in summary.index], rotation=0)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(
        title="Resultado",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
    )
    sns.despine()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument(
        "--turn-limits",
        type=float,
        nargs="+",
        default=DEFAULT_TURN_LIMITS,
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--games-path", type=Path, default=DEFAULT_GAMES_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    args = parser.parse_args()

    head_class = load_head_class()
    rows = [
        play_game(head_class, index, turn_limit, args.seed)
        for turn_limit in args.turn_limits
        for index in range(args.games)
    ]
    write_csv(args.games_path, rows)
    games = pd.DataFrame(rows)
    summary = create_figure(games, args.figure_path)
    write_csv(args.summary_path, summary.to_dict(orient="records"))

    print(summary.to_string(index=False))
    print(f"\nPartidas: {args.games_path}")
    print(f"Resumen: {args.summary_path}")
    print(f"Grafica: {args.figure_path}")


if __name__ == "__main__":
    main()
