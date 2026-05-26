import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


GROUP_DIR = Path(__file__).resolve().parent
ROOT_DIR = GROUP_DIR.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ARTIFACTS_DIR = GROUP_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy_module = load_module_from_file("group_b_policy", GROUP_DIR / "policy.py")
connect_state_module = load_module_from_file(
    "connect4_state",
    ROOT_DIR / "connect4" / "connect_state.py",
)

FVMCAgent = policy_module.FVMCAgent
ConnectState = connect_state_module.ConnectState


DEFAULT_EXPERIMENTS = [
    {
        "experiment_id": "exp_random_5k_eps020",
        "label": "random_only_5000_eps_0.20",
        "train_phases": [
            {"mode": "random", "episodes": 5000, "epsilon": 0.20},
        ],
    },
    {
        "experiment_id": "exp_random_10k_eps020",
        "label": "random_only_10000_eps_0.20",
        "train_phases": [
            {"mode": "random", "episodes": 10000, "epsilon": 0.20},
        ],
    },
    {
        "experiment_id": "exp_self_10k_eps020",
        "label": "self_only_10000_eps_0.20",
        "train_phases": [
            {"mode": "self", "episodes": 10000, "epsilon": 0.20},
        ],
    },
    {
        "experiment_id": "exp_self_20k_eps010",
        "label": "self_only_20000_eps_0.10",
        "train_phases": [
            {"mode": "self", "episodes": 20000, "epsilon": 0.10},
        ],
    },
    {
        "experiment_id": "exp_mixed_8k_12k",
        "label": "random_8000_then_self_12000",
        "train_phases": [
            {"mode": "random", "episodes": 8000, "epsilon": 0.20},
            {"mode": "self", "episodes": 12000, "epsilon": 0.10},
        ],
    },
    {
        "experiment_id": "exp_mixed_8k_12k_refine",
        "label": "random_8000_then_self_12000_then_self_8000",
        "train_phases": [
            {"mode": "random", "episodes": 8000, "epsilon": 0.20},
            {"mode": "self", "episodes": 12000, "epsilon": 0.10},
            {"mode": "self", "episodes": 8000, "epsilon": 0.05},
        ],
    },
    {
        "experiment_id": "exp_self_refine_15k",
        "label": "self_refine_10000_then_5000",
        "train_phases": [
            {"mode": "self", "episodes": 10000, "epsilon": 0.08},
            {"mode": "self", "episodes": 5000, "epsilon": 0.03},
        ],
    },
]


class RandomPolicy:
    def __init__(self, seed: int = 911):
        self.rng = np.random.default_rng(seed)

    def mount(self):
        pass

    def act(self, s: np.ndarray) -> int:
        cols = [c for c in range(7) if s[0, c] == 0]
        return int(self.rng.choice(cols))


def clone_agent(source_agent, seed: int = 911):
    clone = FVMCAgent(auto_train_if_missing=False, seed=seed)
    clone._is_ready = True
    clone.Q = defaultdict(float, source_agent.Q)
    clone.N = defaultdict(int, source_agent.N)
    clone.training_summary = deepcopy(source_agent.training_summary)
    return clone


def frozen_factory_from_agent(source_agent):
    def factory(seed: int):
        return clone_agent(source_agent, seed=seed)

    return factory


def play_single_game(policy_neg, policy_pos) -> int:
    state = ConnectState()
    policy_neg.mount()
    policy_pos.mount()

    while not state.is_final():
        current_policy = policy_neg if state.player == -1 else policy_pos
        action = int(current_policy.act(state.board.copy()))
        state = state.transition(action)

    return int(state.get_winner())


def evaluate_factories(factory_neg, factory_pos, num_games: int, seed: int):
    rng = np.random.default_rng(seed)
    wins_neg = 0
    wins_pos = 0
    draws = 0

    for _ in range(num_games):
        policy_neg = factory_neg(int(rng.integers(0, 1_000_000)))
        policy_pos = factory_pos(int(rng.integers(0, 1_000_000)))
        result = play_single_game(policy_neg, policy_pos)

        if result == -1:
            wins_neg += 1
        elif result == 1:
            wins_pos += 1
        else:
            draws += 1

    return wins_neg, wins_pos, draws


def evaluate_agent_balanced(agent_factory, opponent_factory, num_games: int, seed: int):
    first_half = num_games // 2
    second_half = num_games - first_half

    first_wins, first_losses, first_draws = evaluate_factories(
        agent_factory,
        opponent_factory,
        num_games=first_half,
        seed=seed,
    )
    second_losses, second_wins, second_draws = evaluate_factories(
        opponent_factory,
        agent_factory,
        num_games=second_half,
        seed=seed + 1,
    )

    wins = first_wins + second_wins
    losses = first_losses + second_losses
    draws = first_draws + second_draws

    return {
        "games": num_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / num_games,
        "loss_rate": losses / num_games,
        "draw_rate": draws / num_games,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta experimentos del agente Group B y guarda resultados en CSV."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista los experimentos disponibles y sale.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Ids de experimentos a ejecutar. Si no se indica, corre todos.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=40,
        help="Cantidad de partidas de evaluacion por oponente.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=911,
        help="Semilla base para entrenamiento y evaluacion.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "experiment_results.csv",
        help="Ruta del CSV de salida.",
    )
    return parser.parse_args()


def summarize_phases(phases: list[dict[str, Any]]) -> dict[str, Any]:
    total_episodes = sum(int(phase["episodes"]) for phase in phases)
    phase_parts = []
    for index, phase in enumerate(phases, start=1):
        phase_parts.append(
            f"phase{index}:{phase['mode']}:{phase['episodes']}@eps={phase['epsilon']}"
        )

    return {
        "num_phases": len(phases),
        "total_episodes": total_episodes,
        "phase_plan": " | ".join(phase_parts),
    }


def train_experiment_agent(experiment: dict[str, Any], base_seed: int):
    model_path = ARTIFACTS_DIR / f"{experiment['experiment_id']}.pkl"
    agent = FVMCAgent(
        model_path=model_path,
        auto_train_if_missing=False,
        seed=base_seed,
    )

    for phase_index, phase in enumerate(experiment["train_phases"]):
        agent.train(
            num_episodes=int(phase["episodes"]),
            opponent_mode=str(phase["mode"]),
            epsilon=float(phase["epsilon"]),
        )

    agent.save_model(model_path)
    return agent, model_path


def build_study_opponents(agent) -> dict[str, Any]:
    return {
        "Random": lambda seed: RandomPolicy(seed=seed),
        "Self": frozen_factory_from_agent(agent),
    }


def evaluate_experiment(
    experiment: dict[str, Any],
    num_games: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    agent, model_path = train_experiment_agent(experiment, base_seed)
    agent_factory = frozen_factory_from_agent(agent)
    opponents = build_study_opponents(agent)
    phase_summary = summarize_phases(experiment["train_phases"])

    rows = []
    for offset, (opponent_name, opponent_factory) in enumerate(opponents.items()):
        metrics = evaluate_agent_balanced(
            agent_factory=agent_factory,
            opponent_factory=opponent_factory,
            num_games=num_games,
            seed=base_seed + (offset * 101),
        )
        rows.append(
            {
                "experiment_id": experiment["experiment_id"],
                "label": experiment["label"],
                "model_path": str(model_path),
                "opponent": opponent_name,
                "games": metrics["games"],
                "wins": metrics["wins"],
                "losses": metrics["losses"],
                "draws": metrics["draws"],
                "win_rate": metrics["win_rate"],
                "loss_rate": metrics["loss_rate"],
                "draw_rate": metrics["draw_rate"],
                "stored_states": len(agent.Q),
                "episodes_trained": agent.training_summary["episodes_trained"],
                "sources_random": agent.training_summary["sources"].get("random", 0),
                "sources_self": agent.training_summary["sources"].get("self", 0),
                **phase_summary,
            }
        )

    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "label",
        "model_path",
        "opponent",
        "games",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "loss_rate",
        "draw_rate",
        "stored_states",
        "episodes_trained",
        "sources_random",
        "sources_self",
        "num_phases",
        "total_episodes",
        "phase_plan",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.list:
        for experiment in DEFAULT_EXPERIMENTS:
            print(f"{experiment['experiment_id']}: {experiment['label']}")
        return

    selected = DEFAULT_EXPERIMENTS
    if args.only:
        selected_ids = set(args.only)
        selected = [
            experiment
            for experiment in DEFAULT_EXPERIMENTS
            if experiment["experiment_id"] in selected_ids
        ]
        missing = selected_ids.difference(
            {experiment["experiment_id"] for experiment in selected}
        )
        if missing:
            raise ValueError(f"Unknown experiment ids: {sorted(missing)}")

    all_rows = []

    for index, experiment in enumerate(selected):
        experiment_seed = args.seed + (index * 1000)
        rows = evaluate_experiment(
            experiment=experiment,
            num_games=args.games,
            base_seed=experiment_seed,
        )
        all_rows.extend(rows)

    write_csv(all_rows, args.output)
    print(f"Saved {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
