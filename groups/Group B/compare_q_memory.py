import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

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


def evaluate_agent(agent, num_games: int, seed: int):
    agent_factory = frozen_factory_from_agent(agent)
    opponents = {
        "Random": lambda opp_seed: RandomPolicy(seed=opp_seed),
        "Self": frozen_factory_from_agent(agent),
    }

    rows = []
    for offset, (opponent_name, opponent_factory) in enumerate(opponents.items()):
        metrics = evaluate_agent_balanced(
            agent_factory=agent_factory,
            opponent_factory=opponent_factory,
            num_games=num_games,
            seed=seed + (offset * 101),
        )
        rows.append((opponent_name, metrics))

    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compara entrenamiento desde cero vs incremental guardando Q-values."
    )
    parser.add_argument(
        "--milestones",
        nargs="+",
        type=int,
        default=[1000, 3500, 5000, 10000],
        help="Puntos acumulados de episodios para comparar.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Cantidad de partidas de evaluacion por oponente.",
    )
    parser.add_argument(
        "--mode",
        choices=("random", "self"),
        default="random",
        help="Rival usado durante el entrenamiento de la comparacion.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.20,
        help="Epsilon fijo para esta comparacion.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "q_memory_comparison.csv",
        help="CSV de salida.",
    )
    return parser.parse_args()


def build_rows_for_strategy(strategy_name: str, milestones: list[int], num_games: int, mode: str, epsilon: float):
    rows = []
    base_seed = 7000 if strategy_name == "from_scratch" else 9000

    if strategy_name == "from_scratch":
        for index, milestone in enumerate(milestones):
            agent = FVMCAgent(auto_train_if_missing=False, seed=base_seed + index)
            agent.train(milestone, opponent_mode=mode, epsilon=epsilon)

            for opponent_name, metrics in evaluate_agent(agent, num_games=num_games, seed=base_seed + index * 17):
                rows.append(
                    {
                        "strategy": strategy_name,
                        "episodes": milestone,
                        "train_mode": mode,
                        "train_epsilon": epsilon,
                        "opponent": opponent_name,
                        "games": metrics["games"],
                        "wins": metrics["wins"],
                        "losses": metrics["losses"],
                        "draws": metrics["draws"],
                        "win_rate": metrics["win_rate"],
                        "loss_rate": metrics["loss_rate"],
                        "draw_rate": metrics["draw_rate"],
                        "stored_states": len(agent.Q),
                    }
                )
        return rows

    model_path = ARTIFACTS_DIR / "incremental_q_memory.pkl"
    if model_path.exists():
        model_path.unlink()

    agent = FVMCAgent(model_path=model_path, auto_train_if_missing=False, seed=base_seed)
    previous_milestone = 0

    for index, milestone in enumerate(milestones):
        delta = milestone - previous_milestone
        if delta <= 0:
            raise ValueError("Milestones must be strictly increasing.")

        agent.train(delta, opponent_mode=mode, epsilon=epsilon)
        agent.save_model(model_path)

        reloaded = FVMCAgent(model_path=model_path, auto_train_if_missing=False, seed=base_seed + index + 1)
        reloaded.load_model(model_path)

        for opponent_name, metrics in evaluate_agent(reloaded, num_games=num_games, seed=base_seed + index * 17):
            rows.append(
                {
                    "strategy": strategy_name,
                    "episodes": milestone,
                    "train_mode": mode,
                    "train_epsilon": epsilon,
                    "opponent": opponent_name,
                    "games": metrics["games"],
                    "wins": metrics["wins"],
                    "losses": metrics["losses"],
                    "draws": metrics["draws"],
                    "win_rate": metrics["win_rate"],
                    "loss_rate": metrics["loss_rate"],
                    "draw_rate": metrics["draw_rate"],
                    "stored_states": len(reloaded.Q),
                }
            )

        agent = reloaded
        previous_milestone = milestone

    return rows


def write_csv(rows, output_path: Path):
    fieldnames = [
        "strategy",
        "episodes",
        "train_mode",
        "train_epsilon",
        "opponent",
        "games",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "loss_rate",
        "draw_rate",
        "stored_states",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    milestones = sorted(args.milestones)

    rows = []
    rows.extend(
        build_rows_for_strategy(
            "from_scratch",
            milestones=milestones,
            num_games=args.games,
            mode=args.mode,
            epsilon=args.epsilon,
        )
    )
    rows.extend(
        build_rows_for_strategy(
            "incremental_q_memory",
            milestones=milestones,
            num_games=args.games,
            mode=args.mode,
            epsilon=args.epsilon,
        )
    )

    write_csv(rows, args.output)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
