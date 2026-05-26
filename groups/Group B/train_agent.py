import argparse
import importlib.util
from pathlib import Path
from pprint import pformat
import sys


GROUP_DIR = Path(__file__).resolve().parent
ROOT_DIR = GROUP_DIR.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ARTIFACTS_DIR = GROUP_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_policy_module():
    policy_path = GROUP_DIR / "policy.py"
    spec = importlib.util.spec_from_file_location("group_b_policy", policy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {policy_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continua el entrenamiento del agente Group B desde el ultimo .pkl guardado."
    )
    parser.add_argument(
        "--preset",
        choices=("random_bootstrap", "self_only", "self_heavy", "self_refine"),
        default=None,
        help="Plan de entrenamiento predefinido. Si se usa, ignora --mode/--episodes/--epsilon.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Cantidad de episodios extra a entrenar.",
    )
    parser.add_argument(
        "--mode",
        choices=("random", "self"),
        default="random",
        help="Tipo de rival usado en los episodios extra.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Exploracion epsilon para esta corrida. Si no se indica, usa el valor por defecto del agente.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    policy_module = load_policy_module()
    model_path = GROUP_DIR / "fvmc_model.pkl"
    agent = policy_module.FVMCAgent(
        model_path=model_path,
        auto_train_if_missing=False,
    )

    loaded_existing_model = False
    if model_path.exists():
        agent.load_model(model_path)
        loaded_existing_model = True

    run_config = None
    if args.preset is not None:
        phases = policy_module.FVMCAgent.get_training_preset(args.preset)
        agent.train_with_plan(phases)
        run_config = {
            "preset": args.preset,
            "phases": phases,
        }
    else:
        epsilon = agent.EPSILON if args.epsilon is None else args.epsilon
        agent.train(
            args.episodes,
            opponent_mode=args.mode,
            epsilon=epsilon,
        )
        run_config = {
            "mode": args.mode,
            "episodes": args.episodes,
            "epsilon": epsilon,
        }
    agent.save_model()

    summary = {
        "model_path": str(model_path),
        "loaded_existing_model": loaded_existing_model,
        "run": run_config,
        "stored_states": len(agent.Q),
        "training_summary": agent.training_summary,
    }

    summary_text = pformat(summary, sort_dicts=False)
    summary_path = ARTIFACTS_DIR / "training_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
