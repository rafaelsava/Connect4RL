import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4.connect_state import ConnectState
from groups.Rafa.policy import RafaRootUCBPolicy


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def choose_action(
    agent: RafaRootUCBPolicy,
    state: ConnectState,
    rng: np.random.Generator,
    epsilon: float,
    ucb_c: float,
    inner_rollouts: int,
    turn_time: float | None,
    selection: str,
) -> int:
    actions = [c for c in state.get_free_cols() if state.is_applicable(c)]
    if len(actions) == 1:
        return actions[0]

    key = agent._state_key(state.board, state.player)

    for action in actions:
        if agent._wins_immediately(state, state.player, action):
            return action

    opponent_state = ConnectState(board=state.board, player=-state.player)
    for action in actions:
        if agent._wins_immediately(opponent_state, -state.player, action):
            return action

    if selection == "epsilon_greedy":
        if rng.random() < epsilon:
            return int(rng.choice(actions))

        center_order = {3: 0, 2: 1, 4: 1, 1: 2, 5: 2, 0: 3, 6: 3}
        scored_actions = []
        for action in actions:
            q, visits = agent._lookup(key, action)
            prior = -0.01 * center_order[action] if visits == 0 else 0.0
            scored_actions.append((q + prior, action))

        best_score = max(score for score, _ in scored_actions)
        best_actions = [action for score, action in scored_actions if score == best_score]
        return int(rng.choice(best_actions))

    return choose_action_ucb_rollout(
        agent, state, actions, key, rng, ucb_c, inner_rollouts, turn_time
    )


def choose_action_ucb_rollout(
    agent: RafaRootUCBPolicy,
    state: ConnectState,
    actions: list[int],
    key: tuple[int, ...],
    rng: np.random.Generator,
    ucb_c: float,
    inner_rollouts: int,
    turn_time: float | None,
) -> int:
    q_local = {}
    n_local = {}
    for action in actions:
        q_local[action], n_local[action] = agent._global_prior(key, action)

    budget = max(len(actions), inner_rollouts)
    actual_rollouts = 0
    deadline = None if turn_time is None else time.perf_counter() + turn_time

    # Pull every arm without global prior once so UCB has a meaningful estimate.
    for action in actions:
        if n_local[action] > 0:
            continue
        result = agent._rollout(state.transition(action), state.player)
        q_local[action] = result
        n_local[action] = 1
        actual_rollouts += 1

    while True:
        if deadline is not None:
            if time.perf_counter() >= deadline:
                break
        elif actual_rollouts >= budget:
            break

        total = max(1, sum(n_local.values()))
        action = max(
            actions,
            key=lambda a: q_local[a]
            + ucb_c * math.sqrt(math.log(total + 1) / max(1, n_local[a])),
        )
        result = agent._rollout(state.transition(action), state.player)
        n_local[action] += 1
        q_local[action] += (result - q_local[action]) / n_local[action]
        actual_rollouts += 1

    return agent._select_robust_action(actions, q_local, n_local)


def play_outer_trial(
    agent: RafaRootUCBPolicy,
    rng: np.random.Generator,
    epsilon: float,
    ucb_c: float,
    inner_rollouts: int,
    budget_mode: str,
    training_total_time: float,
    max_turn_time: float,
    selection: str,
) -> tuple[int, list[tuple[int, tuple[int, ...], int]]]:
    state = ConnectState()
    trajectory: list[tuple[int, tuple[int, ...], int]] = []
    time_remaining = {-1: training_total_time, 1: training_total_time}

    while not state.is_final():
        player = state.player
        key = agent._state_key(state.board, player)

        turn_time = None
        if budget_mode == "time":
            total_pieces = int(np.sum(state.board != 0))
            own_turns_left = max(1, (agent.MAX_PIECES - total_pieces) // 2)
            turn_time = min(time_remaining[player] / own_turns_left, max_turn_time)

        turn_start = time.perf_counter()
        action = choose_action(
            agent,
            state,
            rng,
            epsilon,
            ucb_c,
            inner_rollouts,
            turn_time,
            selection,
        )
        if budget_mode == "time":
            time_remaining[player] = max(
                0.0, time_remaining[player] - (time.perf_counter() - turn_start)
            )

        trajectory.append((player, key, action))
        state = state.transition(action)

    return int(state.get_winner()), trajectory


def apply_global_backup(
    agent: RafaRootUCBPolicy,
    winner: int,
    trajectory: list[tuple[int, tuple[int, ...], int]],
    alpha: float,
    gamma: float,
) -> None:
    last_index = len(trajectory) - 1

    for index, (player, key, action) in enumerate(trajectory):
        if winner == 0:
            reward = 0.0
        elif winner == player:
            reward = 1.0
        else:
            reward = -1.0

        discounted_reward = reward * (gamma ** (last_index - index))
        agent._record_value(key, action, discounted_reward, alpha=alpha)


def train_one_game(
    agent: RafaRootUCBPolicy,
    rng: np.random.Generator,
    epsilon: float,
    ucb_c: float,
    inner_rollouts: int,
    budget_mode: str,
    training_total_time: float,
    max_turn_time: float,
    selection: str,
    alpha: float,
    gamma: float,
) -> int:
    winner, trajectory = play_outer_trial(
        agent,
        rng,
        epsilon,
        ucb_c,
        inner_rollouts,
        budget_mode,
        training_total_time,
        max_turn_time,
        selection,
    )
    apply_global_backup(agent, winner, trajectory, alpha, gamma)
    return int(winner)


def worker_play_outer_trial(
    task: tuple[int, float, float, int, str, float, float, str, str, int]
) -> tuple[int, list[tuple[int, tuple[int, ...], int]]]:
    (
        seed,
        epsilon,
        ucb_c,
        inner_rollouts,
        budget_mode,
        training_total_time,
        max_turn_time,
        selection,
        qtable_path,
        global_prior_visits,
    ) = task
    rng = np.random.default_rng(seed)
    agent = RafaRootUCBPolicy(
        qtable_path=qtable_path,
        auto_save=False,
        global_prior_visits=global_prior_visits,
    )
    return play_outer_trial(
        agent,
        rng,
        epsilon,
        ucb_c,
        inner_rollouts,
        budget_mode,
        training_total_time,
        max_turn_time,
        selection,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena la politica Rafa con self-play y guarda Q-values globales."
    )
    parser.add_argument("--games", type=int, default=1_000, help="Partidas de self-play.")
    parser.add_argument("--epsilon", type=float, default=0.15, help="Exploracion para --selection epsilon_greedy.")
    parser.add_argument("--ucb-c", type=float, default=1.2, help="Constante de exploracion UCB.")
    parser.add_argument(
        "--inner-rollouts",
        type=int,
        default=24,
        help="Rollouts locales por estado si --budget-mode rollouts.",
    )
    parser.add_argument(
        "--budget-mode",
        choices=["rollouts", "time"],
        default="rollouts",
        help="Presupuesto del subproceso local: fijo por rollouts o por tiempo.",
    )
    parser.add_argument(
        "--training-total-time",
        type=float,
        default=60.0,
        help="Segundos por jugador y partida si --budget-mode time.",
    )
    parser.add_argument(
        "--max-turn-time",
        type=float,
        default=5.0,
        help="Tope de segundos por turno si --budget-mode time.",
    )
    parser.add_argument(
        "--selection",
        choices=["ucb_rollout", "epsilon_greedy"],
        default="ucb_rollout",
        help="Como escoger acciones durante self-play.",
    )
    parser.add_argument("--alpha", type=float, default=0.08, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.995, help="Descuento por jugada.")
    parser.add_argument("--seed", type=int, default=None, help="Semilla opcional.")
    parser.add_argument("--save-every", type=int, default=100, help="Guardar cada N partidas.")
    parser.add_argument("--log-every", type=int, default=25, help="Mostrar progreso cada N partidas.")
    parser.add_argument(
        "--global-prior-visits",
        type=int,
        default=10,
        help="Peso maximo de visitas heredadas desde la Q-table global hacia q' local.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Procesos paralelos para generar outer trials. 1 = secuencial.",
    )
    parser.add_argument(
        "--qtable",
        type=Path,
        default=RafaRootUCBPolicy.DEFAULT_QTABLE_PATH,
        help="Ruta del archivo .pkl de Q-values.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    agent = RafaRootUCBPolicy(
        qtable_path=args.qtable,
        auto_save=False,
        global_prior_visits=args.global_prior_visits,
    )
    recent = deque(maxlen=max(1, min(args.save_every, 500)))
    totals = {-1: 0, 0: 0, 1: 0}
    start_time = time.perf_counter()
    last_log_time = start_time
    last_log_game = 0
    last_q_states = len(agent.q_table)

    print(f"Entrenando Rafa por self-play: {args.games} partidas")
    print(f"Q-table: {agent.qtable_path}")
    print(
        "Seleccion outer: "
        f"{args.selection} | budget_mode={args.budget_mode} | "
        f"inner_rollouts={args.inner_rollouts} | ucb_c={args.ucb_c} | "
        f"global_prior_visits={args.global_prior_visits}"
    )
    print(f"Workers: {args.workers}")
    print("Ctrl+C guarda progreso antes de salir.\n")

    def handle_completed_game(
        game: int,
        winner: int,
        trajectory: list[tuple[int, tuple[int, ...], int]] | None = None,
    ) -> None:
        nonlocal last_log_time, last_log_game, last_q_states

        if trajectory is not None:
            apply_global_backup(agent, winner, trajectory, args.alpha, args.gamma)

        totals[winner] += 1
        recent.append(winner)

        should_save = game % args.save_every == 0 or game == args.games
        should_log = game % args.log_every == 0 or should_save or game == args.games

        if should_save:
            agent.save_q_values(force=True)

        if should_log:
            recent_red = sum(1 for w in recent if w == -1)
            recent_yellow = sum(1 for w in recent if w == 1)
            recent_draws = sum(1 for w in recent if w == 0)
            now = time.perf_counter()
            elapsed = now - start_time
            total_rate = game / max(elapsed, 1e-9)
            interval_rate = (game - last_log_game) / max(now - last_log_time, 1e-9)
            eta = (args.games - game) / max(total_rate, 1e-9)
            q_states = len(agent.q_table)
            q_entries = sum(len(actions) for actions in agent.q_table.values())
            new_states = q_states - last_q_states
            save_marker = " | guardado" if should_save else ""
            print(
                f"{game:>6}/{args.games} ({game / args.games:>6.1%}) | "
                f"{interval_rate:>6.1f} g/s ahora, {total_rate:>6.1f} g/s prom | "
                f"ETA {format_duration(eta)} | "
                f"recientes R:{recent_red} A:{recent_yellow} E:{recent_draws} | "
                f"estados:{q_states} (+{new_states}) pares:{q_entries}"
                f"{save_marker}"
            )
            last_log_time = now
            last_log_game = game
            last_q_states = q_states

    seed_rng = np.random.default_rng(args.seed)

    def make_worker_task() -> tuple[int, float, float, int, str, float, float, str, str, int]:
        return (
            int(seed_rng.integers(0, np.iinfo(np.uint32).max)),
            args.epsilon,
            args.ucb_c,
            args.inner_rollouts,
            args.budget_mode,
            args.training_total_time,
            args.max_turn_time,
            args.selection,
            str(agent.qtable_path),
            args.global_prior_visits,
        )

    try:
        if args.workers <= 1:
            for game in range(1, args.games + 1):
                winner = train_one_game(
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
                handle_completed_game(game, winner)
        else:
            submitted = 0
            completed = 0
            max_pending = max(1, args.workers * 2)
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                pending = set()
                while submitted < args.games and len(pending) < max_pending:
                    pending.add(executor.submit(worker_play_outer_trial, make_worker_task()))
                    submitted += 1

                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        winner, trajectory = future.result()
                        completed += 1
                        handle_completed_game(completed, winner, trajectory)

                        while submitted < args.games and len(pending) < max_pending:
                            pending.add(
                                executor.submit(worker_play_outer_trial, make_worker_task())
                            )
                            submitted += 1
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido. Guardando progreso...")
    finally:
        agent.save_q_values(force=True)
        print(
            "Totales: "
            f"Rojo={totals[-1]}, Amarillo={totals[1]}, Empates={totals[0]}"
        )
        print(f"Q-values guardados en: {agent.qtable_path}")


if __name__ == "__main__":
    main()
