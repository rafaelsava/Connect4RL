import math
import atexit
import pickle
import time
from pathlib import Path
import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState


class RafaRootUCBPolicy:
    """
    Politica final de Rafa: TBOPI con Q-table global como prior.

    En cada turno corre un subproceso local q' con UCB desde el estado actual.
    La Q-table entrenada por self-play inicializa ese proceso local; luego los
    rollouts heuristicos refinan la decision y se elige la accion robusta con
    mas visitas locales.
    """

    # Connect-4: 6 filas x 7 cols = 42 casillas maximo
    MAX_PIECES = 42

    DEFAULT_QTABLE_PATH = Path(__file__).with_name("rafa_q_values.pkl")
    _shared_tables: dict[Path, dict[tuple[int, ...], dict[int, list[float | int]]]] = {}
    _dirty_updates: dict[Path, int] = {}
    _atexit_registered = False

    def __init__(
        self,
        total_time: float = 60.0,
        exploration_c: float = 1.0,
        qtable_path: str | Path | None = None,
        auto_save: bool = True,
        save_every_updates: int = 2_000,
        global_prior_visits: int = 10,
        rollout_epsilon: float = 0.15,
        rollout_policy: str = "heuristic",
        final_selection: str = "robust",
    ):
        self.total_time = total_time
        self.exploration_c = exploration_c
        self.qtable_path = Path(qtable_path or self.DEFAULT_QTABLE_PATH).resolve()
        self.auto_save = auto_save
        self.save_every_updates = save_every_updates
        self.global_prior_visits = global_prior_visits
        self.rollout_epsilon = rollout_epsilon
        self.rollout_policy = rollout_policy
        self.final_selection = final_selection
        self.q_table = self._load_q_table(self.qtable_path)
        self.rng = np.random.default_rng()
        self.time_remaining = self.total_time

        if not self.__class__._atexit_registered:
            atexit.register(self.__class__._save_all_dirty)
            self.__class__._atexit_registered = True

    def mount(self, total_time: float = None) -> None:
        if total_time is not None:
            self.total_time = total_time
        self.rng = np.random.default_rng()
        self.time_remaining = self.total_time

    @classmethod
    def _load_q_table(
        cls, path: Path
    ) -> dict[tuple[int, ...], dict[int, list[float | int]]]:
        if path in cls._shared_tables:
            return cls._shared_tables[path]

        if not path.exists():
            cls._shared_tables[path] = {}
            cls._dirty_updates[path] = 0
            return cls._shared_tables[path]

        with path.open("rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict) and "q_table" in payload:
            q_table = payload["q_table"]
        else:
            q_table = payload

        cls._shared_tables[path] = q_table
        cls._dirty_updates[path] = 0
        return q_table

    @classmethod
    def _save_path(cls, path: Path, force: bool = False) -> None:
        dirty = cls._dirty_updates.get(path, 0)
        if not force and dirty <= 0:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "version": 1,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "q_table": cls._shared_tables.get(path, {}),
        }
        with tmp_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)
        cls._dirty_updates[path] = 0

    @classmethod
    def _save_all_dirty(cls) -> None:
        for path, dirty in list(cls._dirty_updates.items()):
            if dirty > 0:
                cls._save_path(path, force=True)

    def save_q_values(self, force: bool = True) -> None:
        self.__class__._save_path(self.qtable_path, force=force)

    def _maybe_save_q_values(self) -> None:
        if (
            self.auto_save
            and self.__class__._dirty_updates.get(self.qtable_path, 0)
            >= self.save_every_updates
        ):
            self.save_q_values(force=True)

    def _state_key(self, board: np.ndarray, player: int) -> tuple[int, ...]:
        # Canonical form: "1" is always the player to move, "-1" the opponent.
        return tuple((board * player).astype(np.int8).ravel().tolist())

    def _lookup(self, key: tuple[int, ...], action: int) -> tuple[float, int]:
        q, n = self.q_table.get(key, {}).get(action, [0.0, 0])
        return float(q), int(n)

    def _global_prior(self, key: tuple[int, ...], action: int) -> tuple[float, int]:
        q, visits = self._lookup(key, action)
        if visits <= 0 or self.global_prior_visits <= 0:
            return 0.0, 0
        return q, min(visits, self.global_prior_visits)

    def _record_value(
        self,
        key: tuple[int, ...],
        action: int,
        value: float,
        alpha: float | None = None,
    ) -> None:
        row = self.q_table.setdefault(key, {})
        q, n = self._lookup(key, action)
        n += 1
        if alpha is None:
            q += (value - q) / n
        else:
            q += alpha * (value - q)
        row[int(action)] = [float(q), int(n)]
        self.__class__._dirty_updates[self.qtable_path] = (
            self.__class__._dirty_updates.get(self.qtable_path, 0) + 1
        )

    def _wins_immediately(self, state: ConnectState, player: int, col: int) -> bool:
        if not state.is_applicable(col):
            return False
        return state.transition(col).get_winner() == player

    def _actions_without_immediate_loss(self, state: ConnectState) -> list[int]:
        actions = [c for c in state.get_free_cols() if state.is_applicable(c)]
        safe_actions = []
        for action in actions:
            next_state = state.transition(action)
            if next_state.is_final():
                safe_actions.append(action)
                continue

            opponent = next_state.player
            if not any(
                self._wins_immediately(next_state, opponent, reply)
                for reply in next_state.get_free_cols()
            ):
                safe_actions.append(action)
        return safe_actions if safe_actions else actions

    def _rollout_action(self, state: ConnectState) -> int:
        actions = [c for c in state.get_free_cols() if state.is_applicable(c)]
        if len(actions) == 1:
            return actions[0]

        player = state.player
        for action in actions:
            if self._wins_immediately(state, player, action):
                return action

        opponent_state = ConnectState(board=state.board, player=-player)
        for action in actions:
            if self._wins_immediately(opponent_state, -player, action):
                return action

        safe_actions = self._actions_without_immediate_loss(state)
        if self.rng.random() < self.rollout_epsilon:
            return int(self.rng.choice(safe_actions))

        center_order = {3: 0, 2: 1, 4: 1, 1: 2, 5: 2, 0: 3, 6: 3}
        best_distance = min(center_order[action] for action in safe_actions)
        best_actions = [
            action for action in safe_actions if center_order[action] == best_distance
        ]
        return int(self.rng.choice(best_actions))

    def _select_robust_action(
        self,
        actions: list[int],
        q_local: dict[int, float],
        n_local: dict[int, int],
    ) -> int:
        if self.final_selection == "proportional":
            visits = np.array([n_local[a] for a in actions], dtype=float)
            probs = visits / visits.sum()
            return int(self.rng.choice(actions, p=probs))
        return max(actions, key=lambda a: (n_local[a], q_local[a]))

    def act(self, s: np.ndarray) -> int:
        my_player = -1 if np.sum(s == -1) == np.sum(s == 1) else 1
        state = ConnectState(board=s, player=my_player)

        # Si el estado ya es terminal, no hay jugada valida que hacer
        if state.is_final():
            free = state.get_free_cols()
            return free[0] if free else 0

        actions = [c for c in state.get_free_cols() if state.is_applicable(c)]
        if not actions:
            return 0
        if len(actions) == 1:
            return actions[0]

        # 1. Ganar inmediatamente si es posible
        for a in actions:
            if self._wins_immediately(state, my_player, a):
                return a

        # 2. Bloquear victoria inmediata del oponente
        opp = -my_player
        opp_state = ConnectState(board=s, player=opp)
        for a in actions:
            if self._wins_immediately(opp_state, opp, a):
                return a

        # Estima cuantos turnos propios quedan para repartir el tiempo restante.
        total_pieces = int(np.sum(s != 0))
        our_turns_left = max(1, (self.MAX_PIECES - total_pieces) // 2)
        time_for_turn = min(self.time_remaining / our_turns_left, 5.0)

        state_key = self._state_key(s, my_player)
        q_local = {}
        n_local = {}
        for a in actions:
            q_local[a], n_local[a] = self._global_prior(state_key, a)

        turn_start = time.perf_counter()

        # q' local: se usa solo para decidir en este estado, no se persiste.
        for a in actions:
            if n_local[a] > 0:
                continue
            try:
                q_local[a] = self._rollout(state.transition(a), my_player)
            except ValueError:
                q_local[a] = 0.0
            n_local[a] = 1

        # Bucle UCB hasta agotar el tiempo asignado a este turno
        deadline = turn_start + time_for_turn
        while time.perf_counter() < deadline:
            total = sum(n_local.values())
            a = max(
                actions,
                key=lambda a: q_local[a]
                + self.exploration_c * math.sqrt(math.log(total) / n_local[a]),
            )
            try:
                result = self._rollout(state.transition(a), my_player)
            except ValueError:
                result = 0.0
            n_local[a] += 1
            q_local[a] += (result - q_local[a]) / n_local[a]

        self.time_remaining -= time.perf_counter() - turn_start
        return self._select_robust_action(actions, q_local, n_local)

    def _rollout(self, state: ConnectState, my_player: int) -> float:
        """
        Inner trial: simula la partida hasta el final con politica heuristica.
        Devuelve 1.0 (gana), 0.0 (empate) o -1.0 (pierde) para my_player.
        """
        while not state.is_final():
            if self.rollout_policy == "random":
                col = int(self.rng.choice(state.get_free_cols()))
            else:
                col = self._rollout_action(state)
            state = state.transition(col)

        winner = state.get_winner()
        if winner == my_player:
            return 1.0
        elif winner == 0:
            return 0.0
        else:
            return -1.0


class RafaNoMemoryPolicy(RafaRootUCBPolicy):
    """
    Misma politica online UCB de Rafa, pero sin usar Q-values globales como prior.
    Util para demos y ablations contra RafaRootUCBPolicy.
    """

    def __init__(
        self,
        total_time: float = 60.0,
        exploration_c: float = 1.0,
        **_: object,
    ):
        super().__init__(
            total_time=total_time,
            exploration_c=exploration_c,
            auto_save=False,
            global_prior_visits=0,
        )


class RafaBasePolicy(RafaRootUCBPolicy):
    """
    Version base: Root UCB con rollouts aleatorios, sin Q-table global.
    """

    def __init__(
        self,
        total_time: float = 60.0,
        exploration_c: float = 1.0,
        **_: object,
    ):
        super().__init__(
            total_time=total_time,
            exploration_c=exploration_c,
            auto_save=False,
            global_prior_visits=0,
            rollout_policy="random",
            final_selection="proportional",
        )


class RafaQPolicy(RafaRootUCBPolicy):
    """
    Version intermedia: misma base, pero usando Q-table global como prior.
    """

    def __init__(
        self,
        total_time: float = 60.0,
        exploration_c: float = 1.0,
        qtable_path: str | Path | None = None,
        **_: object,
    ):
        super().__init__(
            total_time=total_time,
            exploration_c=exploration_c,
            qtable_path=qtable_path,
            auto_save=False,
            global_prior_visits=5,
            rollout_policy="random",
            final_selection="proportional",
        )


class RafaImprovedPolicy(RafaRootUCBPolicy, Policy):
    """
    Alias explicito de la politica final para scripts de analisis.
    """

    pass
