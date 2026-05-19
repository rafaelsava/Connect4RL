import math
import time
import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState


class RafaRootUCBPolicy(Policy):
    """
    Root UCB policy para Connect-4.

    En cada turno trata las acciones legales del estado actual como brazos
    de un bandit y reparte simulaciones con UCB hasta agotar el tiempo
    asignado. El tiempo total se distribuye proporcionalmente entre los
    turnos restantes estimados de la partida.
    """

    # Connect-4: 6 filas x 7 cols = 42 casillas maximo
    MAX_PIECES = 42

    def __init__(self, total_time: float = 60.0, exploration_c: float = 1.0):
        self.total_time = total_time
        self.exploration_c = exploration_c
        self.rng = np.random.default_rng()
        self.time_remaining = self.total_time

    def mount(self, total_time: float = None) -> None:
        if total_time is not None:
            self.total_time = total_time
        self.rng = np.random.default_rng()
        self.time_remaining = self.total_time

    def _wins_immediately(self, state: ConnectState, player: int, col: int) -> bool:
        if not state.is_applicable(col):
            return False
        return state.transition(col).get_winner() == player

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

        q_local = {a: 0.0 for a in actions}
        n_local = {a: 0 for a in actions}

        turn_start = time.perf_counter()

        # Inicializacion: una simulacion por accion para que UCB sea valido
        for a in actions:
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

        return max(actions, key=lambda a: q_local[a])

    def _rollout(self, state: ConnectState, my_player: int) -> float:
        """
        Inner trial: simula la partida hasta el final con politica aleatoria.
        Devuelve 1.0 (gana), 0.0 (empate) o -1.0 (pierde) para my_player.
        """
        while not state.is_final():
            cols = state.get_free_cols()
            col = int(self.rng.choice(cols))
            state = state.transition(col)

        winner = state.get_winner()
        if winner == my_player:
            return 1.0
        elif winner == 0:
            return 0.0
        else:
            return -1.0
