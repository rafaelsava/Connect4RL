import math
import time
import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
from typing import override


class RafaTBOPI(Policy):
    """
    Trial-Based Online Policy Improvement para Connect-4.

    En cada turno corre simulaciones UCB hasta agotar el tiempo asignado
    para ese turno. El tiempo total se distribuye proporcionalmente entre
    los turnos restantes estimados de la partida.
    El árbol y los q̂' se descartan al terminar el turno (sin memoria entre turnos).
    """

    # Connect-4: 6 filas x 7 cols = 42 casillas máximo
    MAX_PIECES = 42

    def __init__(self, total_time: float = 60.0, exploration_c: float = 1.0):
        self.total_time = total_time        # segundos para toda la partida
        self.exploration_c = exploration_c  # constante C del UCB

    @override
    def mount(self) -> None:
        self.rng = np.random.default_rng()
        self.time_remaining = self.total_time  # se reinicia al montar el agente

    @override
    def act(self, s: np.ndarray) -> int:
        # Reconstruye el jugador actual: si hay igual cantidad de fichas de
        # ambos colores, le toca a -1 (el primero); si -1 tiene una más, a 1.
        my_player = -1 if np.sum(s == -1) == np.sum(s == 1) else 1
        state = ConnectState(board=s, player=my_player)

        actions = state.get_free_cols()
        if len(actions) == 1:
            return actions[0]

        # Estima cuántos turnos propios quedan para repartir el tiempo restante.
        # Hay (MAX_PIECES - fichas_en_tablero) movimientos en total para ambos;
        # la mitad aproximadamente son nuestros.
        total_pieces = int(np.sum(s != 0))
        our_turns_left = max(1, (self.MAX_PIECES - total_pieces) // 2)
        time_for_turn = self.time_remaining / our_turns_left

        # q̂' local y contadores — se descartan al terminar este turno
        q_local = {a: 0.0 for a in actions}
        n_local = {a: 0   for a in actions}

        turn_start = time.perf_counter()

        # Inicialización: una simulación por acción para que UCB sea válido
        for a in actions:
            q_local[a] = self._rollout(state.transition(a), my_player)
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
            result = self._rollout(state.transition(a), my_player)
            n_local[a] += 1
            q_local[a] += (result - q_local[a]) / n_local[a]

        # Descuenta el tiempo real consumido (inicialización + bucle UCB)
        self.time_remaining -= time.perf_counter() - turn_start

        return max(actions, key=lambda a: q_local[a])

    def _rollout(self, state: ConnectState, my_player: int) -> float:
        """
        Inner trial: simula la partida hasta el final con política aleatoria.
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
