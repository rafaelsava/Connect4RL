# AGENTE DE CONNECT 4 BASADO EN FIRST-VISIT MONTE CARLO

from collections import defaultdict

import numpy as np

from connect4.policy import Policy


class FVMCAgent(Policy):
    """
    Agente de Connect 4 basado en First-Visit Monte Carlo.

    Idea general:
    - Durante `mount()` juega miles de partidas contra un rival aleatorio.
    - En esas partidas estima el valor esperado de jugar cierta columna en
      cierto estado.
    - Durante `act()` primero revisa tacticas inmediatas (ganar o bloquear)
      y, si no encuentra una, usa el valor aprendido.
    """

    NUM_EPISODES = 3500
    GAMMA = 0.95
    EPSILON = 0.2

    def __init__(self):
        # Q[(estado, columna)] guarda el retorno promedio aprendido.
        self.Q = defaultdict(float)
        # N[(estado, columna)] cuenta cuantas veces vimos ese par.
        self.N = defaultdict(int)

    def _key(self, board):
        # Convierte el tablero en una tupla para poder usarlo como clave.
        return tuple(board.flatten())

    def _cols(self, board):
        # Una columna esta disponible si la casilla superior esta vacia.
        return [c for c in range(7) if board[0, c] == 0]

    def _drop(self, board, col, player):
        # Simula poner una ficha en `col` sin modificar el tablero original.
        b = board.copy()
        for r in range(5, -1, -1):
            if b[r, col] == 0:
                b[r, col] = player
                break
        return b

    def _wins(self, board, player):
        # Revisa lineas horizontales de 4.
        for r in range(6):
            for c in range(4):
                if all(board[r, c + i] == player for i in range(4)):
                    return True

        # Revisa lineas verticales de 4.
        for r in range(3):
            for c in range(7):
                if all(board[r + i, c] == player for i in range(4)):
                    return True

        # Revisa diagonales descendentes hacia la derecha.
        for r in range(3):
            for c in range(4):
                if all(board[r + i, c + i] == player for i in range(4)):
                    return True

        # Revisa diagonales descendentes hacia la izquierda.
        for r in range(3):
            for c in range(3, 7):
                if all(board[r + i, c - i] == player for i in range(4)):
                    return True

        return False

    def _whose_turn(self, board):
        count = np.sum(board != 0)
        return -1 if count % 2 == 0 else 1

    def mount(self, timeout=None):
        # Entrenamos alternando colores para aprender ambos lados del tablero.
        for episode in range(self.NUM_EPISODES):
            me = -1 if episode % 2 == 0 else 1
            opp = -me

            # Partimos de un tablero vacio en cada episodio.
            board = np.zeros((6, 7), dtype=int)
            # Guarda la trayectoria del agente: (estado, accion, recompensa).
            trace = []
            # En Connect 4 siempre empieza el jugador -1.
            turn = -1

            while True:
                cols = self._cols(board)
                if not cols:
                    # Si no hay columnas libres y nadie gano, es empate.
                    break

                if turn == me:
                    key = self._key(board)

                    # Politica epsilon-greedy:
                    # - con probabilidad EPSILON explora una accion aleatoria
                    # - si no, explota la mejor accion segun Q
                    if np.random.random() < self.EPSILON:
                        action = int(np.random.choice(cols))
                    else:
                        action = max(cols, key=lambda c: self.Q[(key, c)])

                    board = self._drop(board, action, me)

                    if self._wins(board, me):
                        # Victoria inmediata del agente en este estado.
                        trace.append((key, action, 1.0))
                        break

                    # Si no gano aun, deja recompensa 0 provisional.
                    trace.append((key, action, 0.0))
                else:
                    # El oponente del entrenamiento juega aleatorio.
                    action = int(np.random.choice(cols))
                    board = self._drop(board, action, opp)

                    if self._wins(board, opp) and trace:
                        # Si el rival gana, castigamos la ultima decision
                        # tomada por el agente dentro de esta partida.
                        k, a, _ = trace[-1]
                        trace[-1] = (k, a, -1.0)
                        break

                # Cambiamos el turno para continuar la simulacion.
                turn = -turn

            # Actualizacion First-Visit Monte Carlo:
            # recorremos la partida al reves acumulando el retorno descontado.
            G = 0.0
            seen = set()
            for key, action, reward in reversed(trace):
                G = reward + self.GAMMA * G

                # Solo actualizamos la primera visita del par (estado, accion)
                # dentro del episodio actual.
                if (key, action) not in seen:
                    seen.add((key, action))
                    self.N[(key, action)] += 1
                    self.Q[(key, action)] += (
                        G - self.Q[(key, action)]
                    ) / self.N[(key, action)]

    def act(self, s: np.ndarray) -> int:
        # Columnas legales desde el estado actual.
        cols = self._cols(s)
        # El agente deduce quien debe jugar mirando el tablero actual.
        me = self._whose_turn(s)
        opp = -me

        # 1. Si existe una jugada ganadora inmediata, la toma.
        for c in cols:
            if self._wins(self._drop(s, c, me), me):
                return c

        # 2. Si el rival tiene victoria inmediata, la bloquea.
        for c in cols:
            if self._wins(self._drop(s, c, opp), opp):
                return c

        # 3. Si no hay tactica inmediata, usa la mejor accion segun  el Q.
        key = self._key(s)
        return max(cols, key=lambda c: self.Q[(key, c)])
