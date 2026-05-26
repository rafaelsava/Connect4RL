import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from connect4.policy import Policy


class FVMCAgent(Policy):
    # AGENTE DE CONNECT 4 BASADO EN FIRST-VISIT MONTE CARLO

    NUM_EPISODES = 5000
    GAMMA = 0.95
    EPSILON = 0.2
    MODEL_VERSION = 2
    DEFAULT_AUTO_TRAIN_PLAN = [
        {"mode": "random", "episodes": 8000, "epsilon": 0.20},
        {"mode": "self", "episodes": 12000, "epsilon": 0.10},
        {"mode": "self", "episodes": 8000, "epsilon": 0.05},
    ]
    TRAINING_PRESETS = {
        "random_bootstrap": [
            {"mode": "random", "episodes": 10000, "epsilon": 0.20},
        ],
        "self_only": [
            {"mode": "self", "episodes": 20000, "epsilon": 0.15},
            {"mode": "self", "episodes": 10000, "epsilon": 0.05},
        ],
        "self_heavy": [
            {"mode": "random", "episodes": 8000, "epsilon": 0.20},
            {"mode": "self", "episodes": 12000, "epsilon": 0.10},
            {"mode": "self", "episodes": 8000, "epsilon": 0.05},
        ],
        "self_refine": [
            {"mode": "self", "episodes": 10000, "epsilon": 0.08},
            {"mode": "self", "episodes": 5000, "epsilon": 0.03},
        ],
    }

    def __init__(
        self,
        model_path: str | Path | None = None,
        auto_train_if_missing: bool = True,
        seed: int = 911,
    ):
        # Q[(estado, columna)] guarda el retorno promedio aprendido.
        self.Q = defaultdict(float)
        # N[(estado, columna)] cuenta cuantas veces vimos ese par.
        self.N = defaultdict(int)
        self.rng = np.random.default_rng(seed)
        self.model_path = (
            Path(model_path) if model_path else Path(__file__).with_name("fvmc_model.pkl")
        )
        self.auto_train_if_missing = auto_train_if_missing
        self._is_ready = False
        self.training_summary = {
            "episodes_trained": 0,
            "sources": {},
        }

    def _canonical_key(self, board, player=None):
        # Normaliza el tablero desde la perspectiva del jugador que mueve.
        # Asi reutilizamos conocimiento para ambos colores y el self-play
        # consulta la misma "idea de estado" sin mezclar roles.
        actor = self._whose_turn(board) if player is None else player
        return tuple(int(x) for x in (board * actor).flatten())

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

    def _best_action(self, board, player=None):
        cols = self._cols(board)
        key = self._canonical_key(board, player)
        return max(cols, key=lambda c: (self.Q[(key, c)], -abs(c - 3)))

    def _play_training_episode(self, opponent_mode="random", epsilon=None):
        epsilon = self.EPSILON if epsilon is None else epsilon

        # Entrenamos alternando colores para aprender ambos lados del tablero.
        me = -1 if self.training_summary["episodes_trained"] % 2 == 0 else 1
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
                key = self._canonical_key(board, me)

                # Politica epsilon-greedy:
                # - con probabilidad EPSILON explora una accion aleatoria
                # - si no, explota la mejor accion segun Q
                if self.rng.random() < epsilon:
                    action = int(self.rng.choice(cols))
                else:
                    action = self._best_action(board, me)

                board = self._drop(board, action, me)

                if self._wins(board, me):
                    # Victoria inmediata del agente en este estado.
                    trace.append((key, action, 1.0))
                    break

                # Si no gano aun, deja recompensa 0 provisional.
                trace.append((key, action, 0.0))
            else:
                if opponent_mode == "self":
                    if self.rng.random() < epsilon:
                        action = int(self.rng.choice(cols))
                    else:
                        action = self._best_action(board, opp)
                else:
                    # El oponente del entrenamiento juega aleatorio.
                    action = int(self.rng.choice(cols))

                board = self._drop(board, action, opp)

                if self._wins(board, opp) and trace:
                    # Si el rival gana, castigamos la ultima decision
                    # tomada por el agente dentro de esta partida.
                    k, a, _ = trace[-1]
                    trace[-1] = (k, a, -1.0)
                    break

            # Cambiamos el turno para continuar la simulacion.
            turn = -turn

        return trace

    def _update_from_trace(self, trace):
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

    def train(self, num_episodes, opponent_mode="random", epsilon=None):
        for _ in range(num_episodes):
            trace = self._play_training_episode(
                opponent_mode=opponent_mode,
                epsilon=epsilon,
            )
            self._update_from_trace(trace)
            self.training_summary["episodes_trained"] += 1

        self.training_summary["sources"][opponent_mode] = (
            self.training_summary["sources"].get(opponent_mode, 0) + num_episodes
        )

    def train_with_plan(self, phases):
        for phase in phases:
            self.train(
                num_episodes=int(phase["episodes"]),
                opponent_mode=str(phase["mode"]),
                epsilon=float(phase["epsilon"]),
            )

    @classmethod
    def get_training_preset(cls, preset_name: str):
        if preset_name not in cls.TRAINING_PRESETS:
            available = ", ".join(sorted(cls.TRAINING_PRESETS))
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        return [dict(phase) for phase in cls.TRAINING_PRESETS[preset_name]]

    def save_model(self, path: str | Path | None = None):
        model_path = Path(path) if path else self.model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_version": self.MODEL_VERSION,
            "state_mode": "canonical_current_player",
            "training_summary": self.training_summary,
            "values": [
                {
                    "state": list(state),
                    "action": action,
                    "q": self.Q[(state, action)],
                    "n": self.N[(state, action)],
                }
                for state, action in self.Q.keys()
            ],
        }

        with model_path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, path: str | Path | None = None):
        model_path = Path(path) if path else self.model_path
        with model_path.open("rb") as fh:
            payload = pickle.load(fh)

        self.Q = defaultdict(float)
        self.N = defaultdict(int)

        state_mode = payload.get("state_mode", "raw_board")

        for item in payload.get("values", []):
            state = tuple(int(x) for x in item["state"])
            action = int(item["action"])
            q_value = float(item["q"])
            visits = int(item["n"])

            if state_mode == "raw_board":
                board = np.array(state, dtype=int).reshape(6, 7)
                state = self._canonical_key(board)

            previous_visits = self.N[(state, action)]
            total_visits = previous_visits + visits

            if total_visits == 0:
                continue

            previous_q = self.Q[(state, action)]
            self.Q[(state, action)] = (
                (previous_q * previous_visits) + (q_value * visits)
            ) / total_visits
            self.N[(state, action)] = total_visits

        self.training_summary = payload.get(
            "training_summary",
            {"episodes_trained": 0, "sources": {}},
        )
        self._is_ready = True

    def mount(self, timeout=None):
        if self._is_ready:
            return

        if self.model_path.exists():
            self.load_model(self.model_path)
            return

        if self.auto_train_if_missing:
            self.train_with_plan(self.DEFAULT_AUTO_TRAIN_PLAN)
            self.save_model(self.model_path)
            self._is_ready = True
            return

        self._is_ready = True

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

        # 3. Si no hay tactica inmediata, usa la mejor accion segun el Q.
        return self._best_action(s, me)