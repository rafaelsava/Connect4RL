import time
import math
import random
import numpy as np

from connect4.policy import Policy
from connect4.connect_state import ConnectState


class Node:
    """Nodo del arbol MCTS para una accion que lleva a un estado."""

    def __init__(
        self,
        state: ConnectState,
        parent: "Node | None" = None,
        action: int | None = None,
        actions: list[int] | None = None,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = 0 if parent is None else parent.depth + 1
        self.children: dict[int, Node] = {}
        self.untried_actions = list(state.get_free_cols() if actions is None else actions)
        random.shuffle(self.untried_actions)
        self.visits = 0
        self.value = 0.0
        self.player_just_moved = -state.player

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def ucb1_child(self, exploration: float) -> "Node":
        log_parent = math.log(max(1, self.visits))

        def score(child: Node) -> float:
            if child.visits == 0:
                return float("inf")
            exploitation = child.value / child.visits
            # UCB1 puro — C constante sin decay artificial (FIX #2)
            exploration_term = exploration * math.sqrt(log_parent / child.visits)
            return exploitation + exploration_term

        return max(self.children.values(), key=score)

    def expand(self) -> "Node":
        action = self.untried_actions.pop()
        child_state = self.state.transition(action)
        child = Node(child_state, parent=self, action=action)
        self.children[action] = child
        return child

    def best_action(self) -> int:
        if not self.children:
            return random.choice(self.state.get_free_cols())

        return max(
            self.children.values(),
            key=lambda child: (child.visits, child.value / child.visits if child.visits else -float("inf")),
        ).action


class Eyes(Policy):
    """Politica optimizada basada en MCTS con gestion de tiempo adaptativa y rollout hibrido."""

    EXPLORATION = math.sqrt(2.0)        # C constante y puro (FIX #2)
    GLOBAL_TIME_LIMIT = 58.0            # Margen de seguridad sobre 60s (FIX #1)
    TURN_TIME_LIMIT = 1.8               # Presupuesto base por turno
    MIN_TURN_BUDGET = 0.05              # Minimo para no gastar tiempo en nada

    def mount(self, action_timeout=None) -> None:
        self.time_spent = 0.0
        self.turn_time_limit = self._parse_timeout(action_timeout)

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, "time_spent"):
            self.mount()

        board = np.array(s, copy=True)
        current_player = self._infer_current_player(board)
        root_state = ConnectState(board, current_player)
        legal_actions = root_state.get_free_cols()

        if root_state.is_final():
            return random.choice(legal_actions) if legal_actions else 0

        if len(legal_actions) == 1:
            return legal_actions[0]

        # Reflejo de victoria inmediata
        winning_action = self._winning_action(root_state)
        if winning_action is not None:
            return winning_action

        # Escudo defensivo inmediato
        root_actions = self._actions_without_immediate_loss(root_state)
        if len(root_actions) == 1:
            return root_actions[0]

        # --- GESTION DINAMICA DE TIEMPO (FIX #1) ---
        remaining_global_time = self.GLOBAL_TIME_LIMIT - self.time_spent
        if remaining_global_time <= self.MIN_TURN_BUDGET:
            return random.choice(root_actions)

        # Primera pasada corta para medir incertidumbre real
        start_time = time.time()
        root = Node(root_state, actions=root_actions)
        first_pass_end = start_time + 0.08
        while time.time() < first_pass_end:
            self._run_mcts_iteration(root)

        # Calcular budget adaptativo con datos reales del arbol
        budget = self._compute_budget(
            root,
            root_actions,
            int(np.count_nonzero(board)),
            remaining_global_time,
        )

        # Segunda pasada con el presupuesto ya ajustado
        end_time = start_time + budget
        while time.time() < end_time:
            self._run_mcts_iteration(root)

        self.time_spent += time.time() - start_time
        return int(root.best_action())

    def _compute_budget(
        self,
        root: Node,
        legal_actions: list[int],
        pieces_played: int,
        remaining_global: float,
    ) -> float:
        # Factor de fase del juego
        if pieces_played < 8:
            phase_factor = 0.5      # Apertura: posiciones simétricas y similares
        elif pieces_played > 32:
            phase_factor = 0.6      # Endgame: pocas ramas reales
        else:
            phase_factor = 1.0      # Midgame: maxima complejidad tactica

        # Factor de ramificacion: mas columnas = mas espacio a explorar
        branch_factor = len(legal_actions) / 7.0

        base = self.TURN_TIME_LIMIT * phase_factor * branch_factor

        # Ajuste por incertidumbre real tras la primera pasada
        if root.visits > 30 and root.children:
            best = max(root.children.values(), key=lambda c: c.visits)
            best_ratio = best.visits / max(1, root.visits)

            if best_ratio > 0.6:
                base *= 0.5     # Un movimiento domina claramente → recortar
            elif best_ratio < 0.3:
                base *= 1.3     # Posicion muy reñida → extender

        return min(base, remaining_global - self.MIN_TURN_BUDGET)

    def _run_mcts_iteration(self, root: Node) -> None:
        """Una iteracion completa de MCTS: selection, expansion, simulation, backprop."""
        node = root

        # 1. Selection — UCB1 puro con C constante (FIX #2)
        while (
            not node.state.is_final()
            and node.is_fully_expanded()
            and node.children
        ):
            node = node.ucb1_child(self.EXPLORATION)

        # 2. Expansion
        if not node.state.is_final() and not node.is_fully_expanded():
            node = node.expand()

        # 3. Simulation — rollout hibrido con dos reflejos (FIX #3)
        winner = self._rollout(node.state)

        # 4. Backpropagation
        self._backpropagate(node, winner)

    def _infer_current_player(self, board: np.ndarray) -> int:
        red_count = int(np.count_nonzero(board == -1))
        yellow_count = int(np.count_nonzero(board == 1))
        return -1 if red_count == yellow_count else 1

    def _parse_timeout(self, action_timeout) -> float:
        if isinstance(action_timeout, (int, float)):
            return max(self.MIN_TURN_BUDGET, float(action_timeout) * 0.95)
        if isinstance(action_timeout, (tuple, list)):
            for item in action_timeout:
                parsed = self._parse_timeout(item)
                if parsed != self.TURN_TIME_LIMIT:
                    return parsed
        return self.TURN_TIME_LIMIT

    def _rollout(self, state: ConnectState) -> int:
        """Rollout hibrido: dos reflejos tacticos en un solo loop (FIX #3).
        
        En cada paso, en orden de prioridad:
          1. Si puedo ganar ahora → hazlo.
          2. Si el rival gana si no bloqueo → bloquea.
          3. Si no → movimiento aleatorio.
        """
        current = state

        while not current.is_final():
            legal = current.get_free_cols()
            player = current.player
            opponent = -player

            forced = None
            must_block = None

            for action in legal:
                next_s = current.transition(action)
                if not next_s.is_final():
                    continue
                winner = next_s.get_winner()
                if winner == player:
                    forced = action     # Victoria inmediata → salir del loop
                    break
                if winner == opponent:
                    must_block = action # Amenaza del rival → anotar y seguir buscando victoria

            if forced is not None:
                current = current.transition(forced)
            elif must_block is not None:
                current = current.transition(must_block)
            else:
                current = current.transition(random.choice(legal))

        return current.get_winner()

    def _winning_action(self, state: ConnectState) -> int | None:
        if state.is_final():
            return None
        player = state.player
        for action in state.get_free_cols():
            next_state = state.transition(action)
            if next_state.is_final() and next_state.get_winner() == player:
                return action
        return None

    def _actions_without_immediate_loss(self, state: ConnectState) -> list[int]:
        legal_actions = state.get_free_cols()
        if state.is_final():
            return legal_actions

        safe_actions = []
        for action in legal_actions:
            next_state = state.transition(action)
            if next_state.is_final():
                safe_actions.append(action)
                continue
            if self._winning_action(next_state) is None:
                safe_actions.append(action)
        return safe_actions if safe_actions else legal_actions

    def _backpropagate(self, node: Node, winner: int) -> None:
        current: Node | None = node
        while current is not None:
            current.visits += 1
            if winner == current.player_just_moved:
                current.value += 1.0
            elif winner == 0:
                current.value += 0.0
            else:
                current.value -= 1.0
            current = current.parent