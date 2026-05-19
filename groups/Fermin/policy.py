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
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: dict[int, Node] = {}
        self.untried_actions = list(state.get_free_cols())
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

        # Al jugar, la decision final se toma por robustez: accion mas visitada.
        return max(
            self.children.values(),
            key=lambda child: (child.visits, child.value / child.visits if child.visits else 0.0),
        ).action


class Brain(Policy):
    """Politica online basada exclusivamente en MCTS puro con UCB1."""

    EXPLORATION = math.sqrt(2.0)
    GLOBAL_TIME_LIMIT = 55.0
    TURN_TIME_LIMIT = 1.8
    MIN_TURN_BUDGET = 0.02

    def mount(self) -> None:
        self.time_spent = 0.0

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, "time_spent"):
            self.mount()

        board = np.array(s, copy=True)
        current_player = self._infer_current_player(board)
        root_state = ConnectState(board, current_player)
        legal_actions = root_state.get_free_cols()

        if len(legal_actions) == 1:
            return legal_actions[0]

        remaining_global_time = self.GLOBAL_TIME_LIMIT - self.time_spent
        if remaining_global_time <= self.MIN_TURN_BUDGET:
            return random.choice(legal_actions)

        budget = min(self.TURN_TIME_LIMIT, remaining_global_time - self.MIN_TURN_BUDGET)
        if budget <= 0:
            return random.choice(legal_actions)

        start_time = time.time()
        end_time = start_time + budget
        root = Node(root_state)

        while time.time() < end_time:
            node = root

            # 1. Selection: bajar por hijos ya expandidos usando UCB1.
            while (
                not node.state.is_final()
                and node.is_fully_expanded()
                and node.children
            ):
                node = node.ucb1_child(self.EXPLORATION)

            # 2. Expansion: agregar una accion legal no probada.
            if not node.state.is_final() and not node.is_fully_expanded():
                node = node.expand()

            # 3. Simulation: rollout completamente aleatorio hasta terminal.
            winner = self._rollout(node.state)

            # 4. Backpropagation: recompensa desde la perspectiva alternante.
            self._backpropagate(node, winner)

        self.time_spent += time.time() - start_time
        return int(root.best_action())

    def _infer_current_player(self, board: np.ndarray) -> int:
        red_count = int(np.count_nonzero(board == -1))
        yellow_count = int(np.count_nonzero(board == 1))
        return -1 if red_count == yellow_count else 1

    def _rollout(self, state: ConnectState) -> int:
        current = state
        while not current.is_final():
            action = random.choice(current.get_free_cols())
            current = current.transition(action)
        return current.get_winner()

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
