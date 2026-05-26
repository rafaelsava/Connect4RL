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


class Head(Policy):
    """Politica online basada exclusivamente en MCTS puro con UCB1."""

    EXPLORATION = math.sqrt(2.0)
    GLOBAL_TIME_LIMIT = 58.0
    TURN_TIME_LIMIT = 2.0
    MIN_TURN_BUDGET = 0.05

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

        winning_action = self._winning_action(root_state)
        if winning_action is not None:
            return winning_action

        root_actions = self._actions_without_immediate_loss(root_state)
        if len(root_actions) == 1:
            return root_actions[0]

        remaining_global_time = self.GLOBAL_TIME_LIMIT - self.time_spent
        if remaining_global_time <= self.MIN_TURN_BUDGET:
            return random.choice(root_actions)

        turn_limit = getattr(self, "turn_time_limit", self.TURN_TIME_LIMIT)
        budget = min(turn_limit, remaining_global_time - self.MIN_TURN_BUDGET)
        if budget <= 0:
            return random.choice(root_actions)

        start_time = time.time()
        end_time = start_time + budget
        root = Node(root_state, actions=root_actions)

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
        current = state
        while not current.is_final():
            action = random.choice(current.get_free_cols())
            current = current.transition(action)
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