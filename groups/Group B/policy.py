import numpy as np

from connect4.policy import Policy


class SafeHeuristicAgent(Policy):
    """
    Agente heuristico simple para Connect 4.

    Orden de decision:
    1. ganar si existe una jugada ganadora inmediata
    2. bloquear si el rival puede ganar en una jugada
    3. evitar jugadas que dejen ganar al rival en el siguiente turno
    4. entre las jugadas seguras, preferir las mas prometedoras
    """

    CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]

    def mount(self) -> None:
        # Este agente no necesita entrenamiento previo.
        pass

    def _cols(self, board: np.ndarray) -> list[int]:
        return [c for c in range(7) if board[0, c] == 0]

    def _drop(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
        new_board = board.copy()
        for r in range(5, -1, -1):
            if new_board[r, col] == 0:
                new_board[r, col] = player
                break
        return new_board

    def _wins(self, board: np.ndarray, player: int) -> bool:
        for r in range(6):
            for c in range(4):
                if all(board[r, c + i] == player for i in range(4)):
                    return True

        for r in range(3):
            for c in range(7):
                if all(board[r + i, c] == player for i in range(4)):
                    return True

        for r in range(3):
            for c in range(4):
                if all(board[r + i, c + i] == player for i in range(4)):
                    return True

        for r in range(3):
            for c in range(3, 7):
                if all(board[r + i, c - i] == player for i in range(4)):
                    return True

        return False

    def _whose_turn(self, board: np.ndarray) -> int:
        tokens = np.count_nonzero(board)
        return -1 if tokens % 2 == 0 else 1

    def _window_score(self, window: np.ndarray, me: int, opp: int) -> int:
        me_count = int(np.sum(window == me))
        opp_count = int(np.sum(window == opp))
        empty_count = int(np.sum(window == 0))

        # Si ambos jugadores ya ocupan la misma ventana, no es util.
        if me_count > 0 and opp_count > 0:
            return 0

        if me_count == 4:
            return 100000
        if me_count == 3 and empty_count == 1:
            return 120
        if me_count == 2 and empty_count == 2:
            return 12
        if me_count == 1 and empty_count == 3:
            return 2

        if opp_count == 3 and empty_count == 1:
            return -150
        if opp_count == 2 and empty_count == 2:
            return -15

        return 0

    def _board_score(self, board: np.ndarray, me: int) -> int:
        opp = -me
        score = 0

        # El centro suele ser la mejor zona del tablero.
        center_col = board[:, 3]
        score += int(np.sum(center_col == me)) * 8

        for r in range(6):
            for c in range(4):
                score += self._window_score(board[r, c : c + 4], me, opp)

        for r in range(3):
            for c in range(7):
                score += self._window_score(board[r : r + 4, c], me, opp)

        for r in range(3):
            for c in range(4):
                score += self._window_score(
                    np.array([board[r + i, c + i] for i in range(4)]), me, opp
                )

        for r in range(3):
            for c in range(3, 7):
                score += self._window_score(
                    np.array([board[r + i, c - i] for i in range(4)]), me, opp
                )

        return score

    def _is_safe_move(self, board: np.ndarray, col: int, me: int) -> bool:
        opp = -me
        next_board = self._drop(board, col, me)
        for opp_col in self._cols(next_board):
            opp_board = self._drop(next_board, opp_col, opp)
            if self._wins(opp_board, opp):
                return False
        return True

    def _move_score(self, board: np.ndarray, col: int, me: int) -> int:
        opp = -me
        next_board = self._drop(board, col, me)
        my_score = self._board_score(next_board, me)

        opp_responses = self._cols(next_board)
        if not opp_responses:
            return my_score

        opp_best_score = max(
            self._board_score(self._drop(next_board, opp_col, opp), opp)
            for opp_col in opp_responses
        )
        return my_score - opp_best_score

    def _center_rank(self, col: int) -> int:
        return self.CENTER_ORDER.index(col)

    def act(self, s: np.ndarray) -> int:
        cols = self._cols(s)
        me = self._whose_turn(s)
        opp = -me

        # 1. Si puedo ganar ya, juego esa columna.
        for col in cols:
            if self._wins(self._drop(s, col, me), me):
                return col

        # 2. Si el rival gana en una, bloqueo.
        for col in cols:
            if self._wins(self._drop(s, col, opp), opp):
                return col

        # 3. Evitar jugadas que permitan una victoria inmediata del rival.
        safe_cols = [col for col in cols if self._is_safe_move(s, col, me)]
        candidate_cols = safe_cols if safe_cols else cols

        # 4. Escoger la jugada con mejor heuristica. En empate, preferir centro.
        return max(
            candidate_cols,
            key=lambda col: (
                self._move_score(s, col, me),
                -self._center_rank(col),
            ),
        )
