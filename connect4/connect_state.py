# Abstract
from connect4.environment_state import EnvironmentState

# Types
from typing import Any

# Libraries
import numpy as np


class ConnectState(EnvironmentState):
    __slots__ = ("board", "player", "last_move", "_winner", "_is_final")

    ROWS = 6
    COLS = 7

    def __init__(
        self,
        board: np.ndarray | None = None,
        player: int = -1,
        last_move: tuple[int, int] | None = None,
        winner: int | None = None,
        copy_board: bool = True,
    ):
        if board is None:
            self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        else:
            self.board = board.astype(np.int8, copy=copy_board)
        self.player = int(player)  # -1 = Red, 1 = Yellow type: ignore
        self.last_move = last_move
        self._winner = winner
        self._is_final = None

    def is_final(self) -> bool:
        if self._is_final is None:
            self._is_final = self.get_winner() != 0 or not np.any(self.board[0] == 0)
        return self._is_final

    def is_applicable(self, event: Any) -> bool:
        return (
            isinstance(event, int)
            and 0 <= event < self.COLS
            and self.is_col_free(event)
            and not self.is_final()
        )

    def get_winner(self) -> int:
        if self._winner is not None:
            return self._winner

        if self.last_move is not None:
            row, col = self.last_move
            player = self.board[row, col]
            self._winner = self._winner_from_cell(row, col, player)
            return self._winner

        # Check all 4 directions
        for r in range(self.ROWS):
            for c in range(self.COLS):
                player = self.board[r, c]
                if player == 0:
                    continue

                # Right
                if c + 3 < self.COLS and all(
                    self.board[r, c + i] == player for i in range(4)
                ):
                    self._winner = int(player)
                    return self._winner
                # Down
                if r + 3 < self.ROWS and all(
                    self.board[r + i, c] == player for i in range(4)
                ):
                    self._winner = int(player)
                    return self._winner
                # Diagonal right-down
                if (
                    r + 3 < self.ROWS
                    and c + 3 < self.COLS
                    and all(self.board[r + i, c + i] == player for i in range(4))
                ):
                    self._winner = int(player)
                    return self._winner
                # Diagonal left-down
                if (
                    r + 3 < self.ROWS
                    and c - 3 >= 0
                    and all(self.board[r + i, c - i] == player for i in range(4))
                ):
                    self._winner = int(player)
                    return self._winner

        self._winner = 0
        return self._winner

    def _winner_from_cell(self, row: int, col: int, player: int) -> int:
        if player == 0:
            return 0

        board = self.board
        rows = self.ROWS
        cols = self.COLS
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (-1, 1):
                r = row + sign * dr
                c = col + sign * dc
                while (
                    0 <= r < rows
                    and 0 <= c < cols
                    and board[r, c] == player
                ):
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                return int(player)
        return 0

    def is_col_free(self, col: int) -> bool:
        return bool(self.board[0, col] == 0)

    def get_heights(self) -> list[int]:
        heights = []
        for c in range(self.COLS):
            col = self.board[:, c]
            for r in range(self.ROWS):
                if col[r] != 0:
                    heights.append(self.ROWS - r)
                    break
            else:
                heights.append(0)
        return heights

    def get_free_cols(self) -> list[int]:
        return [c for c in range(self.COLS) if self.board[0, c] == 0]

    def transition(self, col: int) -> "ConnectState":
        if (
            not isinstance(col, int)
            or col < 0
            or col >= self.COLS
            or self.board[0, col] != 0
            or self.is_final()
        ):
            raise ValueError(f"Move not allowed in column {col}.")

        new_board = self.board.copy()
        placed_row = -1
        for r in reversed(range(self.ROWS)):
            if new_board[r, col] == 0:
                new_board[r, col] = self.player
                placed_row = r
                break

        winner = self._winner_from_cell(placed_row, col, self.player)
        return ConnectState(
            new_board,
            -self.player,
            (placed_row, col),
            winner,
            copy_board=False,
        )

    def show(self, size: int = 1500, ax=None) -> None:
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        pos_red = np.where(self.board == -1)
        pos_yellow = np.where(self.board == 1)

        ax.scatter(pos_yellow[1] + 0.5, 5.5 - pos_yellow[0], color="yellow", s=size)
        ax.scatter(pos_red[1] + 0.5, 5.5 - pos_red[0], color="red", s=size)

        ax.set_ylim([0, self.board.shape[0]])
        ax.set_xlim([0, self.board.shape[1]])
        ax.set_xticks(np.arange(self.board.shape[1] + 1))
        ax.set_yticks(np.arange(self.board.shape[0] + 1))
        ax.grid(True)

        ax.set_title("Connect Four")

        if fig is not None:
            plt.show()
