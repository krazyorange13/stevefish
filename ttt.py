import numpy as np


class TTT:
    # algorithms should support larger sizes, but not tested yet
    WIDTH = 3
    HEIGHT = 3
    WIN_RUN = 3
    N_PLAYERS = 2

    def __init__(self, board: np.ndarray | None = None):
        # verify size
        board = (
            board
            if board is not None
            and board.shape == (TTT.HEIGHT, TTT.WIDTH)
            and board.dtype == np.int32
            else None
        )

        # create board
        # indexed as self.board[y][x] or self.board[row][col]
        self.board: np.ndarray = (
            board
            if board is not None
            else np.zeros((TTT.HEIGHT, TTT.WIDTH), dtype=np.int32)
        )

    def move(self, p, row, col):
        if [row, col] not in self.get_legal_moves():
            return False

        # turn check
        counts = [
            np.count_nonzero(self.board == p) for p in range(1, TTT.N_PLAYERS + 1)
        ]
        least_p = np.argmin(counts) + 1
        if p != least_p:
            print(
                f"TTT.move warning: p={p} != least_p={least_p}. The turns may be out of order."
            )

        self.board[row][col] = p
        return True

    def get_legal_moves(self) -> np.ndarray:
        # return all empty squares
        # array of [row,col] pairs
        return np.argwhere(self.board == 0)

    def get_wins(self):
        # return a list of bools, where [0] corresponds to player 1
        return [self.get_win(p) for p in range(1, TTT.N_PLAYERS + 1)]

    def get_win(self, p: int):
        return self._get_win_rows(p) or self._get_win_cols(p) or self._get_win_diags(p)

    def _get_win_rows(self, p):
        for row_i in range(TTT.HEIGHT):
            if self._get_win_row(row_i, p):
                return True
        return False

    def _get_win_cols(self, p):
        for col_i in range(TTT.WIDTH):
            if self._get_win_col(col_i, p):
                return True
        return False

    def _get_win_row(self, row_i, p):
        counter = 0
        for col_i in range(TTT.WIDTH):
            if self.board[row_i][col_i] == p:
                counter += 1
                if counter >= TTT.WIN_RUN:
                    return True
            else:
                counter = 0
        return False

    def _get_win_col(self, col_i, p):
        counter = 0
        for row_i in range(TTT.HEIGHT):
            if self.board[row_i][col_i] == p:
                counter += 1
                if counter >= TTT.WIN_RUN:
                    return True
            else:
                counter = 0
        return False

    def _get_win_diags(self, p):
        return self._get_win_diags_up(p) or self._get_win_diags_down(p)

    def _get_win_diags_up(self, p):
        # diags from bottom left to top right
        # print(f"TTT._get_win_diags_up({p})")
        for diag_i in range(self._get_n_diags()):
            if self._get_win_diag_up(diag_i, p):
                return True
        return False

    def _get_win_diags_down(self, p):
        # diags from top left to bottom right
        # print(f"TTT._get_win_diags_down({p})")
        for diag_i in range(self._get_n_diags()):
            if self._get_win_diag_down(diag_i, p):
                return True
        return False

    def _get_win_diag_up(self, diag_i, p):
        # diag starting going toward top right
        # print(f"\tTTT._get_win_diag_up({diag_i}, {p})")
        # starting at row_i=min(0,diag_i-(TTT.HEIGHT-1)) and col_i=min(diag_i,TTT.HEIGHT-1)
        # ending at row_i=min(diag_i,TTT.WIDTH-1) and col_i=min(0,diag_i-(TTT.WIDTH-1))
        row_i_start = max(0, diag_i - (TTT.HEIGHT - 1))
        col_i_start = min(diag_i, TTT.HEIGHT - 1)
        row_i_end = min(diag_i, TTT.WIDTH - 1)
        col_i_end = max(0, diag_i - (TTT.WIDTH - 1))
        # print(f"\t\trow_i_start = {row_i_start}, row_i_end = {row_i_end}")
        # print(f"\t\tcol_i_start = {col_i_start}, col_i_end = {col_i_end}")
        counter = 0
        for row_i, col_i in zip(
            range(row_i_start, row_i_end + 1),
            range(col_i_start, col_i_end - 1, -1),
        ):
            if self.board[row_i][col_i] == p:
                counter += 1
            else:
                counter = 0
            # print(f"\t\t{counter}")
            if counter >= TTT.WIN_RUN:
                # print("\t\tTrue")
                return True
        # print("\t\tFalse")
        return False

    def _get_win_diag_down(self, diag_i, p):
        # diag starting going toward bottom right
        # print(f"\tTTT._get_win_diag_down({diag_i}, {p})")
        # same as _get_win_diag_up but col_i_start and col_i_end are swapped
        row_i_start = max(0, diag_i - (TTT.HEIGHT - 1))
        col_i_start = min(diag_i, TTT.HEIGHT - 1)
        row_i_end = min(diag_i, TTT.WIDTH - 1)
        col_i_end = max(0, diag_i - (TTT.WIDTH - 1))
        # print(f"\t\trow_i_start = {row_i_start}, row_i_end = {row_i_end}")
        # print(f"\t\tcol_i_start = {col_i_start}, col_i_end = {col_i_end}")
        counter = 0
        for row_i, col_i in zip(
            range(row_i_start, row_i_end + 1),
            range(col_i_end, col_i_start + 1),
        ):
            if self.board[row_i][col_i] == p:
                counter += 1
            else:
                counter = 0
            # print(f"\t\t{counter}")
            if counter >= TTT.WIN_RUN:
                # print("\t\tTrue")
                return True
        # print("\t\tFalse")
        return False

    def _get_n_diags_opportunities(self):
        # number of diags that can be won
        # this is half of the total diags, for up and down multiply by 2
        return (TTT.HEIGHT - TTT.WIN_RUN + 1) * (TTT.WIDTH - TTT.WIN_RUN + 1)

    def _get_n_diags(self):
        # number of diags that can be drawn
        return TTT.HEIGHT + TTT.WIDTH - 1

    def get_draw(self):
        # no moves and no winner
        return len(self.get_legal_moves()) == 0 and not any(self.get_wins())

    def __str__(self):
        return str(self.board)


if __name__ == "__main__":
    game = TTT()
    print(game)
