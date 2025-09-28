import random
import torch
import chess


class ChessGame:
    def __init__(self):
        self.board = chess.Board()

    def run(self):
        while not (
            self.board.is_checkmate()
            or self.board.is_game_over()
            or self.board.can_claim_draw()
            or self.board.can_claim_fifty_moves()
            or self.board.can_claim_threefold_repetition()
            # idk what these do
            # or self.board.is_variant_draw()
            # or self.board.is_variant_end()
            # or self.board.is_variant_loss()
            # or self.board.is_variant_win()
        ):
            moves = list(self.board.generate_legal_moves())
            if not moves:
                return
            move = random.choice(moves)
            self.board.push(move)
            print(move)
            print(self.board)
            print("\n")
            input("Press Enter to continue...")
        print("result:", self.board.result())


if __name__ == "__main__":
    game = ChessGame()
    game.run()
