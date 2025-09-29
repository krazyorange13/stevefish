import random
import torch
import chess
from agent import Agent

class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.agent1 = Agent(color=chess.WHITE)
        self.agent2 = Agent(color=chess.BLACK)

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
            
            # agent 1 move
            move, val =self.agent1.get_best_move_and_val()
            agent1_move = self.agent1.train_step(move, val)

            # update main board
            self.board.push(agent1_move)
            # update agents boards
            self.agent1.board.push(agent1_move)
            self.agent2.board.push(agent1_move)

            # check if game over after agent 1 move
            if self.board.is_game_over():
                return
            
            # agent 2 move
            agent2_move = self.agent2.get_best_move_and_val()[0]
            self.board.push(agent2_move)
            
            # update main board
            self.board.push(agent1_move)
            # update agents boards
            self.agent1.board.push(agent2_move)
            self.agent2.board.push(agent2_move)

            print(self.board)
            print("\n")
            input("Press Enter to continue...")
        print("result:", self.board.result())


if __name__ == "__main__":
    game = ChessGame()
    game.run()
