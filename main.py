import random
import torch
import chess
from agent import Agent
import sys
import signal

class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.agent1 = Agent(color=chess.WHITE)
        self.agent2 = Agent(color=chess.BLACK)
        self.current_game = 0

    def signal_handler(self, sig, frame):
        print("\n" + "="*50)
        print("Training paused! Choose an option:")
        print("1. Test current model (play game against itself)")
        print("2. Continue training")
        print("3. Save model and exit")
        print("="*50)
        
        while True:
            choice = input("Enter choice (1/2/3): ").strip()
            
            if choice == '1':
                print("Testing current model...")
                self.agent1.play_test_game(self.agent2)
                print("\nTest complete. Choose again:")
                print("1. Test again")
                print("2. Continue training") 
                print("3. Save model and exit")
                
            elif choice == '2':
                print("Resuming training...")
                break
                
            elif choice == '3':
                self.save_and_exit()
                
            else:
                print("Invalid choice. Please enter 1, 2, or 3 or STFU.")

    def save_and_exit(self):
        torch.save(self.agent1.model.state_dict(), f'chess_model_game_{self.current_game}.pth')
        print(f"Model saved as chess_model_game_{self.current_game}.pth")
        print("Exiting...")
        sys.exit(0)


    def sync_boards(self):
        self.agent1.board = self.board.copy()
        self.agent2.board = self.board.copy()

    def run_one_game(self, show_board=False):
        while not (self.board.is_game_over()):

            self.sync_boards()

            if self.board.turn == chess.WHITE:
                # agent 1 move
                move, val = self.agent1.get_best_move_and_val()
                if move:
                    # train agent1
                    self.agent1.train_step(move, val)
                    self.board.push(move)
                else:
                    break
            else:
            
                # agent 2 move
                move, val = self.agent2.get_best_move_and_val()
                if move:
                    # train agent2
                    self.agent2.train_step(move, val)
                    self.board.push(move)
                else:
                    break

            if (show_board):
                print(self.board)
                print("\n")
                input("Press Enter to continue...")
        print("result:", self.board.result())

    def run(self):
        signal.signal(signal.SIGINT, self.signal_handler)

        num_games = 100
        
        print("Training started! Press Ctrl+C anytime to pause and test the model.")
        print(f"Training for {num_games} games...")

        for i in range(num_games):
            self.current_game = i + 1
            print(f"Starting game {i+1}/{num_games}")

            if (i == num_games - 1):
                self.run_one_game(show_board=True)
            else:
                self.run_one_game()
                self.board.reset()
                self.sync_boards()
            print(f"Finished game {i+1}/{num_games}\n\n")
                


if __name__ == "__main__":
    game = ChessGame()
    game.run()
