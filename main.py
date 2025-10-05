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
        self.agent2.epsilon = 0.0
        self.current_game = 0
        self.game_reward_agent1 = 0
        self.game_reward_agent2 = 0
        self.copy_frequency = 50
        self.opponent_reward = 0
        self.pending_agent1_training = None

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

            elif choice.lower() == 'stfu':
                # exit without saving
                print("No u")
                sys.exit(0)
                
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

    def copy_agent1_to_agent2(self):
        self.agent2.model.load_state_dict(self.agent1.model.state_dict())
        print("Copied Agent 1's model to Agent 2.")

    def run_one_game(self, show_board=False):
        game_reward_agent1 = 0
        game_reward_agent2 = 0
        self.pending_agent1_training = None
        while not (self.board.is_game_over()):

            self.sync_boards()

            if self.board.turn == chess.WHITE:
                # agent 1 move
                move, val = self.agent1.get_best_move_and_val()
                if move:
                    # train agent1
                    # game_reward_agent1 += self.agent1.train_step(move, val, self.opponent_reward+25)
                    # self.board.push(move)

                    # store the current state for training after agent2's response
                    old_board = self.board.copy()
                    # old_board_state = self.agent1.board_to_tensor(self.board)

                    self.board.push(move)

                    reward = self.agent1.getReward(old_board, self.board, move)

                    # store the move for training after agent2 responds
                    self.pending_agent1_training = {
                        'move': move,
                        'old_board': old_board,
                        'value': val,
                        'reward': reward
                    }
                else:
                    break

                self.opponent_reward = 0
            else:
            
                # agent 2 move
                move, val = self.agent2.get_best_move_and_val()
                if move:
                    # do not train agent2
                    old_state = self.board.copy()
                    self.board.push(move)
                    self.opponent_reward = self.agent2.getReward(old_state, self.board, move)

                    # this only works bc agent2 isnt trained on this reward. this reward uses agent2's current move and agent1's previous move
                    game_reward_agent2 += self.opponent_reward - self.pending_agent1_training['reward'] if self.pending_agent1_training else 0

                    if self.pending_agent1_training:
                        total_reward = self.agent1.train_step(
                            self.pending_agent1_training['move'],
                            self.pending_agent1_training['old_board'],
                            self.board,
                            self.pending_agent1_training['reward'],
                            self.pending_agent1_training['value'],
                            opponent_reward=self.opponent_reward + 25
                        )
                        game_reward_agent1 += total_reward

                    self.pending_agent1_training = None
                else:
                    break

        # handle final agent1 training if game ended before agent2 could respond
        if self.pending_agent1_training:
            actual_training_reward = self.agent1.train_step(
                self.pending_agent1_training['move'],
                self.pending_agent1_training['old_board'],
                self.board,
                self.pending_agent1_training['reward'],
                self.pending_agent1_training['value'],
                opponent_reward=0
            )
            game_reward_agent1 += actual_training_reward
        
        print("result:", self.board.result())
        print("Game reward Agent 1 (White):", game_reward_agent1)
        print("Game reward Agent 2 (Black):", game_reward_agent2)

    def run(self):
        signal.signal(signal.SIGINT, self.signal_handler)

        num_games = 3000
        
        print("Training started! Press Ctrl+C anytime to pause and test the model.")
        print(f"Training for {num_games} games...")

        for i in range(num_games+1):
            self.current_game = i + 1
            print(f"Starting game {i+1}/{num_games}")

            # copy agent1 to agent2 every copy_frequency games
            if i % self.copy_frequency == 0:
                self.copy_agent1_to_agent2()

            if (i == num_games):
                self.agent1.play_test_game(self.agent2)
                self.save_and_exit()
            else:
                self.run_one_game()
                self.board.reset()
                self.sync_boards()
            print(f"Finished game {i+1}/{num_games}\n\n")
                


if __name__ == "__main__":
    game = ChessGame()
    game.run()
