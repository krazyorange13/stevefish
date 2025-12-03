import random
import torch
import chess
from agent import Agent
import sys
import signal

class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.agent = Agent(color=chess.WHITE)
        self.base_opponent = Agent(color=chess.BLACK)
        self.current_game = 0
        self.game_reward_agent1 = 0
        self.game_reward_agent2 = 0
        self.copy_frequency = 50
        self.opponent_reward = 0
        self.pending_agent1_training = None
        self.training_memory = []
        self.opponent_pool = []

    def signal_handler(self, sig, frame):
        
        while True:
            print("\n" + "="*50)
            print("Training paused! Choose an option:")
            print("1. Test current model against an opponent")
            print("2. Continue training")
            print("3. Save model and exit")
            print("4. Show training loss graph")
            print("="*50)
            choice = input("Enter choice (1/2/3/4): ").strip()
            
            if choice == '1':
                test_opponent = None
                while test_opponent == None:
                    # test against a certain type of opponent
                    opponent_type = input("Enter opponent type (current/older/random): ").strip().lower()
                    if opponent_type == 'current':
                        test_opponent = self.base_opponent
                        test_opponent.epsilon = 0.05
                    elif opponent_type == 'older' and len(self.opponent_pool) > 0:
                        test_opponent = random.choice(self.opponent_pool)
                    elif opponent_type == 'random':
                        test_opponent = self.base_opponent
                        test_opponent.epsilon = 1.0
                    else:
                        print("Invalid opponent type.")
                        continue
                
                print(f"Testing against {opponent_type} opponent...")
                self.agent.play_test_game(test_opponent)
                
            elif choice == '2':
                print("Resuming training...")
                break
                
            elif choice == '3':
                self.save_and_exit()

            elif choice == '4':
                # display losses and stuff and graphs
                self.agent.plot_losses()
                break

            elif choice.lower() == 'stfu':
                # exit without saving
                print("No u")
                sys.exit(0)
                
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4 (or STFU).")

    def save_and_exit(self):
        torch.save(self.agent.model.state_dict(), f'chess_model_game_{self.current_game}.pth')
        print(f"Model saved as chess_model_game_{self.current_game}.pth")
        print("Exiting...")
        sys.exit(0)

    def sync_boards(self, current_opponent):
        self.agent.board = self.board.copy()
        current_opponent.board = self.board.copy()
        self.base_opponent.board = self.board.copy()

    def copy_agent_opponent(self):
        if hasattr(self.base_opponent, 'model'):
            # Keep only last 5 versions
            if len(self.opponent_pool) >= 5:
                random_index = random.randint(0, len(self.opponent_pool) - 1)
                self.opponent_pool.pop(random_index)

            old_agent = Agent(color=chess.BLACK)
            old_agent.model.load_state_dict(self.base_opponent.model.state_dict())
            old_agent.epsilon = 0.1
            self.opponent_pool.append(old_agent)
            
        self.base_opponent.model.load_state_dict(self.agent.model.state_dict())
        self.base_opponent.epsilon = 0.05
        print("Copied Agent 1's model to Agent 2.")

    def get_training_opponent(self, game_number):
        rand = random.random()
        
        if rand < 1:
            # 60% current opponent (main self-play)
            self.base_opponent.epsilon = 0.05
            return self.base_opponent, "Current Opponent"
        elif rand < 0.8 and len(self.opponent_pool) > 0:
            # 20% older version (prevents forgetting basics)
            opponent = random.choice(self.opponent_pool)
            opponent.epsilon = 0.1
            return opponent, "Older Opponent"
        else:
            # 20% random opponent (explores new positions and punishes bc high eps)
            self.base_opponent.epsilon = 1.0
            return self.base_opponent, "Random Opponent"

    def run_one_game(self, show_board=False):
        current_opponent, opponent_type = self.get_training_opponent(self.current_game)

        game_reward_agent1 = 0
        game_reward_agent2 = 0
        self.pending_agent1_training = None

        self.agent.current_game_moves = 0

        self.training_memory = []

        while not (self.board.is_game_over()):

            self.sync_boards(current_opponent)

            if self.board.turn == chess.WHITE:
                # agent 1 move
                move, val = self.agent.get_best_move_and_val()
                if move:
                    # train agent1
                    # game_reward_agent1 += self.agent1.train_step(move, val, self.opponent_reward+25)
                    # self.board.push(move)

                    # store the current state for training after agent2's response
                    old_board = self.board.copy()
                    # old_board_state = self.agent1.board_to_tensor(self.board)

                    self.board.push(move)

                    self.agent.current_game_moves += 1

                    reward = self.agent.getReward(old_board, self.board, move)

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
                move, val = current_opponent.get_best_move_and_val()
                if move:
                    # do not train agent2
                    old_state = self.board.copy()
                    self.board.push(move)
                    self.opponent_reward = current_opponent.getReward(old_state, self.board, move)

                    # this only works bc agent2 isnt trained on this reward. this reward uses agent2's current move and agent1's previous move
                    game_reward_agent2 += self.opponent_reward - self.pending_agent1_training['reward'] if self.pending_agent1_training else 0

                    if self.pending_agent1_training:

                        self.training_memory.append({
                            'move': self.pending_agent1_training['move'],
                            'old_board': self.pending_agent1_training['old_board'],
                            'new_board': self.board.copy(),
                            'reward': self.pending_agent1_training['reward'],
                            'value': self.pending_agent1_training['value'],
                            'opponent_reward': self.opponent_reward + self.agent.move_penalty*10
                        })

                    self.pending_agent1_training = None
                else:
                    break

        # handle final agent1 training if game ended before agent2 could respond
        if self.pending_agent1_training:

            self.training_memory.append({
                'move': self.pending_agent1_training['move'],
                'old_board': self.pending_agent1_training['old_board'],
                'new_board': self.board.copy(),
                'reward': self.pending_agent1_training['reward'],
                'value': self.pending_agent1_training['value'],
                'opponent_reward': 0
            })
        
        print("result:", self.board.result())
        self.agent.log_game_result(self.board.result(), opponent_type)

        print("Opponent type this game:", opponent_type)



        print(f"Training on {len(self.training_memory)} moves from this game...")
        
        # train agent1 on entire game in reverse order
        total_training_reward = 0
        self.training_memory.reverse()
        for mem in self.training_memory:
            training_reward = self.agent.train_step(
                mem['move'],
                mem['old_board'],
                mem['new_board'],
                mem['reward'],
                mem['value'],
                opponent_reward=mem['opponent_reward']
            )
            total_training_reward += training_reward


        final_reward = self.agent.train_batch(target_model=self.base_opponent)
        total_training_reward += final_reward

        print("Game reward Agent 1 (White):", total_training_reward)
        print("Game reward Agent 2 (Black):", game_reward_agent2)

        # reduce epsilon
        self.agent.epsilon = max(self.agent.epsilon * self.agent.epsilon_decay, self.agent.epsilon_min)

        # record moves per game
        if self.agent.current_game_moves > 0:
            self.agent.moves_per_game.append(self.agent.current_game_moves)

    def run(self):
        signal.signal(signal.SIGINT, self.signal_handler)

        num_games = 5000
        
        print("Training started! Press Ctrl+C anytime to pause and test the model.")
        print(f"Training for {num_games} games...")

        for i in range(num_games+1):
            self.current_game = i + 1
            print(f"Starting game {i+1}/{num_games}")

            # copy agent1 to agent2 every copy_frequency games
            if i % self.copy_frequency == 0:
                self.copy_agent_opponent()

            if (i == num_games):
                self.agent.play_test_game(self.base_opponent)
                self.save_and_exit()
            else:
                self.run_one_game()
                self.board.reset()
                self.sync_boards(self.base_opponent)
            print(f"Finished game {i+1}/{num_games}\n\n")
                


if __name__ == "__main__":
    game = ChessGame()
    game.run()
