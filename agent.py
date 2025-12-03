import chess
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

class Agent():
    def __init__(self, color=chess.WHITE):
        self.board = chess.Board()
        self.model = Model().to(device)
        self.color = color
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9995)
        self.criterion = nn.SmoothL1Loss()
        self.epsilon = 1.0
        self.epsilon_decay = 0.993
        self.epsilon_min = 0.05
        self.discount = 0.7

        self.training_buffer = []

        self.losses = []

        self.q_value_stats = {
            'predicted_mean': [],
            'target_mean': []
        }

        self.q_value_diversity = {
            'predicted_std': [],
            'target_std': [], 
            'predicted_range': [],
            'target_range': []
        }

        self.gradient_norms = []
        self.game_results = []
        self.opponent_types = []

        self.moves_per_game = []
        self.current_game_moves = 0

        self.move_penalty = 1.0

    def printBoard(self, board):
        board = board.unicode()

        # remove circles in board representation with dots
        board = board.replace('⭘', '.')

        print(board)

    def board_to_tensor(self, board, color=None):
        if color is None:
            color = self.color

        if color == chess.BLACK:
            board = board.mirror()
        board_tensor = torch.zeros(128, dtype=torch.float32).to(device)

        piece_to_value = {
            None: 0,
            chess.PAWN: 1,
            chess.ROOK: 2,
            chess.KNIGHT: 3,
            chess.BISHOP: 4,
            chess.QUEEN: 5,
            chess.KING: 6
        }

        # first 64 values for board
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                value = piece_to_value[piece.piece_type]

                # negative for enemy pieces
                if piece.color != color:
                    value = -value
                
                board_tensor[square] = value
        
        # next 64 values for attack/defense info
        for square in chess.SQUARES:
            attacks = 0
            if board.is_attacked_by(color, square):
                attacks += 1
            if board.is_attacked_by(not color, square):
                attacks -= 1
            board_tensor[64 + square] = attacks

        # extra for legal moves and check status
        # board_tensor[128] = board.legal_moves.count()
        # board_tensor[129] = board.is_check()
        
        return board_tensor
    
        # # Create 8x8x12 tensor (12 piece types: 6 for us, 6 for opponent)
        # board_3d = torch.zeros(12, 8, 8, dtype=torch.float32)

        # # Fill the 3D board representation
        # for square in chess.SQUARES:
        #     piece = board.piece_at(square)
        #     if piece:
        #         row, col = divmod(square, 8)
        #         if color == chess.WHITE:
        #             row = 7 - row  # Flip for white perspective
                
        #         piece_idx = piece_to_value[piece.piece_type]
        #         if piece.color != color:
        #             piece_idx += 6  # Opponent pieces in indices 6-11
                
        #         board_3d[piece_idx, row, col] = 1.0
        
        # # Flatten to 768 features (12 * 8 * 8 = 768)
        # return board_3d.flatten().to(device)
    
    def get_best_move_and_val(self, board=None, color=None):
        if board is None:
            board = self.board
        if color is None:
            color = self.color

        # get legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0.0
        
        # epsilon-greedy
        if random.random() < self.epsilon:
            rand = random.random()

            if rand < 0.4:
                # capture any piece possilbe
                capture_moves = [move for move in legal_moves if board.is_capture(move)]
                if capture_moves:
                    return random.choice(capture_moves), 0.0
            elif rand < 0.8:
                # move a piece that is under attack
                attacked_moves = []
                for move in legal_moves:
                    if board.is_attacked_by(not color, move.from_square):
                        attacked_moves.append(move)
                if attacked_moves:
                    return random.choice(attacked_moves), 0.0
                
            return random.choice(legal_moves), 0.0
        
        # # get worst q val for each legal move for the enemy
        # best_move = None
        # worst_value = float('inf')
        # for move in legal_moves:
        #     # apply move temporarily
        #     self.board.push(move)
        #     opp_color = not self.color
        #     board_tensor = self.board_to_tensor(self.board, color=opp_color).unsqueeze(0)

        #     with torch.no_grad():
        #         value = self.model.forward(board_tensor).item()

        #     # update best move (worst state for opponent)
        #     if value < worst_value:
        #         worst_value = value
        #         best_move = move

        #     # undo move
        #     self.board.pop()

        # get max q val for each legal move
        best_move = None
        best_value = -float('inf')
        for move in legal_moves:
            # apply move temporarily
            board.push(move)
            board_tensor = self.board_to_tensor(board).unsqueeze(0)

            # foward pass to get q value for each legal move
            with torch.no_grad():
                value = self.model.forward(board_tensor).item()

            # update best move
            if value > best_value:
                best_value = value
                best_move = move

            # undo move
            board.pop()

        return best_move, best_value

        # return best_move, worst_value 
    
    def train_step(self, move, old_board, new_board, reward, value, opponent_reward=0):
        # # get tensor of old board
        # old_board_state = self.board_to_tensor(old_board).unsqueeze(0)
        
        # with torch.no_grad():
        #     if new_board.is_game_over():
        #         target_q = reward - opponent_reward
        #     else:
        #         # next_board_state = self.board_to_tensor(self.board).unsqueeze(0)
        #         next_board_state = self.board_to_tensor(new_board).unsqueeze(0)
        #         next_max_q = self.model(next_board_state).item()
        #         target_q = (reward - opponent_reward) + self.discount * next_max_q

        # predicted_q = self.model.forward(old_board_state)
        # target_q = torch.tensor([[target_q]], dtype=torch.float32, device=device)

        # # Update model
        # self.optimizer.zero_grad()
        # loss = self.criterion(predicted_q, target_q)
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        # self.optimizer.step()

        # return reward - opponent_reward

        self.training_buffer.append({
            'move': move,
            'old_board': old_board,
            'new_board': new_board,
            'reward': reward,
            'value': value,
            'opponent_reward': opponent_reward
        })

        # if len(self.training_buffer) >= self.batch_size:
        #     return self.train_batch()
        
        return 0
    
    def train_batch(self, target_model=None):
        if len(self.training_buffer) == 0:
            return 0
        
        old_states = []
        targets = []
        total_reward = 0

        for sample in self.training_buffer:
            # Convert position to tensor
            old_board_state = self.board_to_tensor(sample['old_board'])
            old_states.append(old_board_state)

            reward = sample['reward'] - sample['opponent_reward']
            
            # Calculate target Q-value
            with torch.no_grad():
                if sample['new_board'].is_game_over():
                    target_q = reward
                else:
                    next_state = self.board_to_tensor(sample['new_board']).unsqueeze(0)

                    # Use target model if provided, otherwise use self.model
                    if target_model is not None:
                        next_max_q = target_model.model(next_state).item()
                    else:
                        next_max_q = self.model(next_state).item()
                    
                    target_q = reward + self.discount * next_max_q
            
            targets.append(target_q)
            total_reward += reward

        # Convert to batch tensors
        old_states_batch = torch.stack(old_states)
        targets_batch = torch.tensor(targets, dtype=torch.float32, device=device).unsqueeze(1)
        
        # Train on entire batch at once
        predicted_qs = self.model(old_states_batch)

        with torch.no_grad():
            # Track Q-value statistics
            pred_values = predicted_qs.detach().cpu().numpy().flatten()
            target_values = targets_batch.detach().cpu().numpy().flatten()
            self.q_value_stats['predicted_mean'].append(pred_values.mean())
            self.q_value_stats['target_mean'].append(target_values.mean())

            self.q_value_diversity['predicted_std'].append(pred_values.std())
            self.q_value_diversity['target_std'].append(target_values.std())
            self.q_value_diversity['predicted_range'].append(pred_values.max() - pred_values.min())
            self.q_value_diversity['target_range'].append(target_values.max() - target_values.min())
    
        
        self.optimizer.zero_grad()
        loss = self.criterion(predicted_qs, targets_batch)
        loss.backward()

        # Track gradient norms (before clipping)
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)
        
        self.optimizer.step()
        # self.scheduler.step()

        # add losses to history
        self.losses.append(loss.item())

        # Add occasional debug output
        if random.random() < 0.05:  # 5% of batches
            print(f"Batch - Loss: {loss.item():.3f}, Q-pred: {pred_values.mean():.3f}, Q-target: {target_values.mean():.3f}, Grad: {total_norm:.3f}")
    
        
        # Clear buffer for next batch
        self.training_buffer.clear()
        
        return total_reward

    def getReward(self, old_state, new_state, move, color=None):
        if color is None:
            color = self.color

        reward = 0

        piece_values = {
            chess.PAWN: 1,
            chess.ROOK: 5,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.QUEEN: 9,
            chess.KING: 0
        }

        if (new_state.is_checkmate()):
            if new_state.turn != color:
                reward += 100 # goood boyyyyy
            else:
                reward -= 100
        elif (new_state.is_stalemate() or new_state.is_insufficient_material()
                or new_state.can_claim_draw() or new_state.can_claim_fifty_moves()
                or new_state.can_claim_threefold_repetition()):
                reward -= 5

        # Reward for capturing pieces
        captured_piece = old_state.piece_at(move.to_square)
        if captured_piece is not None:
            
            piece_value = piece_values.get(captured_piece.piece_type, 0)
            reward += piece_value

            # extra bonus if the captured piece was hanging (not defended)
            if not old_state.is_attacked_by(not color, move.to_square):
                reward += piece_value * 0.5

        # penalty for hanging its own pieces
        our_piece = new_state.piece_at(move.to_square)
        if our_piece and new_state.is_attacked_by(not color, move.to_square):
            if not new_state.is_attacked_by(color, move.to_square):
                reward -= piece_values.get(our_piece.piece_type, 0) * 0.5
    

        if new_state.is_check():
            reward += 1

        if move.promotion:
            reward += 10
        
        # smol penalty for each move to encourage faster wins (or losses)
        reward = reward * 10

        return reward - self.move_penalty
    
    def log_game_result(self, result, opponent_type):
        self.game_results.append(result)
        self.opponent_types.append(opponent_type)

    def get_win_rate(self):
        if len(self.game_results) == 0:
            return 0.0
        wins = self.game_results.count("1-0")
        draws = self.game_results.count("1/2-1/2")
        return wins / len(self.game_results), draws / len(self.game_results)
    
    def plot_losses(self):
        # print average loss and stuff
        avg_loss = sum(self.losses) / len(self.losses)
        print(f"Average Training Loss: {avg_loss:.4f}")
        print(f"Total Batch Steps: {len(self.losses)}")
        print(f"Current Epsilon: {self.epsilon:.4f}")
        win_rate, draw_rate = self.get_win_rate()
        print(f"Current Win Rate: {win_rate:.2%}, Draw Rate: {draw_rate:.2%}")
        print(f"Total Games Played: {len(self.game_results)}")

        plt.figure(figsize=(18, 10))

        # display losses graph at this point
        plt.subplot(3, 3, 1)
        plt.plot(self.losses)
        plt.axhline(y=avg_loss, color='red', linestyle='--', alpha=0.7, label='Average Loss')
        plt.legend()
        plt.title("Training Loss Over Time")
        plt.xlabel("Batches")
        plt.ylabel("Loss")
        
        # Moving average
        plt.subplot(3, 3, 2)
        window = 20
        smoothed = [sum(self.losses[max(0, i-window):i+1])/min(i+1, window) 
                   for i in range(len(self.losses))]
        if smoothed:
            mean_smoothed = sum(smoothed) / len(smoothed)
            plt.axhline(y=mean_smoothed, color='red', linestyle='--', alpha=0.7, 
                    label='Average Smoothed Loss')
        plt.plot(smoothed, label='Smoothed Loss')
        plt.title("Smoothed Training Loss Over Time")
        plt.xlabel("Batches")
        plt.ylabel("Smoothed Loss")
        plt.legend()
        
        # latest trend of losses
        plt.subplot(3, 3, 3)
        if (len(smoothed) < 50):
            plt.plot(smoothed)
        else:
            plt.plot(smoothed[-50:])
        plt.title("Recent Loss Trend")
        plt.xlabel("Recent Batches")
        plt.ylabel("Smoothed Loss")

        # Q-values
        plt.subplot(3, 3, 4)
        if len(self.q_value_stats['predicted_mean']) > 0:
            pred_means = self.q_value_stats['predicted_mean']
            target_means = self.q_value_stats['target_mean']

            # Calculate differences
            q_differences = [abs(p - t) for p, t in zip(pred_means, target_means)]

            plt.plot(pred_means, label='Predicted Q', alpha=0.8)
            plt.plot(target_means, label='Target Q', alpha=0.8)
            plt.plot(q_differences, label='|Pred - Target|', alpha=0.8, color='red')
            
            # Add mean lines
            mean_pred = sum(pred_means) / len(pred_means)
            mean_target = sum(target_means) / len(target_means)
            
            plt.axhline(y=mean_pred, color='blue', linestyle='--', alpha=0.7, 
                    label=f'Pred Mean: {mean_pred:.1f}')
            plt.axhline(y=mean_target, color='orange', linestyle='--', alpha=0.7, 
                    label=f'Target Mean: {mean_target:.1f}')
            
            plt.title("Q-Value Means")
            plt.xlabel("Batches")
            plt.ylabel("Q-Value")
            plt.legend()

        # win rate
        plt.subplot(3, 3, 5)
        if len(self.game_results) > 50:
            # calculate rolling win rate over time
            interval = 100
            half_interval = interval // 2
            step = 5

            # win_rates = []
            game_numbers = []

            current_win_rates = []
            older_win_rates = []
            random_win_rates = []

            for i in range(half_interval, len(self.game_results) - half_interval, step):
                window_results = self.game_results[i-half_interval:i+half_interval]
                window_types = self.opponent_types[i-half_interval:i+half_interval]

                # Win rates by opponent type
                for opp_type, wr_list in [
                    ("Current Opponent", current_win_rates),
                    ("Older Opponent", older_win_rates), 
                    ("Random Opponent", random_win_rates)
                ]:
                    type_games = [res for res, typ in zip(window_results, window_types) if typ == opp_type]
                    if len(type_games) >= 3:
                        wins = type_games.count('1-0')
                        wr_list.append(wins / len(type_games))
                    else:
                        wr_list.append(None)

                game_numbers.append(i + 1)
            
            # Plot lines
            if current_win_rates:
                plt.plot(game_numbers, current_win_rates, 'b-', label='vs Current', alpha=0.8)
            if older_win_rates:
                plt.plot(game_numbers, older_win_rates, 'g-', label='vs Older', alpha=0.8)
            if random_win_rates:
                plt.plot(game_numbers, random_win_rates, 'r-', label='vs Random', alpha=0.8)
            
            plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='50% Win Rate')
            plt.title("Win Rate Over Time")
            plt.xlabel("Games")
            plt.ylabel("Win Rate")
            plt.ylim(0, 1)
            plt.legend()

        # gradient norms
        plt.subplot(3, 3, 6)
        if len(self.gradient_norms) > 0:
            plt.plot(self.gradient_norms, alpha=0.7, color='orange')

            # Add mean gradient norm line
            mean_grad = sum(self.gradient_norms) / len(self.gradient_norms)
            plt.axhline(y=mean_grad, color='red', linestyle='--', alpha=0.7, 
                    label=f'Mean: {mean_grad:.3f}')

            plt.axhline(y=100.0, color='red', linestyle='--', alpha=0.7, label='Clip Threshold')
            plt.title("Gradient Norms")
            plt.xlabel("Batches")
            plt.ylabel("Gradient Norm")
            plt.legend()

        # Add new subplot for game lengths
        plt.subplot(3, 3, 7)
        if len(self.moves_per_game) > 0:
            plt.plot(self.moves_per_game, 'o-', alpha=0.7, markersize=3)
            
            # Add mean line
            mean_moves = sum(self.moves_per_game) / len(self.moves_per_game)
            plt.axhline(y=mean_moves, color='red', linestyle='--', alpha=0.7, 
                    label=f'Mean: {mean_moves:.1f}')
            
            plt.title("Game Lengths Over Time")
            plt.xlabel("Game Number")
            plt.ylabel("Moves per Game")
            plt.legend()

        # q value diversity plot (bc we always need more diversity)
        plt.subplot(3, 3, 8)
        if hasattr(self, 'q_value_diversity') and len(self.q_value_diversity['predicted_std']) > 0:
            pred_stds = self.q_value_diversity['predicted_std']
            target_stds = self.q_value_diversity['target_std']
            pred_ranges = self.q_value_diversity['predicted_range']
            target_ranges = self.q_value_diversity['target_range']
            
            plt.plot(pred_stds, label='Predicted Std', alpha=0.8)
            plt.plot(target_stds, label='Target Std', alpha=0.8)
            # plt.plot(pred_ranges, label='Predicted Range', alpha=0.5, linestyle='--')
            # plt.plot(target_ranges, label='Target Range', alpha=0.5, linestyle='--')
            
            # Add mean lines
            mean_pred_std = sum(pred_stds) / len(pred_stds)
            mean_target_std = sum(target_stds) / len(target_stds)
            
            plt.axhline(y=mean_pred_std, color='blue', linestyle='--', alpha=0.7, 
                    label=f'Pred Std Mean: {mean_pred_std:.1f}')
            plt.axhline(y=mean_target_std, color='orange', linestyle='--', alpha=0.7, 
                    label=f'Target Std Mean: {mean_target_std:.1f}')
            
            plt.title("Q-Value Diversity (Standard Deviation)")
            plt.xlabel("Batches")
            plt.ylabel("Standard Deviation")
            plt.legend()

        plt.tight_layout()
        plt.show()
    
    def play_test_game(self, opponent_agent=None):

        print("\n" + "="*40)
        print("STARTING TEST GAME")
        print("="*40)
        
        # save current training state
        original_board = self.board.copy()
        original_epsilon = self.epsilon
        
        # create new board and disable epsilon
        test_board = chess.Board()
        self.board = test_board
        self.epsilon = 0.0
        
        if opponent_agent is None:
            opponent_agent = Agent(color=chess.BLACK)
        
        # save opponent state and set up for test
        opponent_original_board = opponent_agent.board.copy()
        opponent_original_epsilon = opponent_agent.epsilon
        opponent_agent.board = test_board
        opponent_agent.epsilon = 0.0
        
        move_count = 0
        
        try:
            print("Initial board:")
            self.printBoard(test_board)
            
            while not test_board.is_game_over():
                input("Press Enter to continue to the next move...")

                move_count += 1
                
                if test_board.turn == chess.WHITE:
                    move, value = self.get_best_move_and_val()
                    player = "White"
                else:
                    move, value = opponent_agent.get_best_move_and_val()
                    player = "Black"
                
                if move:
                    captured_piece = test_board.piece_at(move.to_square)

                    test_board.push(move)
                    
                    self.printBoard(test_board)
                    print(f"Move {move_count}: {player} plays {move} (value: {value:.3f})")
                    if captured_piece:
                        print(f"{player} captures {captured_piece.symbol()} on {chess.square_name(move.to_square)}")
                    
                else:
                    break
            
            print(f"\nFinal board after {move_count} moves:")
            self.printBoard(test_board)
            
            # print result
            if test_board.is_checkmate():
                winner = "White" if test_board.turn == chess.BLACK else "Black"
                print(f"\nResult: {winner} wins by checkmate!")
            elif test_board.is_stalemate():
                print("\nResult: Draw by stalemate!")
            elif test_board.is_insufficient_material():
                print("\nResult: Draw by insufficient material!")
            else:
                print("\nResult: Draw! For some random reason!")
                
        finally:
            # restore everything
            self.board = original_board
            self.epsilon = original_epsilon
            opponent_agent.board = opponent_original_board
            opponent_agent.epsilon = opponent_original_epsilon
            
        print("="*40)
        print("TEST GAME COMPLETE - Resuming training...")
        print("="*40 + "\n")
    

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )

        # Input: 8x8x12 (8x8 board, 12 piece types)
        # self.conv_layers = nn.Sequential(
        #     # First conv layer - detect basic piece patterns
        #     nn.Conv2d(12, 64, kernel_size=3, padding=1),  # 8x8x64
        #     nn.LeakyReLU(0.1),
            
        #     # Second conv layer - detect piece interactions
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1), # 8x8x128
        #     nn.LeakyReLU(0.1),
            
        #     # Third conv layer - detect larger patterns
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1), # 8x8x256
        #     nn.LeakyReLU(0.1),
            
        #     # Global average pooling instead of flatten
        #     nn.AdaptiveAvgPool2d(1)  # 1x1x256
        # )
        
        # Fully connected layers for final evaluation
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(256, 1)
        # )

    def forward(self, x):
        return self.layers(x)

        # batch_size = x.shape[0]
        # board_tensor = x.view(batch_size, 12, 8, 8)
        
        # conv_out = self.conv_layers(board_tensor)
        # conv_out = conv_out.view(batch_size, -1)  # Flatten to (batch_size, 256)
        
        # return self.fc_layers(conv_out)