import chess
import torch
import torch.nn as nn
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

class Agent():
    def __init__(self, color=chess.WHITE):
        self.board = chess.Board()
        self.model = Model().to(device)
        self.color = color
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=0.01, betas=(0.9, 0.999))
        self.criterion = nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.discount = 0.9

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

        return board_tensor
    
    def get_best_move_and_val(self):
        # get legal moves
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None, 0.0
        
        # epsilon-greedy
        if random.random() < self.epsilon:
            rand = random.random()

            if rand < 0.6:
                # capture any piece possilbe
                capture_moves = [move for move in legal_moves if self.board.is_capture(move)]
                if capture_moves:
                    return random.choice(capture_moves), 0.0
            elif rand < 0.8:
                # move a piece that is under attack
                attacked_moves = []
                for move in legal_moves:
                    if self.board.is_attacked_by(not self.color, move.from_square):
                        attacked_moves.append(move)
                if attacked_moves:
                    return random.choice(attacked_moves), 0.0
                
            return random.choice(legal_moves), 0.0
        
        # get max q val for each legal move
        best_move = None
        best_value = -float('inf')
        for move in legal_moves:
            # apply move temporarily
            self.board.push(move)
            board_tensor = self.board_to_tensor(self.board).unsqueeze(0)

            # foward pass to get q value for each legal move
            with torch.no_grad():
                value = self.model.forward(board_tensor).item()

            # update best move
            if value > best_value:
                best_value = value
                best_move = move

            # undo move
            self.board.pop()

        return best_move, best_value
    

    
    def train_step(self, move, old_board, new_board, reward, value, opponent_reward=0):
        # get tensor of old board
        old_board_state = self.board_to_tensor(old_board).unsqueeze(0)
        
        with torch.no_grad():
            if new_board.is_game_over():
                target_q = reward - opponent_reward
            else:
                # next_board_state = self.board_to_tensor(self.board).unsqueeze(0)
                next_board_state = self.board_to_tensor(new_board).unsqueeze(0)
                next_max_q = self.model(next_board_state).item()
                target_q = (reward - opponent_reward) + self.discount * next_max_q

        predicted_q = self.model.forward(old_board_state)
        target_q = torch.tensor([[target_q]], dtype=torch.float32, device=device)

        # Update model
        self.optimizer.zero_grad()
        loss = self.criterion(predicted_q, target_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return reward - opponent_reward

    def getReward(self, old_state, new_state, move):
        reward = 0

        piece_values = {
            chess.PAWN: 100,
            chess.ROOK: 500,
            chess.KNIGHT: 300,
            chess.BISHOP: 300,
            chess.QUEEN: 900,
            chess.KING: 0
        }

        if (new_state.is_checkmate()):
            if new_state.turn != self.color:
                reward += 100000 # goood boyyyyy
            else:
                reward -= 100000
        elif (new_state.is_stalemate() or new_state.is_insufficient_material()
                or new_state.can_claim_draw() or new_state.can_claim_fifty_moves()
                or new_state.can_claim_threefold_repetition()):
                reward -= 100

        # Reward for capturing pieces
        captured_piece = old_state.piece_at(move.to_square)
        if captured_piece is not None:
            
            piece_value = piece_values.get(captured_piece.piece_type, 0)
            reward += piece_value

            # extra bonus if the captured piece was hanging (not defended)
            if not old_state.is_attacked_by(not self.color, move.to_square):
                reward += piece_value * 2.0

        # penalty for hanging its own pieces
        our_piece = new_state.piece_at(move.to_square)
        if our_piece and new_state.is_attacked_by(not self.color, move.to_square):
            if not new_state.is_attacked_by(self.color, move.to_square):
                reward -= piece_values.get(our_piece.piece_type, 0) * 0.8
    

        if new_state.is_check():
            reward += 15

        if move.promotion:
            reward += 800
        
        # smol penalty for each move to encourage faster wins (or losses)
        reward -= 25

        return reward
    
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
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    