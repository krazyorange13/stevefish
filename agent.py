import chess
import torch
import torch.nn as nn
import random

class Agent():
    def __init__(self, color=chess.WHITE):
        self.board = chess.Board()
        self.model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.color = color
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = 0.1
        self.discount = 0.9

    def board_to_tensor(self, board):
        board_tensor = torch.zeros(64, dtype=torch.float32)

        piece_to_value = {
            None: 0,
            chess.PAWN: 1,
            chess.ROOK: 2,
            chess.KNIGHT: 3,
            chess.BISHOP: 4,
            chess.QUEEN: 5,
            chess.KING: 6
        }

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                value = piece_to_value[piece.piece_type]

                # Negative for black pieces (or enemy pieces later on)
                if piece.color != self.color:
                    value = -value
                
                board_tensor[square] = value

        return board_tensor
    
    def get_best_move_and_val(self):
        # get legal moves
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None, 0.0
        
        # epsilon-greedy
        if random.random() < self.epsilon:
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
    

    
    def train_step(self, move, value):
        # get tensor of old board
        old_board = self.board
        old_board_state = self.board_to_tensor(self.board).unsqueeze(0)

        # make move
        self.board.push(move)

        # calculate reward
        reward = self.getReward(old_board, self.board, move)
        
        # the old prediction for the previous state should be updated based on reward
        # Q-learning update algorithm: Q(s, a) = Q(s, a) + alpha * (reward + discount * max_next_Q - Q(s, a))
        
        
        with torch.no_grad():
            if self.board.is_game_over():
                target_q = reward
            else:
                next_max_q = self.get_best_move_and_val()[1]
                target_q = reward + self.discount * next_max_q

        predicted_q = self.model.forward(old_board_state)
        target_q = torch.tensor([[target_q]], dtype=torch.float32)

        # Update model
        self.optimizer.zero_grad()
        loss = self.criterion(predicted_q, target_q)
        loss.backward()
        self.optimizer.step()

        return move

    def getReward(self, old_state, new_state, move):
        reward = 0

        if (new_state.is_checkmate()):
            reward += 100
        elif (new_state.is_stalemate() or new_state.is_insufficient_material()
                or new_state.can_claim_draw() or new_state.can_claim_fifty_moves()
                or new_state.can_claim_threefold_repetition()):
                reward += 10

        return reward