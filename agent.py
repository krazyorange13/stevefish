import chess
import torch
import torch.nn as nn
import random

class Agent:
    def __init__(self):
        self.board = chess.Board()
        self.model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = 0.1

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
                if piece.color == chess.BLACK:
                    value = -value
                
                board_tensor[square] = value

        return board_tensor
    
    def get_best_move(self):
        # get legal moves
        legal_moves = self.board.legal_moves
        if not legal_moves:
            return None
        
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # get max q val for each legal move
        best_move = None
        best_value = -float('inf')
        for move in legal_moves:
            # apply move temporarily
            self.board.push(move)
            board_tensor = self.board_to_tensor(self.board)

            # foward pass to get q value for each legal move
            with torch.no_grad():
                value = self.model.forward(board_tensor).item()

            # update best move
            if value > best_value:
                best_value = value
                best_move = move

            # undo move
            self.board.pop()

        return best_move

    
    def train_step(self, move):
        # get tensor of old board
        old_board_state = self.board_to_tensor(self.board)

        # make move
        self.board.push(move)

        # get tensor of new board
        new_board_state = self.board_to_tensor(self.board)

        # calculate reward
        reward = self.getReward(old_board_state, new_board_state, move)
        
        # the old prediction for the previous state should be updated based on reward
        predicted_q = self.model.forward(old_board_state)
        target_q = reward

        # Update model
        self.optimizer.zero_grad()
        loss = self.criterion(predicted_q, target_q)
        loss.backward()
        self.optimizer.step()

    def getReward(self, old_state, new_state, move):
        # TODO: implement reward function
        return 0