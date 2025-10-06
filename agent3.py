# tic tac toe rl self-play agent

import math
import random

from collections import namedtuple, deque
from itertools import count

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

device = "cpu"
if torch.accelerator.is_available():
    if accelerator := torch.accelerator.current_accelerator():
        device = accelerator.type

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.layers(x)


class TTTGame:
    def __init__(self):
        self.board = [0] * 9
        # 0 1 2
        # 3 4 5
        # 6 7 8
        self.turn = 1
        # X is 1, O is 2

    def reset(self):
        # basically __init__
        self.board = [0] * 9
        self.turn = 1

    def do_move(self, p, pos):
        if not self.is_legal_move(p, pos):
            return False

        self.board[pos] = p
        self.turn = (p % 2) + 1

        return True

    def is_legal_move(self, p, pos):
        return self.turn == p and self.board[pos] == 0 and not self.get_win()

    def get_legal_moves(self):
        return [i for i, x in enumerate(self.board) if x == 0]

    def get_tensor(self, device):
        # 3x3x2 one-hot encoded
        # first 3x3 layer is Xs
        # second 3x3 layer is Os
        # each layer has 1 if piece there, else 0
        xs = [[self.board[y * 3 + x] == 1 for x in range(3)] for y in range(3)]
        os = [[self.board[y * 3 + x] == 2 for x in range(3)] for y in range(3)]
        return torch.tensor([xs, os], device=device, dtype=torch.float)
        # i think pytorch has a function for this but whatever

    def get_win(self):
        if self.check_win(1):
            return 1
        if self.check_win(2):
            return 2
        else:
            return 0

    def get_draw(self):
        # no winner and board full
        return not self.get_win() and 0 not in self.board

    def check_win(self, p):
        return (
            self._check_win_rows(p)
            or self._check_win_cols(p)
            or self._check_win_diags(p)
        )

    def _check_win_rows(self, p):
        for i in range(3):
            i *= 3
            if p == self.board[i] == self.board[i + 1] == self.board[i + 2]:
                return True
        return False

    def _check_win_cols(self, p):
        for i in range(3):
            if p == self.board[i] == self.board[i + 3] == self.board[i + 6]:
                return True
        return False

    def _check_win_diags(self, p):
        return (
            p == self.board[0] == self.board[4] == self.board[8]
            or p == self.board[2] == self.board[4] == self.board[6]
        )

    def step(self, action, p, device):
        action = action.tolist()

        next_state = None
        reward = 0
        done = 0

        if action.count(True) != 1:
            # invalid move! we should only have one True
            reward = -100
            return next_state, reward, done

        move = action.index(True)
        self.do_move(p, move)

        next_state = self.get_tensor(device=device)

        win = self.get_win()
        if win:
            if win == p:
                reward = 1  # win :)
            else:
                reward = -1  # loss :(

        if self.get_draw():
            # we still reward this bc perfect ttt is a draw
            reward = 0  # draw :)
            # idk i'll try leaving this at zero actually

        # true if game over
        done = self.get_win() or self.get_draw()

        return next_state, reward, done


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

game = TTTGame()
n_observations = 3 * 3 * 2
n_actions = 3 * 3

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(game: TTTGame):
    global steps_done

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            X = torch.flatten(game.get_tensor(device=device))
            y = policy_net(X)
            # convert to boolean mask, 1 if greater than than zero, else 0
            y = torch.gt(y, 0)
            return y
    else:
        # pick a random legal move
        moves = game.get_legal_moves()
        m = random.choice(moves)
        y = [False] * 9
        y[m] = True
        y = torch.tensor(y, dtype=torch.bool)
        return y


episode_results = []


def plot(show_result=False):
    plt.figure(1)
    results_t = torch.tensor(episode_results, dtype=torch.int)
    if show_result:
        plt.title("result")
    else:
        plt.clf()
        plt.title("training...")
    plt.xlabel("episode")
    plt.ylabel("result")
    plt.plot(results_t.numpy())
    plt.pause(0.001)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # mask for all transitions that weren't the last one
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # i'm really not sure what is going on here :P
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # idk if this will work
        next_state_values[non_final_mask] = torch.gt(
            target_net(non_final_next_states), 1
        )

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    print(end=".")

    state = TTTGame()
    state_t = state.get_tensor(device=device)

    for t in count():
        action = select_action(state)
        next_state, reward, done = state.step(action, p=1, device=device)

        if done:
            next_state = None
        else:
            # TODO: right now it's just doing a random move!!!
            # we don't want that. although since tic-tac-toe is so simple, it might learn something lol
            m = random.choice(state.get_legal_moves())
            action = [False] * 9
            action[m] = True
            action = torch.tensor(action, device=device, dtype=torch.bool)
            next_state, reward, done = state.step(action, p=2, device=device)
            next_state = torch.tensor(next_state, device=device, dtype=torch.bool)

        memory.push(state, action, next_state, reward)

        optimize_model()

        # i think this slowly adjusts the target net to be similar, but behind, the policy net
        # that way it's not training on itself exactly
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            if state.get_win() == 1:
                result = 1
                # won :)
            elif state.get_win() == 2:
                # lost :(
                result = -1
            else:
                # includes draw
                result = 0
            episode_results.append(result)
            plot()
            break

print("complete")
plot(show_result=True)
plt.ioff()
plt.show()
