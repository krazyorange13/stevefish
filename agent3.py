# tic tac toe rl self-play agent

import sys
import math
import random

from datetime import datetime

from collections import namedtuple, deque, Counter
from itertools import count

import struct
import ctypes

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from ttt import TTT

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


class Analysis:
    def __init__(self):
        self.steps = 0
        self.dump_rate = 10000

        self.games = []
        self.games_random = []
        self.losses = []

    def push_reward(self, reward):
        self.games.append(reward)

    def push_random_opp(self, reward):
        self.games_random.append(reward)

    def push_loss(self, loss):
        self.losses.append(loss)

    def monitor(self):
        self.steps += 1
        if self.steps % self.dump_rate == 0:
            self.dump()

    def dump(self):
        # version_num = (ctypes.c_int)(1)

        # games_buf = (ctypes.c_double * len(self.games))()
        # games_buf[:] = self.games
        # games_len = (ctypes.c_long)(len(self.games))

        # games_random_len = (ctypes.c_long)(len(self.games_random))
        # games_random_buf = (ctypes.c_double * len(self.games_random))()
        # games_random_buf[:] = self.games_random

        # losses_len = (ctypes.c_long)(len(self.losses))
        # losses_buf = (ctypes.c_double * len(self.losses))()
        # losses_buf[:] = self.losses

        version_num = struct.pack("<I", 2)
        games_len = struct.pack("<I", len(self.games))
        games_random_len = struct.pack("<I", len(self.games_random))
        losses_len = struct.pack("<I", len(self.losses))

        # can be optimized using ctypes but i want to keep things simple for now until i get it working
        games_buf = struct.pack("<" + "f" * len(self.games), *self.games)
        games_random_buf = struct.pack(
            "<" + "f" * len(self.games_random), *self.games_random
        )
        losses_buf = struct.pack("<" + "f" * len(self.losses), *self.losses)

        file_name = f"ttt_{datetime.now().date()}_{self.steps}.dat"
        with open(file_name, "wb") as file:
            # write version number
            file.write(version_num)
            # write length information for easy seeking
            file.write(games_len)
            file.write(games_random_len)
            file.write(losses_len)
            # write actual data
            file.write(games_buf)
            file.write(games_random_buf)
            file.write(losses_buf)


class IllegalMoveException(Exception):
    """Exception raised when an illegal move is attempted."""


class NoLegalMovesException(Exception):
    """Exception raised when no legal moves are available."""


class InvalidPlayerException(Exception):
    """Invalid player."""


class Environment:
    def __init__(self, nets):
        self.game = TTT()

        self.nets = nets
        random.shuffle(self.nets)

        self.policy_net_p = self.nets.index(policy_net) + 1
        self.target_net_p = self.nets.index(target_net) + 1

    def step(self, action, p):
        reward = 0
        next_state = None

        # illegal move! not good :(
        if not self.game.get_legal_moves_simple().flatten()[action]:
            # we should never get here, we're masking illegal moves in `optimize()`
            print(self.game.board == 0)
            print(action)
            raise IllegalMoveException()

        row, col = divmod(action, self.game.WIDTH)
        self.game.move(p, row, col)
        next_state = self.game.board.copy()

        if self.game.get_win(p):
            # hooray :D we won!
            reward = 1
        # elif self.game.get_win_next_turn():
        elif any(self.game.get_wins()):
            # any potential wins NEXT TURN (so the opponent)?
            # either we will lose, or the opponent will blunder and we can keep going,
            # but either way we want to give negative reward
            reward = -1
        elif self.game.get_draw():
            # draws are good too :) but we'll do neutral for now
            reward = 0.9
        else:
            reward = 0

        return reward, next_state


class DQN(nn.Module):
    N_OBSERVATIONS = 27  # three channel encoded
    N_ACTIONS = 9  # 3x3 board, Q value for each square

    def __init__(self):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten(0)
        self.layers = nn.Sequential(
            nn.Linear(DQN.N_OBSERVATIONS, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, DQN.N_ACTIONS),
        )

    def forward(self, x):
        return self.layers(x)


BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = 20000
TAU = 0.005
LR = 3e-4

eps_steps = 0


# https://medium.com/data-science/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b
# https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://miro.medium.com/v2/resize:fit:1100/format:webp/1*ibWj_Ym7JWhz551PrHTUkA.png
def train(n_episodes):
    print("start")

    for episode_i in range(n_episodes):
        # set up environment

        env = Environment([policy_net, target_net])

        first_step(env)

        for step_i in count():
            # replay memory gathers training sample by interacting with the environment
            done = step(env, memory, analysis)

            # compute loss and optimize networks on random training data
            optimize(optimizer, memory, analysis)

            # update target network toward policy network
            polyak()

            if done:
                print(end=".", flush=True)
                break

        # test against random bot
        if episode_i % 100 == 0:
            env = Environment(nets=[policy_net, target_net])
            first_step(env)
            for step_i in count():
                done = step(env, memory, analysis, random_opp=True)
                if done:
                    break

        # save model checkpoint
        if episode_i % 10000 == 0 and episode_i != 0:
            save_checkpoint(episode_i)

        # monitor training progress
        analysis.monitor()

    print()
    print("finish")


def first_step(env: Environment):
    if env.target_net_p != 1:
        return

    state = torch.from_numpy(env.game.board.copy()).flatten().unsqueeze(0)
    raw_state = torch.from_numpy(env.game.board)
    state = encode_board(state, env.target_net_p)
    action = greedy_action(state, raw_state, target_net)
    _, _ = env.step(action, env.target_net_p)


def step(env: Environment, memory: ReplayMemory, analysis: Analysis, random_opp=False):
    state_unencoded = torch.from_numpy(env.game.board.copy()).flatten().unsqueeze(0)
    state = encode_board(state_unencoded, env.policy_net_p)
    raw_state = torch.from_numpy(env.game.board)
    action = greedy_action(state, raw_state, policy_net)
    reward, _next_state = env.step(action, env.policy_net_p)
    next_state_unencoded = torch.from_numpy(_next_state).flatten().unsqueeze(0)
    raw_next_state = torch.from_numpy(env.game.board)
    done = env.game.get_done()
    done_opp = False

    if done:
        next_state = None
    else:
        next_state = encode_board(next_state_unencoded, env.target_net_p)
        action_opp = greedy_action(
            next_state, raw_next_state, target_net, just_random=random_opp
        )
        reward_opp, _next_state_opp = env.step(action_opp, env.target_net_p)
        next_state_opp_unencoded = (
            torch.from_numpy(_next_state_opp).flatten().unsqueeze(0)
        )
        next_state_opp = encode_board(next_state_opp_unencoded, env.policy_net_p)

        done_opp = env.game.get_done()
        next_state = next_state_opp

        # win if opp lost or lose if opp won
        if abs(reward_opp) == 1:
            reward = reward_opp * -1
        else:
            reward = reward_opp

    if not random_opp:
        for flip_x in [False, True]:
            for flip_y in [False, True]:
                aug_state_unencoded = augment_board(
                    state_unencoded, flip_x=flip_x, flip_y=flip_y
                )
                aug_state = encode_board(
                    aug_state_unencoded,
                    p=env.policy_net_p,
                )
                aug_next_state_unencoded = augment_board(
                    next_state_unencoded, flip_x=flip_x, flip_y=flip_y
                )
                aug_next_state = encode_board(
                    aug_next_state_unencoded,
                    p=env.policy_net_p,
                )
                memory.push(
                    aug_state,
                    torch.tensor([[action]]),
                    torch.tensor([reward]),
                    aug_next_state,
                )

    if done or done_opp:
        if not random_opp:
            analysis.push_reward(reward)
        else:
            analysis.push_random_opp(reward)
        return True

    return False


def optimize(optimizer, memory, analysis):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # next_state_batch = torch.stack(batch.next_state)

    # some next_states are finished! we don't want to run the target_net on them.
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # print(state_batch.shape)
    # print(action_batch.shape)
    # print(reward_batch.shape)
    # print(non_final_next_states.shape)

    # regenerate Q values for every state
    state_q_values = policy_net(state_batch)

    # print(state_q_values.shape)

    # select JUST the Q values of the actions we chose before (with argmax n stuff, remember?)
    # i'm pretty confident now in what .gather is doing here :)
    # instead of action_batch we could just use a legal mask and argmax but idk
    state_q_values = state_q_values.gather(1, action_batch)
    # state_q_values is now a long list of the best Q value for every state

    # print(state_q_values.shape)

    # okay i'm forming a foggy idea of what's going on with this part
    # we're using the net to predict one state ahead, and we'll train
    # the policy_net to predict this one state ahead value, so that
    # way it gets better at predicting into the future
    # i think i get it :D
    # still unclear though why we're using the target_net?
    # more stable or smth probably

    # illegal_masks = non_final_next_states != 0
    next_state_q_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        # TODO: do we need to do legal masking here??
        target_net_predictions = target_net(non_final_next_states).max(1).values
        # print(next_state_q_values.shape)
        # print(non_final_mask.shape)
        # print(target_net_predictions.shape)
        next_state_q_values[non_final_mask] = target_net_predictions
    # next_state_q_values is now a long list of the best Q value for every next_state

    # GAMMA helps fight DQN overestimation?
    # TODO: look at Double DQN (once we get the simple stuff figured out lol)
    expected_state_q_values = (next_state_q_values * GAMMA) + reward_batch
    expected_state_q_values.unsqueeze_(1)

    # ok idk what any of this is doing lol
    # calculate loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_q_values, expected_state_q_values)

    analysis.push_loss(loss.item())

    # optimize model
    optimizer.zero_grad()
    loss.backward()
    # in-place gradient clipping (i think prevents anything from getting too crazy)
    nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    # awesome
    optimizer.step()


def polyak():
    # gradual update of target network's weights toward policy network's
    for policy_param, target_param in zip(
        policy_net.parameters(), target_net.parameters()
    ):
        target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)


def greedy_action(
    state: torch.Tensor, raw_state: torch.Tensor, net: nn.Module, just_random=False
):
    global eps_steps
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * eps_steps / EPS_DECAY
    )
    eps_steps += 1

    # if True:
    if sample > eps_threshold and not just_random:
        # use net to get move
        # flatten is to convert 3x3 to 9
        X = state
        y = net(X.float()).squeeze()
        # legal mask for posterity
        illegal_mask = (raw_state != 0).flatten()

        if torch.all(illegal_mask):
            raise NoLegalMovesException()

        y[illegal_mask] = float("-inf")
        move = torch.argmax(y).item()

        return move

    else:
        # pick a random move
        actions = torch.nonzero((raw_state == 0).flatten())
        action = random.choice(actions).item()
        return action


def encode_board(x, p):
    # print("x:", x.shape)
    # x should be a 1x9 board
    # channel encode it!
    no = (x == 0).float()
    xs = (x == 1).float()
    os = (x == 2).float()
    # normalize the board so that the player and opponent are always the same
    if p == 1:
        board = torch.cat([no, xs, os], dim=1)
    elif p == 2:
        board = torch.cat([no, os, xs], dim=1)
    else:
        raise InvalidPlayerException()
    return board


def augment_board(x, flip_x=False, flip_y=False, rotations=0):
    # x should be a 1x9 board
    # rotations should be in range [0, 3]
    # i dont want to implement rotations :(
    # hopefully flips are good enough for now lol
    flips = []
    # i actually have no idea if these dimensions are correct :P
    if flip_x:
        flips.append(0)
    if flip_y:
        flips.append(1)
    return torch.flip(x.reshape([3, 3]), dims=flips).flatten().unsqueeze(0)


def save_checkpoint(episode_i):
    save_path = f"ttt_{datetime.now().date()}_{episode_i}.tar"
    torch.save(
        {
            "policy_net_state_dict": policy_net.state_dict(),
            "target_net_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "eps_steps": eps_steps,
        },
        save_path,
    )


if __name__ == "__main__":
    policy_net = DQN().to(device)
    target_net = DQN().to(device)

    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(500000)
    analysis = Analysis()  # watch rewards as training progresses

    try:
        load_path = sys.argv[sys.argv.index("--load") + 1]
        print(f"load {load_path}")
        checkpoint = torch.load(load_path, weights_only=True)
        policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        target_net.load_state_dict(checkpoint["target_net_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        eps_steps = checkpoint["eps_steps"]
        policy_net.train()
    except (ValueError, IndexError):
        pass

    try:
        run_name = sys.argv[sys.argv.index("--save") + 1]
    except (ValueError, IndexError):
        run_name = ""

    try:
        train(1000000)  # 1 million games
    except KeyboardInterrupt:
        print("\ncancel")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_name = "_" + run_name if run_name else ""
    save_path = f"ttt_{timestamp}{run_name}.tar"

    print(f"save {save_path}")
    torch.save(
        {
            "policy_net_state_dict": policy_net.state_dict(),
            "target_net_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "eps_steps": eps_steps,
        },
        save_path,
    )
