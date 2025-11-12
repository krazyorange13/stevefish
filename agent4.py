import math
import random
from dataclasses import dataclass
from collections import deque
from itertools import count

import torch
import torch.nn as nn

import ttt2


@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor | None

    def _ttt_c(self, n):
        if int(n) == 1:
            return "x"
        elif int(n) == -1:
            return "o"
        else:
            return "."

    def _ttt_1(self, ttt):
        if ttt is not None:
            return (
                self._ttt_c(ttt[0][0]) + self._ttt_c(ttt[0][1]) + self._ttt_c(ttt[0][2])
            )
        else:
            return "   "

    def _ttt_2(self, ttt):
        if ttt is not None:
            return (
                self._ttt_c(ttt[0][3]) + self._ttt_c(ttt[0][4]) + self._ttt_c(ttt[0][5])
            )
        else:
            return "nun"

    def _ttt_3(self, ttt):
        if ttt is not None:
            return (
                self._ttt_c(ttt[0][6]) + self._ttt_c(ttt[0][7]) + self._ttt_c(ttt[0][8])
            )
        else:
            return "   "

    def __str__(self):
        a = self.action[0].item()
        r = self.reward[0].item()
        s1 = f"  {self._ttt_1(self.state)}        {self._ttt_1(self.next_state)}"
        s2 = f"T({self._ttt_2(self.state)}, {a}, {r}, {self._ttt_2(self.next_state)})"
        s3 = f"  {self._ttt_3(self.state)}        {self._ttt_3(self.next_state)}"
        return "\n".join([s1, s2, s3]) + "\n"


class ReplayMemory:
    def __init__(self, cap):
        self.mem = deque([], maxlen=cap)

    def push(self, T: Transition):
        self.mem.append(T)
        T_ = Transition(
            torch.flip(T.state, [1]),
            torch.tensor([[8 - T.action.item()]]),
            T.reward.clone().detach(),
            torch.flip(T.next_state, [1]) if T.next_state is not None else None,
        )
        self.mem.append(T_)

    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)

    def __len__(self):
        return len(self.mem)


class Analysis:
    def __init__(self):
        self.transitions = []
        self.eps = []
        self.loss = []

    def push(self, transition, eps, loss):
        self.transitions.append(transition)
        self.eps.append(eps)
        self.loss.append(loss)

    def dump(self):
        self.transitions = []
        self.eps = []
        self.loss = []
        # try:
        #     address = ("localhost", 6000)
        #     conn = Client(address, authkey=b"agent4pypasswrod")
        #     conn.send({"load": "transitions", "data": self.transitions})
        #     conn.send({"load": "eps", "data": self.eps})
        #     conn.send({"load": "loss", "data": self.loss})
        #     self.transitions = []
        #     self.eps = []
        #     self.loss = []
        # except ConnectionRefusedError:
        #     pass


class DQN(nn.Module):
    # ez pz no encoding
    IN = 9
    OUT = 9

    def __init__(self):
        super(DQN, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(DQN.IN, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, DQN.OUT),
        )

    def forward(self, x):
        return self.seq(x)


class System:
    BATCH_SIZE = 32
    LR = 3e-4
    GAMMA = 0.9
    TAU = 0.005
    EPS_START = 0.9
    EPS_END = 0.1
    EPS_DECAY = 500_000
    REPLAYMEM_SIZE = 50_000
    DUMP_RATE = 50_000

    def __init__(self):
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=self.LR, amsgrad=True
        )
        self.memory = ReplayMemory(self.REPLAYMEM_SIZE)
        self.analysis = Analysis()
        self.eps_steps = 0
        self.eps_threshold = self.EPS_START
        self.loss = 0

    def train(self, n_episodes):
        print("start")

        for episode_i in range(n_episodes + 1):
            ttt = ttt2.TTT2()
            self.first_step(ttt)
            for step_i in count():
                done = self.step(ttt)
                self.optimize()
                self.polyak()
                if done:
                    print(end=".", flush=True)
                    break

            if episode_i % self.DUMP_RATE == 0 and episode_i != 0:
                path = f"ttt2_{episode_i}.pth"
                torch.save(self.policy_net.state_dict(), path)

                for i in range(
                    len(self.analysis.transitions) - 15, len(self.analysis.transitions)
                ):
                    print("  eps", self.analysis.eps[i])
                    print("  lss", self.analysis.loss[i])
                    print(self.analysis.transitions[i])
                self.analysis.dump()

        print("done")

    def first_step(self, ttt):
        if random.randint(0, 1):
            _state = ttt.b.clone().detach()
            _action = self.select_action(_state, self.target_net, noeps=True)
            ttt.mov(_action, torch.tensor([-1], dtype=torch.float).unsqueeze(1))

    def step(self, ttt: ttt2.TTT2):
        state = ttt.b.clone().detach()
        action = self.select_action(state, self.policy_net)
        ttt.mov(action, torch.tensor([1], dtype=torch.float).unsqueeze(1))
        if ttt.win(torch.tensor([1])):
            reward = torch.tensor([[1]])
            next_state = None
            T = Transition(state, action, reward, next_state)
            self.memory.push(T)
            self.analysis.push(T, self.eps_threshold, self.loss)
            return True
        elif ttt.drw():
            reward = torch.tensor([[0]])
            next_state = None
            T = Transition(state, action, reward, next_state)
            self.memory.push(T)
            self.analysis.push(T, self.eps_threshold, self.loss)
            return True
        else:
            # TODO
            # TODO check for careful use of TTT2.asp
            # TODO
            _state = ttt.asp(-1).clone().detach()
            _action = self.select_action(_state, self.target_net, noeps=True)
            ttt.mov(_action, torch.tensor([-1], dtype=torch.float).unsqueeze(1))
            if ttt.win(torch.tensor([-1])):
                reward = torch.tensor([[-1]])
                next_state = None
                T = Transition(state, action, reward, next_state)
                self.memory.push(T)
                self.analysis.push(T, self.eps_threshold, self.loss)
                return True
            elif ttt.drw():
                reward = torch.tensor([[0]])
                next_state = None
                T = Transition(state, action, reward, next_state)
                self.memory.push(T)
                self.analysis.push(T, self.eps_threshold, self.loss)
                return True
            else:
                reward = torch.tensor([[0]])
                next_state = ttt.b.clone().detach()
                T = Transition(state, action, reward, next_state)
                self.memory.push(T)
                self.analysis.push(T, self.eps_threshold, self.loss)

    def select_action(self, b: torch.Tensor, net: DQN, noeps=False):
        sample = random.random()
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1.0 * self.eps_steps / self.EPS_DECAY
        )
        self.eps_steps += 1

        if sample > self.eps_threshold or noeps:
            # use net
            with torch.no_grad():
                X = b  # .clone().detach()
                y = net(X)
                _y = y  # .clone().detach()
                _y[b != 0] = float("-inf")
                action = torch.argmax(_y, dim=1).unsqueeze(1)
                return action
        else:
            # random
            _y = torch.rand([1, 9])
            _y[b != 0] = float("-inf")
            action = torch.argmax(_y, dim=1).unsqueeze(1)
            return action

    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)

        state_batch = torch.cat([T.state for T in transitions])
        action_batch = torch.cat([T.action for T in transitions])
        reward_batch = torch.cat([T.reward for T in transitions])

        non_final_mask = torch.tensor([T.next_state is not None for T in transitions])
        non_final_next_states = torch.cat(
            [T.next_state for T in transitions if T.next_state is not None]
        )
        non_final_illegal_masks = torch.cat(
            [T.next_state != 0 for T in transitions if T.next_state is not None]
        )

        state_q_values = self.policy_net(state_batch)
        state_q_values = state_q_values.gather(1, action_batch)

        next_state_q_values = torch.zeros(self.BATCH_SIZE)
        with torch.no_grad():
            # TODO
            # TODO need to use TTT2.asp? we're training on these next_states but idk if we're the right player
            # TODO
            target_net_predictions = self.target_net(non_final_next_states)
            target_net_predictions[non_final_illegal_masks] = float("-inf")
            target_net_predictions = target_net_predictions.max(1).values
            next_state_q_values[non_final_mask] = target_net_predictions

        expected_state_q_values = (
            next_state_q_values * self.GAMMA
        ) + reward_batch.squeeze()
        expected_state_q_values.unsqueeze_(1)

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_q_values, expected_state_q_values)
        self.loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def polyak(self):
        for policy_param, target_param in zip(
            self.policy_net.parameters(), self.target_net.parameters()
        ):
            target_param.data.copy_(
                self.TAU * policy_param.data + (1 - self.TAU) * target_param.data
            )


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    seed = None
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    system = System()
    system.train(1_000_000)
