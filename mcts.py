import math
import random
from dataclasses import dataclass

import torch

from ttt2 import TTT2


def get_legals(ttt):
    return torch.nonzero(ttt.b == 0).flatten()


def get_player_mov(ttt):
    legals = get_legals(ttt)
    while True:
        user = input("move [0-8]: ")
        if user.lower() in ["d", "done"]:
            return -1
        try:
            user = int(user)
        except ValueError:
            print("invalid choice >:(")
            continue
        if not (user >= 0 and user <= 8):
            print("invalid choice >:(")
            continue
        elif user not in legals:
            print("invalid choice >:(")
            continue
        else:
            break
    return user


def get_board_from_player():
    ttt = TTT2()
    while True:
        user = input("board [.xo]: ")
        if len(user) != 9:
            print("invalid entry >:(")
            continue
        if not all(c in [".", "x", "o"] for c in user):
            print("invalid entry >:(")
            continue
        else:
            break
    for i, c in enumerate(user):
        if c == "x":
            ttt.b[0][i] = 1
        elif c == "o":
            ttt.b[0][i] = -1
        else:
            ttt.b[0][i] = 0
    return ttt


def print_ttt(ttt):
    for i, n in enumerate(ttt.b.flatten()):
        if int(n) == 1:
            c = "x"
        elif int(n) == -1:
            c = "o"
        else:
            c = "."
        print(end=c)
        if i % 3 == 2:
            print()


def print_ttt_(ttt):
    for i, n in enumerate(ttt.b.flatten()):
        if int(n) == 1:
            c = "x"
        elif int(n) == -1:
            c = "o"
        else:
            c = "."
        print(end=c)
    print()


def normalize_ttt(ttt: TTT2):
    def _is_corner_x(ttt: TTT2):
        return (
            ttt.b[0][0] != 1 or ttt.b[0][2] != 1 or ttt.b[0][6] != 1 or ttt.b[0][8] != 1
        )

    def _is_edge_x(ttt: TTT2):
        return (
            ttt.b[0][1] != 1 or ttt.b[0][3] != 1 or ttt.b[0][5] != 1 or ttt.b[0][7] != 1
        )

    if _is_corner_x(ttt):
        print("is corner", ttt.b.tolist())
        while ttt.b[0][0] != 1:
            print("\trot 90")
            ttt = ttt.aug_rot_90()
    elif _is_edge_x(ttt):
        print("is edge", ttt.b.tolist())
        while ttt.b[0][1] != 1:
            print("\trot 90")
            ttt = ttt.aug_rot_90()

    return ttt


class Node:
    C = math.sqrt(2)

    def __init__(self):
        self.state: TTT2 = TTT2()
        self.parent: Node | None = None
        self.children: list[Node] = []
        self.leaf: bool | None = None
        self.T = 0  # total rewards
        self.N = 0  # total visits

    def get_ucb_score(self):
        # use UCB (Upper Confidence Bound) formula
        if self.N == 0:
            return float("inf")

        top_node = self.parent if self.parent is not None else self

        #       exploration        exploitation
        return (self.T / self.N) + self.C * math.sqrt(math.log(top_node.N) / self.N)

    def sex(self):
        if self.leaf is True:
            return

        legal_moves = self.state.get_legal_moves()
        for legal_move in legal_moves:
            node = Node()
            node.state = TTT2(self.state)
            _m = legal_move
            _p = self.state.get_next_turn()
            m = torch.tensor([[_m]])
            p = torch.tensor([[_p]], dtype=torch.float)
            node.state.mov(m, p)
            node.parent = self
            self.children.append(node)

        self.leaf = len(legal_moves) == 0

    def explore(self, player):
        curr = self

        while len(curr.children) != 0:
            max_u = max(child.get_ucb_score() for child in curr.children)
            max_childs = [
                i
                for i, child in enumerate(curr.children)
                if child.get_ucb_score() == max_u
            ]
            if len(max_childs) == 0:
                print("oh no! :( len(max_childs) == 0")
            child = curr.children[random.choice(max_childs)]
            curr = child

        if curr.N < 1:
            curr.T = curr.T + curr.rollout(player)
        else:
            curr.sex()
            if len(curr.children) != 0:
                curr = random.choice(curr.children)
            curr.T = curr.T + curr.rollout(player)

        curr.N += 1

        prnt = curr

        while prnt.parent:
            prnt = prnt.parent
            prnt.N += 1
            prnt.T = prnt.T + curr.T

    def rollout(self, player):
        if self.leaf is True:
            return 0

        v = 0
        done = False
        _state = TTT2(self.state)
        while not done:
            if _state.win(torch.tensor([player])):
                v = 1
                break
            elif _state.win(torch.tensor([player * -1])):
                v = 0
                break
            elif _state.drw():
                v = 0.5
                break
            _m = random.choice(_state.get_legal_moves())
            _p = _state.get_next_turn()
            m = torch.tensor([[_m]])
            p = torch.tensor([[_p]], dtype=torch.float)
            _state.mov(m, p)

        return v

    def next(self):
        if len(self.children) == 0:
            print("game done")
            return

        max_n = max(child.N for child in self.children)
        max_childs = [child for child in self.children if child.N == max_n]
        if len(max_childs) == 0:
            print("no max childs :(")
        max_child = random.choice(max_childs)
        return max_child


def policy(node, player):
    for i in range(10000):
        node.explore(player)

    next_node = node.next()

    next_node.parent = None

    return next_node


if __name__ == "__main__":
    ttt = TTT2()
    while True:
        print_ttt(ttt)
        _m = get_player_mov(ttt)
        _p = 1
        m = torch.tensor([[_m]])
        p = torch.tensor([[_p]], dtype=torch.float)
        ttt.mov(m, p)

        node = Node()
        node.state = ttt
        next_node = policy(node, -1)
        ttt = next_node.state
    # while True:
    #     ttt = get_board_from_player()
    #     print_ttt(ttt)
    #     print("normalized")
    #     print_ttt(normalize_ttt(ttt))
    pass
