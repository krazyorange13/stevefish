import math
import random

import torch

from ttt2 import TTT2


class Node:
    C = math.sqrt(2)

    def __init__(self, parent: "Node | None", state: TTT2):
        self.parent: "Node | None" = parent
        self.children: list[Node] = []
        self.state: TTT2 = state
        self.T = 0  # rewards
        self.N = 0  # visits

    def select(self):
        if len(self.children) == 0:
            return None

        max_ucb = max(child.ucb_score() for child in self.children)
        max_children = [
            child for child in self.children if child.ucb_score() == max_ucb
        ]
        child = random.choice(max_children)
        return child

    def expand(self):
        if self.state.done():
            return
        moves_l = self.state.get_legal_moves()
        for move_l in moves_l:
            _state = TTT2(self.state)
            _m = torch.tensor([[move_l]])
            _p = torch.tensor([[_state.get_next_turn()]], dtype=torch.float)
            _state.mov(_m, _p)
            node = Node(self, _state)
            self.children.append(node)

    def simulate(self):
        _state = TTT2(self.state)
        if self.state.done():
            if _state.win(torch.tensor([1])):
                return 1
            elif _state.win(torch.tensor([-1])):
                return -1
            elif _state.drw():
                return 0
        while not _state.done():
            m = random.choice(_state.get_legal_moves())
            p = _state.get_next_turn()
            _m = torch.tensor([[m]])
            _p = torch.tensor([[p]], dtype=torch.float)
            _state.mov(_m, _p)
            if _state.win(torch.tensor([1])):
                return 1
            elif _state.win(torch.tensor([-1])):
                return -1
            elif _state.drw():
                return 0
        return 0

    def backprop(self, reward):
        self.T += reward
        self.N += 1
        if self.parent:
            self.parent.backprop(reward)

    def reward(self, result, player):
        if result == player:
            return 1
        elif result == 0:
            return 0.5
        else:
            return 0

    def ucb_score(self):
        if self.N == 0:
            return float("inf")

        top_node = self
        if self.parent is not None:
            top_node = self.parent

        return (self.T / self.N) + self.C * math.sqrt(math.log(top_node.N) / self.N)

    def __str__(self):
        return f"[{self.T}/{self.N}] {self.state}"


def print_node_tree(node: Node):
    _print_node_tree(node, 0)


def _print_node_tree(node: Node, indent):
    if node.N == 0:
        return
    print(f"{'\t' * indent}{node}")
    for child in node.children:
        _print_node_tree(child, indent + 1)


def print_best_node_tree(node: Node):
    _print_best_node_tree(node, 0)


def _print_best_node_tree(node: Node, indent):
    print(f"{'\t' * indent}{node}")
    if len(node.children) == 0:
        return
    max_n = max(node.N for node in node.children)
    max_children = [child for child in node.children if child.N == max_n]
    for max_child in max_children:
        _print_best_node_tree(max_child, indent + 1)


def policy(node: Node):
    curr_node = node
    while True:
        next_node = curr_node.select()
        if next_node is None:
            break
        else:
            curr_node = next_node
    curr_node.expand()
    leaf_node = curr_node.select()
    if leaf_node is None:
        leaf_node = curr_node
    result = leaf_node.simulate()
    reward = leaf_node.reward(result, 1)
    leaf_node.backprop(reward)


if __name__ == "__main__":
    mcts = Node(None, TTT2())
    for i in range(100000):
        policy(mcts)
    print_best_node_tree(mcts)
