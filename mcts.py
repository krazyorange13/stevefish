from dataclasses import dataclass

import torch

from ttt2 import TTT2


@dataclass
class State:
    state: TTT2
    children: list[TTT2]
    score: float


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


if __name__ == "__main__":
    while True:
        ttt = get_board_from_player()
        print_ttt(ttt)
        print("normalized")
        print_ttt(normalize_ttt(ttt))
