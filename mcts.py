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


if __name__ == "__main__":
    ttt = get_board_from_player()
    print_ttt(ttt)
    print("rot 90 deg")
    print_ttt(ttt.aug_transpose().aug_flip_rows())
    print("rot 180 deg")
    print_ttt(ttt.aug_reverse())
    print("rot 270 deg")
    print_ttt(ttt.aug_transpose().aug_flip_cols())
    # print("reverse")
    # print_ttt(ttt.aug_reverse())
    # print("flip rows")
    # print_ttt(ttt.aug_flip_rows())
    # print("flip cols")
    # print_ttt(ttt.aug_flip_cols())
    # print("transpose")
    # print_ttt(ttt.aug_transpose())
