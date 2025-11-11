import sys
import torch

import torch

from ttt2 import TTT2
from agent4 import DQN


def get_legals(ttt):
    return torch.nonzero(ttt.b == 0).flatten()


def get_player_mov(ttt):
    legals = get_legals(ttt)
    while True:
        user = input("move [0-8]: ")
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


def get_policy_mov(ttt, net, policy_p):
    X = ttt.asp(policy_p)
    y = net(X)
    _y = y
    _y[ttt.b != 0] = float("-inf")
    action = torch.argmax(_y, dim=1).unsqueeze(1)
    return action


def print_ttt(ttt):
    for i, n in enumerate(ttt.b.flatten()):
        print(end=_ttt_c(n))
        if i % 3 == 2:
            print()


def _ttt_c(n):
    if int(n) == 1:
        return "x"
    elif int(n) == -1:
        return "o"
    else:
        return "."


def play(policy_net):
    ttt = TTT2()

    while True:
        user = input("X or O? ").lower()
        if user in ["x", "o"]:
            break
        else:
            print("invalid choice >:(")
            continue

    if user == "x":
        player_p = 1
        policy_p = -1
    elif user == "o":
        player_p = -1
        policy_p = 1
    else:
        print("invalid user choice")
        sys.exit(2)

    if player_p == 1:
        print_ttt(ttt)
        user = get_player_mov(ttt)
        m = torch.tensor([[user]])
        p = torch.tensor([[player_p]], dtype=torch.float)
        ttt.mov(m, p)

    while True:
        move = get_policy_mov(ttt, policy_net, policy_p)
        m = move
        p = torch.tensor([policy_p], dtype=torch.float).unsqueeze(1)
        ttt.mov(m, p)

        if ttt.win(torch.tensor([policy_p])):
            print_ttt(ttt)
            print("bro you lost tic tac toe to a neural net 💀🥀")
            break
        elif ttt.drw():
            print_ttt(ttt)
            print("bro you got a draw with the neural net 😐")
            break

        print_ttt(ttt)
        user = get_player_mov(ttt)
        m = torch.tensor([user]).unsqueeze(1)
        p = torch.tensor([player_p], dtype=torch.float).unsqueeze(1)
        ttt.mov(m, p)

        if ttt.win(torch.tensor([player_p])):
            print_ttt(ttt)
            print("bro ur such a tryhard go easy on the poor neural net 😞")
            break
        elif ttt.drw():
            print_ttt(ttt)
            print("bro you got a draw with the neural net 😐")
            break


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("wrong usage dum dum")
        sys.exit(1)

    path = sys.argv[1]
    policy_net = DQN()
    policy_net.load_state_dict(torch.load(path, weights_only=True))
    policy_net.eval()

    while True:
        play(policy_net)
        while True:
            user = input("play again? ").lower()
            if user in ["y", "yes"]:
                break
            elif user in ["n", "no"]:
                sys.exit(0)
            else:
                print("invalid choice >:(")
