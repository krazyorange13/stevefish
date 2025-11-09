import torch


class TTT2:
    WINS = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]
    )

    def __init__(self):
        self.b = torch.zeros([2, 9])

        # self.board is a flat 1x9 tensor
        # player 1 (X) is represented by 1
        # player 2 (O) is represented by -1
        # an empty square is represented by 0

        # TODO board augmentations
        # TODO multiple boards at once (batch dim)

    def mov(self, m, p):
        self.b.scatter_(1, m.unsqueeze(1), p.unsqueeze(1))

    def win(self, p):
        lins = self.b[:, self.WINS]
        _p = p.unsqueeze(1).unsqueeze(2)
        isp = lins == _p
        three = isp.all(dim=2)
        win = three.any(dim=1)
        print(f"lins {lins.shape}: {lins}")
        print(f"_p {_p.shape}: {_p}")
        print(f"isp {isp.shape}: {isp}")
        print(f"three {three.shape}: {three}")
        print(f"win {win.shape}: {win}")
        return win

    def wina(self):
        p = torch.tensor([1, -1])
        lins = self.b[:, self.WINS]
        _p = p.view(1, 2, 1, 1)
        isp = lins.unsqueeze(1) == _p
        three = isp.all(dim=3)
        win = three.any(dim=2)
        print(f"lins {lins.shape}: {lins}")
        print(f"_p {_p.shape}: {_p}")
        print(f"isp {isp.shape}: {isp}")
        print(f"three {three.shape}: {three}")
        print(f"win {win.shape}: {win}")
        return win

    def asp(self, p):
        return self.b * p

    def aug(self):
        # TODO board augmentations
        pass
