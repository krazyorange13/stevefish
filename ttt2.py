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

    def __init__(self, ttt: "TTT2 | torch.Tensor | None" = None):
        if isinstance(ttt, TTT2):
            self.b = ttt.b.clone().detach()
        elif isinstance(ttt, torch.Tensor):
            self.b = ttt.clone().detach()
        else:
            self.b = torch.zeros([1, 9])

        # self.board is a flat 1x9 tensor
        # player 1 (X) is represented by 1
        # player 2 (O) is represented by -1
        # an empty square is represented by 0

        # TODO multiple boards at once (batch dim)

    def mov(self, m, p):
        self.b.scatter_(1, m, p)

    def win(self, p):
        lins = self.b[:, self.WINS]
        _p = p.unsqueeze(1).unsqueeze(2)
        isp = lins == _p
        three = isp.all(dim=2)
        win = three.any(dim=1)
        # print(f"lins {lins.shape}: {lins}")
        # print(f"_p {_p.shape}: {_p}")
        # print(f"isp {isp.shape}: {isp}")
        # print(f"three {three.shape}: {three}")
        # print(f"win {win.shape}: {win}")
        return win

    def wina(self):
        p = torch.tensor([1, -1])
        lins = self.b[:, self.WINS]
        _p = p.view(1, 2, 1, 1)
        isp = lins.unsqueeze(1) == _p
        three = isp.all(dim=3)
        win = three.any(dim=2)
        # print(f"lins {lins.shape}: {lins}")
        # print(f"_p {_p.shape}: {_p}")
        # print(f"isp {isp.shape}: {isp}")
        # print(f"three {three.shape}: {three}")
        # print(f"win {win.shape}: {win}")
        return win

    def drw(self):
        return (self.b == 0).sum(dim=1) == 0

    def asp(self, p):
        return self.b * p

    def aug_reverse(self):
        return TTT2(torch.flip(self.b, [1]))

    def aug_flip_rows(self):
        b = torch.flip(self.b.reshape([1, 3, 3]), [2]).reshape([1, 9])
        return TTT2(b)

    def aug_flip_cols(self):
        b = torch.flip(self.b.reshape([1, 3, 3]), [1]).reshape([1, 9])
        return TTT2(b)

    def aug_transpose(self):
        b = self.b.reshape([3, 3]).T.unsqueeze(1).reshape([1, 9])
        return TTT2(b)

    def aug_rot_90(self):
        return self.aug_transpose().aug_flip_rows()

    def aug_rot_180(self):
        return self.aug_reverse()

    def aug_rot_270(self):
        return self.aug_transpose().aug_flip_cols()

    def get_legal_moves(self):
        return torch.nonzero(self.b == 0).flatten().tolist()

    def get_next_turn(self):
        xs = (self.b == 1).sum()
        os = (self.b == -1).sum()
        if os < xs:
            return -1
        else:
            return 1
