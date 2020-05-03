import torch
import torch.nn as nn
import numpy as np


class Linear(nn.Module):
    def __init__(self, device):
        super(Linear, self).__init__()

        self.device = device

        self.l1_dist = []
        self.l2_dist = []

        self.l1_inp = []
        self.l2_inp = []

        self.l1 = nn.Linear(784, 48)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(48, 24)
        self.r2 = nn.ReLU()
        self.l3 = nn.Linear(24, 10)

    def forward(self, inp):
        x = inp.view(inp.size(0), -1)
        x = self.l1(x)

        x1 = x.detach()
        self.l1_dist.append((x1.mean(), x1.std()))
        self.l1_inp = x1

        x = self.r1(x)
        x = self.l2(x)

        x2 = x.detach()
        self.l2_dist.append((x2.mean(), x2.std()))
        self.l2_inp = x2

        x = self.r2(x)
        x = self.l3(x)

        return x
