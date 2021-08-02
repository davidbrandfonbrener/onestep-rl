import torch
from torch import nn
from utils import MLP

class QMLP(nn.Module):
    def __init__(self, state_dim, action_dim, width, depth):
        super().__init__()
        self.net = MLP(input_shape=(state_dim + action_dim), output_dim=1,
                        width=width, depth=depth)

    def forward(self, s, a):
        x = torch.cat([
            torch.flatten(s, start_dim=1),
            torch.flatten(a, start_dim=1)
        ], axis=-1)
        return self.net(x)

class DoubleQMLP(nn.Module):
    def __init__(self, state_dim, action_dim, width, depth):
        super().__init__()
        self.net1 = MLP(input_shape=(state_dim + action_dim), output_dim=1,
                        width=width, depth=depth)
        self.net2 = MLP(input_shape=(state_dim + action_dim), output_dim=1,
                        width=width, depth=depth)

    def forward(self, s, a):
        x = torch.cat([
            torch.flatten(s, start_dim=1),
            torch.flatten(a, start_dim=1)
        ], axis=-1)
        q1 = self.net1(x)
        q2 = self.net2(x)
        return q1, q2
