"""任务塔：CTR 塔、CVR 塔、插补塔各一个，互不共参。"""

from torch import nn


class Tower(nn.Module):
    """MLP 塔，输出单个 logit。"""

    def __init__(self, in_dim: int, hidden=(64, 32), dropout: float = 0.0):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, int(h)), nn.ReLU(), nn.Dropout(dropout)]
            last = int(h)
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
