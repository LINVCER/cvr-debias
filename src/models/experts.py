"""专家层：MMoE 门控专家与因果解耦专家（C3）。

解耦专家层的要点是用掩码把专家按因果角色隔开：
- CTR 门控只看倾向专属专家与共享专家；
- CVR / 插补门控只看转化专属专家与共享专家。
这样 CTR 任务的梯度不再直接塑造 CVR 表征，对应 ESCM² 的潜在独立优先（PIP）要求。
"""

import torch
from torch import nn


class MLPExpert(nn.Module):
    """单专家：两层 MLP。"""

    def __init__(self, in_dim: int, expert_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, expert_dim),
            nn.ReLU(),
            nn.Linear(expert_dim, expert_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class GatedExperts(nn.Module):
    """MMoE 式门控专家：每个任务一套门控，专家池全共享。"""

    def __init__(self, in_dim: int, expert_dim: int, n_experts: int = 8, tasks=("ctr", "cvr")):
        super().__init__()
        self.tasks = tuple(tasks)
        self.experts = nn.ModuleList([MLPExpert(in_dim, expert_dim) for _ in range(n_experts)])
        self.gates = nn.ModuleDict({t: nn.Linear(in_dim, n_experts) for t in self.tasks})

    def forward(self, x: torch.Tensor) -> dict:
        e = torch.stack([f(x) for f in self.experts], dim=1)  # [B, E, D]
        out = {}
        for t in self.tasks:
            w = torch.softmax(self.gates[t](x), dim=-1).unsqueeze(-1)
            out[t] = (w * e).sum(dim=1)
        return out


class DisentangledExperts(nn.Module):
    """因果解耦专家层：专家池划为倾向专属、转化专属、共享三组。"""

    def __init__(self, in_dim: int, expert_dim: int, n_p: int = 2, n_c: int = 2, n_s: int = 4):
        super().__init__()
        if n_p + n_c + n_s <= 0:
            raise ValueError("专家总数必须为正")
        self.n_p, self.n_c, self.n_s = n_p, n_c, n_s
        n_total = n_p + n_c + n_s
        self.experts = nn.ModuleList([MLPExpert(in_dim, expert_dim) for _ in range(n_total)])

        mask_ctr = torch.zeros(n_total)
        mask_ctr[:n_p] = 1.0
        mask_ctr[n_p + n_c:] = 1.0
        mask_cvr = torch.zeros(n_total)
        mask_cvr[n_p:] = 1.0
        self.register_buffer("mask_ctr", mask_ctr)
        self.register_buffer("mask_cvr", mask_cvr)

        self.gate_ctr = nn.Linear(in_dim, n_total)
        self.gate_cvr = nn.Linear(in_dim, n_total)

    def forward(self, x: torch.Tensor) -> dict:
        """返回任务表征，以及给正交正则用的两组专属专家输出。

        :param x: 共享特征向量，[B, in_dim]
        :return: {ctr, cvr, u_p, u_c}
        """
        e = torch.stack([f(x) for f in self.experts], dim=1)  # [B, n_total, D]
        out = {}
        for name, gate, mask in (
            ("ctr", self.gate_ctr, self.mask_ctr),
            ("cvr", self.gate_cvr, self.mask_cvr),
        ):
            logits = gate(x).masked_fill(mask == 0, float("-inf"))
            w = torch.softmax(logits, dim=-1).unsqueeze(-1)
            out[name] = (w * e).sum(dim=1)
        # 两组专属专家的输出单独返回，给正交正则用
        out["u_p"] = e[:, : self.n_p].flatten(1)
        out["u_c"] = e[:, self.n_p : self.n_p + self.n_c].flatten(1)
        return out
