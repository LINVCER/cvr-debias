"""多任务损失加权。

四项损失尺度差异极大（CTCVR 正样本率仅 0.021%），因此 log_var 参数化并初始化为 0，
否则前几百步就数值溢出。
"""

import torch
from torch import nn


class UncertaintyWeighting(nn.Module):
    """同方差不确定性加权（C4）：权重由可学习的 log σ_k 决定。"""

    def __init__(self, task_names, init: float = 0.0):
        super().__init__()
        self.task_names = list(task_names)
        self.log_var = nn.Parameter(torch.full((len(self.task_names),), float(init)))

    def forward(self, losses: dict) -> tuple:
        """按不确定性加权汇总各项损失。

        :param losses: 任务名到标量损失的字典
        :return: (总损失, 各任务当前权重明细)
        """
        total = 0.0
        detail = {}
        for i, name in enumerate(self.task_names):
            if name not in losses:
                continue
            log_var = self.log_var[i]
            total = total + torch.exp(-log_var) * losses[name] + 0.5 * log_var
            detail[f"w_{name}"] = torch.exp(-log_var).detach()
        return total, detail


class FixedWeighting(nn.Module):
    """手工固定权重：去 C4 的消融口径。"""

    def __init__(self, weights: dict):
        super().__init__()
        self.weights = dict(weights)

    def forward(self, losses: dict) -> tuple:
        total = 0.0
        detail = {}
        for name, w in self.weights.items():
            if name in losses:
                total = total + float(w) * losses[name]
                detail[f"w_{name}"] = torch.tensor(float(w))
        return total, detail


def build_weighting(kind: str, task_names, weights=None):
    if kind == "uncertainty":
        return UncertaintyWeighting(task_names)
    if kind == "fixed":
        return FixedWeighting(weights or {n: 1.0 for n in task_names})
    raise ValueError(f"未知加权方式 {kind}")
