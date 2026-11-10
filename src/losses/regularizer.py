"""结构正则：专家表征正交（C3）。

目的是让 CTR 任务的梯度不再直接塑造 CVR 表征，对应 ESCM² 的潜在独立优先（PIP）要求。
"""

import torch


def orthogonal_penalty(u_p: torch.Tensor, u_c: torch.Tensor) -> torch.Tensor:
    """归一化后的两组专属专家输出，互相关矩阵的 Frobenius 范数平方。

    :param u_p: 倾向专属专家输出，[B, n_p * expert_dim]
    :param u_c: 转化专属专家输出，[B, n_c * expert_dim]
    """
    if u_p.numel() == 0 or u_c.numel() == 0:
        return u_p.new_zeros(())
    up = (u_p - u_p.mean(0, keepdim=True)) / (u_p.std(0, keepdim=True) + 1e-6)
    uc = (u_c - u_c.mean(0, keepdim=True)) / (u_c.std(0, keepdim=True) + 1e-6)
    # 列已标准化，内积除以 batch 得到相关系数量级；再按专家对数归一，结果落在 O(1)
    batch = max(u_p.size(0), 1)
    denom = batch * max(u_p.size(1) * u_c.size(1), 1)
    return up.t().matmul(uc).pow(2).sum() / denom
