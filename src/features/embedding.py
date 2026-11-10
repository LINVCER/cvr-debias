"""共享稀疏特征层。

22 个稀疏域共用一套 embedding 逻辑，每域一张表、统一维度。
稀疏场景下共享 embedding 是知识迁移的主要通道，不做域间拆分。

K=8 定长多值契约（与数据管道一致）：
- 输入 ids 形状 [B, F, K]，id 语义：0=缺失/补齐、1=OOV、>=2 真实取值
- 每域先查 K 个 embedding，再对非 PAD 位做 masked mean pooling，得到 [B, F, dim]
"""

import torch
from torch import nn


class EmbeddingBank(nn.Module):
    """多域稀疏特征 → 稠密向量。

    输入 ids 形状 [B, F, K]；第 i 列对应第 i 个域，K 为多值定长。
    """

    def __init__(self, field_dims, dim: int = 16, max_len: int = 8):
        super().__init__()
        if not field_dims:
            raise ValueError("field_dims 不能为空")
        self.field_dims = [int(d) for d in field_dims]
        self.dim = int(dim)
        self.max_len = int(max_len)
        self.tables = nn.ModuleList([nn.Embedding(d, self.dim) for d in self.field_dims])
        for table in self.tables:
            nn.init.normal_(table.weight, std=0.01)
            # PAD(id=0) 置零：它是补齐位，不参与梯度，也保证 masked mean 的分子只含真实值
            with torch.no_grad():
                table.weight[0].zero_()
        self.out_dim = self.dim * len(self.field_dims)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """[B, F, K] → masked mean pooling → [B, F * dim] 拼接特征。

        兼容 [B, F]（单值数据，等价于 K=1）。
        """
        if ids.dim() == 2:
            ids = ids.unsqueeze(-1)
        if ids.dim() != 3 or ids.size(1) != len(self.field_dims):
            raise ValueError(
                f"ids 期望形状 [B, {len(self.field_dims)}, K]，实际 {tuple(ids.shape)}"
            )
        ids = ids.long()
        mask = (ids != 0).unsqueeze(-1)                      # [B, F, K, 1]
        counts = mask.sum(dim=2).clamp(min=1)                # [B, F, 1]，防整域缺失除零
        pooled = []
        for i, table in enumerate(self.tables):
            emb = table(ids[:, i, :])                        # [B, K, dim]
            summed = (emb * mask[:, i]).sum(dim=1)           # [B, dim]
            pooled.append(summed / counts[:, i])
        return torch.cat(pooled, dim=1)                      # [B, F * dim]
