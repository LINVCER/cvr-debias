"""UC-ESCM（本科口径为 C1 + C2 + C5；C3/C4 结构保留但进展望章）。

四层：共享特征层 → 解耦专家层 → 任务塔层 → 因果损失层。
三塔：CTR 塔输出 ê、CVR 塔输出 ẑ、插补塔输出 δ̂（不与 CVR 塔共参）。
"""

import torch.nn.functional as F
from torch import nn

from src.features.embedding import EmbeddingBank
from src.models.base import register
from src.models.experts import DisentangledExperts
from src.models.towers import Tower

# δ̂ 拟合的是逐样本 BCE（下界 0、量级 0–20），必须非负且有上界，
# 否则 dr/vr_dr 里的线性项会让它无界漂移（12 号 E30：曾漂到 −1e12）
IMP_MAX = 20.0


@register("uc_escm")
class UCESCM(nn.Module):
    def __init__(
        self,
        field_dims,
        emb_dim: int = 16,
        expert_dim: int = 32,
        n_propensity: int = 2,
        n_conversion: int = 2,
        n_shared: int = 4,
        hidden=(64, 32),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.emb = EmbeddingBank(field_dims, emb_dim)
        self.experts = DisentangledExperts(
            self.emb.out_dim, expert_dim, n_propensity, n_conversion, n_shared
        )
        self.ctr_tower = Tower(expert_dim, hidden, dropout)
        self.cvr_tower = Tower(expert_dim, hidden, dropout)
        self.imp_tower = Tower(expert_dim, hidden, dropout)

    def forward(self, ids):
        x = self.emb(ids)
        reps = self.experts(x)
        # softplus 保非负 + clamp 上界：δ̂ 的目标是 BCE 损失值，负值无物理意义
        imp = F.softplus(self.imp_tower(reps["cvr"])).clamp(max=IMP_MAX)
        return {
            "ctr_logit": self.ctr_tower(reps["ctr"]),
            "cvr_logit": self.cvr_tower(reps["cvr"]),
            "imp": imp,
            "extras": {"u_p": reps["u_p"], "u_c": reps["u_c"]},
        }
