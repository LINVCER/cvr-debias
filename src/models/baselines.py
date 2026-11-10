"""对照基线与结构基线：朴素 CVR、ESMM、MMoE。"""

from torch import nn

from src.features.embedding import EmbeddingBank
from src.models.base import register
from src.models.experts import GatedExperts
from src.models.towers import Tower


@register("mlp_cvr")
class MLPCVR(nn.Module):
    """朴素 CVR：只在点击空间训练，SSB 完全未处理。"""

    def __init__(self, field_dims, emb_dim=16, hidden=(64, 32), dropout=0.0):
        super().__init__()
        self.emb = EmbeddingBank(field_dims, emb_dim)
        self.cvr_tower = Tower(self.emb.out_dim, hidden, dropout)

    def forward(self, ids):
        x = self.emb(ids)
        return {"ctr_logit": None, "cvr_logit": self.cvr_tower(x), "imp": None, "extras": {}}


@register("esmm")
class ESMM(nn.Module):
    """ESMM：共享 embedding，在全空间监督 CTR 与 CTCVR，pCVR 隐式学得。"""

    def __init__(self, field_dims, emb_dim=16, hidden=(64, 32), dropout=0.0):
        super().__init__()
        self.emb = EmbeddingBank(field_dims, emb_dim)
        self.ctr_tower = Tower(self.emb.out_dim, hidden, dropout)
        self.cvr_tower = Tower(self.emb.out_dim, hidden, dropout)

    def forward(self, ids):
        x = self.emb(ids)
        return {
            "ctr_logit": self.ctr_tower(x),
            "cvr_logit": self.cvr_tower(x),
            "imp": None,
            "extras": {},
        }


@register("mmoe")
class MMoE(nn.Module):
    """MMoE：专家分解 + 门控缓解任务冲突，但不触及 SSB。"""

    def __init__(self, field_dims, emb_dim=16, expert_dim=32, n_experts=8, hidden=(32,), dropout=0.0):
        super().__init__()
        self.emb = EmbeddingBank(field_dims, emb_dim)
        self.experts = GatedExperts(self.emb.out_dim, expert_dim, n_experts, tasks=("ctr", "cvr"))
        self.ctr_tower = Tower(expert_dim, hidden, dropout)
        self.cvr_tower = Tower(expert_dim, hidden, dropout)

    def forward(self, ids):
        x = self.emb(ids)
        reps = self.experts(x)
        return {
            "ctr_logit": self.ctr_tower(reps["ctr"]),
            "cvr_logit": self.cvr_tower(reps["cvr"]),
            "imp": None,
            "extras": {},
        }
