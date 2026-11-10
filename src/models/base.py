"""模型注册表与统一输出契约。

所有模型 forward(ids) 都返回同一组键：
- ctr_logit: [B]，缺失时为 None
- cvr_logit: [B]，缺失时为 None
- imp:       [B]，插补塔输出，缺失时为 None
- extras:    结构正则需要的中间量（u_p / u_c），缺失时为空字典

损失与模型正交：任一模型可配任一损失，这是 12 个基线能写成「结构 × 损失」笛卡尔积的前提。
"""

from torch import nn

MODEL_REGISTRY = {}


def register(name):
    def _wrap(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return _wrap


def build_model(name: str, cfg: dict, field_dims) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型 {name}，可用：{sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](field_dims=field_dims, **cfg)


def empty_output(batch_size: int, device=None) -> dict:
    """构造标准输出字典，缺的项置 None。"""
    return {"ctr_logit": None, "cvr_logit": None, "imp": None, "extras": {}}
