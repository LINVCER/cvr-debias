"""因果损失层：朴素风险、IPS、DR、方差正则 DR（C2）。

记一条曝光样本为 (x, y, z)，y 为点击、z 为转化。z 仅在 y = 1 时可观测，
因此观测指示 o = y、倾向得分 e = P(o = 1 | x) = pCTR。

- L_ideal = 全曝光空间上的期望损失；
- L_naive = 点击空间上的期望损失，是 L_ideal 的有偏估计（MNAR）；
- L_IPS   = Σ o·δ/e；
- L_DR    = Σ [δ̂ + o(δ - δ̂)/e]，e 或 δ̂ 任一正确即无偏。

方差正则项来自 MRDR（Farajtabar et al., ICML 2018），不是本研究原创：本研究把它从
off-policy evaluation 场景迁移到 CVR 全空间，并与校准倾向联合分析。开题与论文必须按这个
口径表述——原论文是强化学习的策略评估，不是 CVR，迁移本身需要在论文里说明。
"""

import math

import torch
import torch.nn.functional as F

EPS = 1e-6


class SelfNormalizer:
    """SNIPS 自归一化因子的跨 batch EMA。

    Z = E[click/e] 理论上等于 1，但单批只有约 650 个点击，且 1/e 在 τ 截断后重尾
    （单个 e=1e-4 的点击可贡献 Z 的 0.6），逐 batch 重估会让有效学习率跨数量级抖动。
    这里用 EMA 平滑；e 已在 `propensity_score` 里 detach，Z 天然不参与反传。
    """

    def __init__(self, decay: float = 0.99):
        self.decay = float(decay)
        self.value: float | None = None

    def normalize(self, w_raw: torch.Tensor) -> torch.Tensor:
        z_batch = float(w_raw.mean().detach())
        if not math.isfinite(z_batch) or z_batch <= 0:
            z_batch = 1.0
        if self.value is None:
            self.value = z_batch
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * z_batch
        return w_raw / max(self.value, EPS)


def bce_logits(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, label.float())


def propensity_score(
    ctr_logit: torch.Tensor,
    temperature: float = 1.0,
    tau: float = 0.0,
    detach: bool = True,
) -> torch.Tensor:
    """校准后的倾向得分 ẽ。

    :param ctr_logit: CTR 塔 logit
    :param temperature: 温度标度 T（验证集拟合后传入）
    :param tau: 截断下界 τ（验证集分位数）
    :param detach: 权重不回传梯度。默认 True：否则模型会靠推高 pCTR 来转化器损失
    """
    s = ctr_logit.detach() if detach else ctr_logit
    e = torch.sigmoid(s / temperature)
    if tau > 0:
        e = e.clamp_min(tau)
    return e.clamp_min(EPS)


def naive_cvr_loss(cvr_logit: torch.Tensor, click: torch.Tensor, conv: torch.Tensor) -> torch.Tensor:
    """点击空间朴素 CVR 损失：只在 y = 1 上算，SSB 未处理。"""
    delta = F.binary_cross_entropy_with_logits(cvr_logit, conv.float(), reduction="none")
    return (click * delta).sum() / click.sum().clamp_min(1.0)


def ctcvr_loss(
    ctr_logit: torch.Tensor, cvr_logit: torch.Tensor, click: torch.Tensor, conv: torch.Tensor
) -> torch.Tensor:
    """全空间 CTCVR 监督：标签为 y·z，概率为 pCTR·pCVR。

    用对数和形式保证 pCTCVR ≈ 0.021% 时不数值下溢。
    """
    y = (click * conv).float()
    a, b = ctr_logit, cvr_logit
    log_p = F.logsigmoid(a) + F.logsigmoid(b)
    log_1mp = torch.log(torch.sigmoid(-a) + torch.sigmoid(a) * torch.sigmoid(-b) + EPS)
    return -(y * log_p + (1.0 - y) * log_1mp).mean()


def ips_loss(
    cvr_logit: torch.Tensor,
    click: torch.Tensor,
    conv: torch.Tensor,
    e: torch.Tensor,
) -> torch.Tensor:
    """逆倾向加权风险。δ̂ 为逐样本转化损失。"""
    delta = F.binary_cross_entropy_with_logits(cvr_logit, conv.float(), reduction="none")
    return (click * delta / e).mean()


def dr_loss(
    cvr_logit: torch.Tensor,
    imp: torch.Tensor,
    click: torch.Tensor,
    conv: torch.Tensor,
    e: torch.Tensor,
    detach_imp: bool = True,
    self_normalized: bool = False,
    normalizer: "SelfNormalizer | None" = None,
) -> torch.Tensor:
    """双重稳健风险：全空间插补 + 点击空间残差纠偏。

    :param detach_imp: 插补项是否 detach。True 时 `mean(imp)` 对 imp 无梯度，
        但也切断了 imp 塔经共享 embedding 回流到 CVR 塔的间接通道（E33 附注）。
    :param self_normalized: SNIPS 自归一化权重（Swaminathan & Joachims, NeurIPS 2015）。
        用 w = (click/e) / E[click/e] 替换 click/e。因为 Σw = N，
        `∂dr/∂imp_i = (1 - w_i)/N` 在整个 batch 上求和恒为 0，漂移项在 batch 内
        精确抵消，因此可以安全地不 detach（E30 的 −1e12 漂移根因）。
    :param normalizer: 跨 batch 的 Z 估计器；不给则退回逐 batch（仅用于单测）。
    """
    delta = F.binary_cross_entropy_with_logits(cvr_logit, conv.float(), reduction="none")
    if self_normalized:
        w = click / e
        w = normalizer.normalize(w) if normalizer is not None else w / w.mean().clamp_min(EPS)
        imp_term = imp.detach() if detach_imp else imp
        return (imp_term + w * (delta - imp_term)).mean()
    imp_for_dr = imp.detach() if detach_imp else imp
    return (imp_for_dr + click * (delta - imp_for_dr) / e).mean()


def imp_mse_loss(imp: torch.Tensor, cvr_logit: torch.Tensor, conv: torch.Tensor, click: torch.Tensor):
    """ESCM²-DR 的插补塔目标：点击空间普通 MSE（去 C2 时回退到这里）。"""
    delta = F.binary_cross_entropy_with_logits(cvr_logit.detach(), conv.float(), reduction="none")
    return ((delta - imp).pow(2) * click).sum() / click.sum().clamp_min(1.0)


def vr_dr_loss(
    cvr_logit: torch.Tensor,
    imp: torch.Tensor,
    click: torch.Tensor,
    conv: torch.Tensor,
    e: torch.Tensor,
    gamma: float = 1.0,
    detach_imp: bool = True,
    self_normalized: bool = False,
    normalizer: "SelfNormalizer | None" = None,
) -> tuple:
    """方差正则双重稳健损失（C2）。

    插补塔的训练目标改为 DR 估计量方差上界的加权平方误差，
    权重 (1-ẽ)/ẽ² 正是 MRDR 的推导结果。

    :param detach_imp: dr 部分的 imp 是否 detach。
    :param self_normalized: 见 `dr_loss`；SNIPS 下可安全取 detach_imp=False。
    :return: (dr, gamma * imp_loss)
    """
    delta = F.binary_cross_entropy_with_logits(cvr_logit, conv.float(), reduction="none")
    if self_normalized:
        w_snips = click / e
        w_snips = (
            normalizer.normalize(w_snips) if normalizer is not None
            else w_snips / w_snips.mean().clamp_min(EPS)
        )
        imp_term = imp.detach() if detach_imp else imp
        dr = (imp_term + w_snips * (delta - imp_term)).mean()
    else:
        imp_for_dr = imp.detach() if detach_imp else imp
        dr = (imp_for_dr + click * (delta - imp_for_dr) / e).mean()

    w = click * (1.0 - e) / e.pow(2)
    imp_loss = (w * (delta.detach() - imp).pow(2)).sum() / click.sum().clamp_min(1.0)
    return dr, gamma * imp_loss
