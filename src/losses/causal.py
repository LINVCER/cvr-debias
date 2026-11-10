"""因果损失层：朴素风险、IPS、DR、方差正则 DR（C2）。

记一条曝光样本为 (x, y, z)，y 为点击、z 为转化。z 仅在 y = 1 时可观测，
因此观测指示 o = y、倾向得分 e = P(o = 1 | x) = pCTR。

- L_ideal = 全曝光空间上的期望损失；
- L_naive = 点击空间上的期望损失，是 L_ideal 的有偏估计（MNAR）；
- L_IPS   = Σ o·δ/e；
- L_DR    = Σ [δ̂ + o(δ - δ̂)/e]，e 或 δ̂ 任一正确即无偏。

方差正则项来自 MRDR（SIGIR 2021），不是本研究原创：本研究把它从点击空间推广到
全空间 ESCM² 框架，并与校准倾向联合分析。开题与论文必须按这个口径表述。
"""

import torch
import torch.nn.functional as F

EPS = 1e-6


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
) -> torch.Tensor:
    """双重稳健风险：全空间插补 + 点击空间残差纠偏。

    :param detach_imp: 插补项是否 detach，默认 True。
        若为 False，`mean(imp)` 这一项会让 imp 无界漂移——它的梯度是 1 - click/e，
        期望近似 0 但不严格为 0，训练上万步后会累积到 −1e12 量级（见 12 号 E30）。
        imp 塔的学习应完全交给 imp_mse_loss，DR 项只把它当常数基线用。
    """
    delta = F.binary_cross_entropy_with_logits(cvr_logit, conv.float(), reduction="none")
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
) -> tuple:
    """方差正则双重稳健损失（C2）。

    插补塔的训练目标改为 DR 估计量方差上界的加权平方误差，
    权重 (1-ẽ)/ẽ² 正是 MRDR 的推导结果。

    :param detach_imp: dr 部分的 imp 是否 detach，默认 True（与 dr_loss 同理，
        不 detach 会让 imp 通过线性项无界漂移）。DR-JL 交替训练口径也用它。
    :return: (dr, gamma * imp_loss)
    """
    delta = F.binary_cross_entropy_with_logits(cvr_logit, conv.float(), reduction="none")
    imp_for_dr = imp.detach() if detach_imp else imp

    dr = (imp_for_dr + click * (delta - imp_for_dr) / e).mean()

    w = click * (1.0 - e) / e.pow(2)
    imp_loss = (w * (delta.detach() - imp).pow(2)).sum() / click.sum().clamp_min(1.0)
    return dr, gamma * imp_loss
