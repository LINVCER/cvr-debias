"""校准层：温度标度与分位数截断（C1）。

三个纪律：
1. 温度 T 是后验拟合，不进主训练图；训练完 CTR 头后在验证集单独拟合；
2. τ 取训练集 pCTR 的 α 分位数，α 的选择准则不是 AUC，而是校准误差与权重方差的加权和；
3. 校准参数单独落盘，服务侧可热更新（分钟级），不动模型（小时级）。
"""

import math

import torch
import torch.nn.functional as F

from src.evaluation.metrics import ece

EPS = 1e-6


def fit_temperature(
    logits,
    labels,
    grid=None,
    t_min: float = 0.1,
    t_max: float = 20.0,
    steps: int = 200,
) -> float:
    """在验证集上最小化 NLL 拟合温度 T。

    只用普通 NLL，**不要加正样本权重**。实测在合成数据上（点击率 4.3%、原始 pCTR
    均值 0.0458）普通 NLL 给出 T≈0.96（校准前后均值一致），而按负正比加权后 T 被拉到
    5.57，预测均值升到 0.23、PCOC 从 1.0 变成 5.6——加权把「均值偏差」引进了温度，
    而温度标度本身只能调整分布形状，无法纠正均值，于是校准被破坏。
    """
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.float32)
    if grid is None:
        grid = torch.logspace(math.log10(t_min), math.log10(t_max), steps)

    def nll(t: float) -> float:
        return float(F.binary_cross_entropy_with_logits(logits / t, labels))

    # 以 T = 1 为基准，避免平坦区间上网格把边界值选成最优
    best_t, best_nll = 1.0, nll(1.0)
    for t in grid:
        value = nll(float(t))
        if value < best_nll - 1e-6:
            best_t, best_nll = float(t), value
    return best_t


def select_tau(logits, temperature: float, alpha: float) -> float:
    """截断下界 τ = 验证集 / 训练集 pCTR 的 α 分位数。"""
    calib = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float32) / float(temperature))
    if alpha <= 0:
        return 0.0
    return max(float(torch.quantile(calib, float(alpha))), EPS)


def select_alpha(
    logits,
    click,
    alphas=(0.001, 0.005, 0.01, 0.05),
    temperature: float = 1.0,
    var_weight: float = 0.5,
) -> dict:
    """在验证集上选 α。

    准则 = ECE（校准误差）+ var_weight × 点击样本上 1/ẽ 的变异系数平方。
    前者随 α 增大而变差（截断引入偏差），后者随 α 增大而变好（权重不再爆炸），
    两者相加形成单峰，这就是命题 P2/P3 在实验上的可观测形式。

    ECE 必须算在**截断后**的 ẽ 上：若用未截断的 calib，ECE 对 α 是常数，
    准则退化成「只看方差」，必然选中最大的 α，单峰也就无从验证。
    """
    logits = torch.as_tensor(logits, dtype=torch.float32)
    click_t = torch.as_tensor(click, dtype=torch.float32)
    calib = torch.sigmoid(logits / float(temperature))
    best = {"alpha": 0.0, "tau": 0.0, "score": float("inf")}
    for a in alphas:
        tau = select_tau(logits, temperature, a)
        e = calib.clamp_min(max(tau, EPS))
        pos_w = 1.0 / e[click_t > 0.5]
        if pos_w.numel() > 1 and float(pos_w.mean()) > 0:
            rel_var = float(pos_w.var(unbiased=False) / pos_w.mean().pow(2))
        else:
            rel_var = 0.0
        score = ece(click_t.numpy(), e.numpy()) + var_weight * rel_var
        if score < best["score"]:
            best = {"alpha": float(a), "tau": float(tau), "score": float(score)}
    return best
