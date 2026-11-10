"""校准纪律：温度只在验证集上拟合，τ 取分位数，α 的准则不是 AUC。"""

import torch

from src.calibration.temperature import fit_temperature, select_alpha, select_tau


def test_temperature_is_valid_and_fitted_on_given_data():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2000, generator=g) * 2
    labels = (torch.rand(2000, generator=g) < torch.sigmoid(logits)).float()

    t = fit_temperature(logits, labels)
    assert 0.1 <= t <= 20.0

    # 过温（T > 1）应降低过自信模型的 NLL
    overconfident = logits * 3
    t_fit = fit_temperature(overconfident, labels)
    raw = torch.nn.functional.binary_cross_entropy_with_logits(overconfident, labels).item()
    fitted = torch.nn.functional.binary_cross_entropy_with_logits(overconfident / t_fit, labels).item()
    assert fitted <= raw


def test_tau_is_quantile_of_calibrated_scores():
    logits = torch.linspace(-10, 10, 1001)
    tau = select_tau(logits, temperature=1.0, alpha=0.01)
    assert 0.0 <= tau <= 1.0
    manual = float(torch.quantile(torch.sigmoid(logits), 0.01))
    assert abs(tau - manual) < 1e-6


def test_alpha_returned_from_candidate_set():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(3000, generator=g)
    click = (torch.rand(3000, generator=g) < torch.sigmoid(logits)).float()
    alphas = (0.001, 0.005, 0.01)
    chosen = select_alpha(logits, click, alphas, temperature=1.0)
    assert chosen["alpha"] in alphas
    assert chosen["tau"] >= 0.0
    assert chosen["score"] == chosen["score"]  # 非 NaN


def test_temperature_keeps_base_rate():
    """回归测试：温度标度不能把均值拉走。

    曾用负正比加权 NLL 拟合温度，T 被从 ≈1 拉到 5.6，预测均值从 0.046 升到 0.23，
    PCOC 从 1.0 变成 5.6。温度只能改形状，一旦均值被改，校准就是假的。
    """
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(3000, generator=g) * 4 - 4
    labels = (torch.rand(3000, generator=g) < torch.sigmoid(logits)).float()

    t = fit_temperature(logits, labels)
    raw_mean = float(torch.sigmoid(logits).mean())
    cal_mean = float(torch.sigmoid(logits / t).mean())
    base = float(labels.mean())
    assert abs(raw_mean - base) < 0.05
    assert abs(cal_mean - base) < 0.05
    assert abs(t - 1.0) < 1.0
