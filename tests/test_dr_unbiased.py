"""命题 P1 的数值验证：DR 在插补正确时无偏；IPS 在倾向正确时期望无偏。"""

import torch

from src.losses.causal import dr_loss, ips_loss, propensity_score


def _synthetic_labels(n: int = 200_000, e: float = 0.5, p_conv: float = 0.3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    click = (torch.rand(n, generator=g) < e).float()
    conv = (torch.rand(n, generator=g) < p_conv).float()
    conv = conv * click                      # 转化只在点击时可观测
    cvr_logit = torch.zeros(n)
    e_t = torch.full((n,), e)
    return cvr_logit, click, conv, e_t


def test_dr_exact_when_imputation_correct():
    """若 δ̂ = δ，则 DR 恒等于理想风险（无偏性在有限样本下也应严格成立）。"""
    cvr_logit, click, conv, e = _synthetic_labels()
    delta = torch.nn.functional.binary_cross_entropy_with_logits(cvr_logit, conv, reduction="none")
    dr = dr_loss(cvr_logit, delta, click, conv, e)
    assert torch.allclose(dr, delta.mean(), atol=1e-6)


def test_ips_unbiased_in_expectation():
    """e 正确时，IPS 风险在容差内逼近全空间理想风险。"""
    cvr_logit, click, conv, e = _synthetic_labels()
    ideal = torch.nn.functional.binary_cross_entropy_with_logits(cvr_logit, conv).item()
    ips = ips_loss(cvr_logit, click, conv, e).item()
    assert abs(ips - ideal) < 0.01


def test_propensity_detached_and_clipped():
    """权重不回传梯度，且截断下界生效。"""
    logits = torch.full((100,), -8.0, requires_grad=True)
    e = propensity_score(logits, temperature=1.0, tau=1e-3)
    assert not e.requires_grad
    assert float(e.min()) >= 1e-3 - 1e-9
