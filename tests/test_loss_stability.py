"""数值稳定性回归测试：因果损失的 imp 漂移与 CVR 头坍缩。

背景（12 号 E30/E31）：`dr` / `vrdr` 曾把插补塔输出 `imp` 作为无界线性项
`mean(imp)` 直接放进损失。该线性项的梯度是 `1 - click/e`，期望近乎 0 但不严格为 0，
训练两万余步后 imp 被推到 −1e12 量级，CVR 塔随之坍缩为常数（AUC 恰为 0.5000）。

本文件锁住三条不变量：
1. `detach_imp=True` 时损失不得向 imp 回传梯度（漂移的唯一通道被切断）；
2. 开关未被写死：`detach_imp=False` 时梯度确实存在；
3. 数百步训练后 imp 保持在 [0, IMP_MAX] 且 CVR 输出未坍缩为常数。

这些用例刻意绕开真实数据与 DataLoader，纯张量合成，CPU 上几秒内完成。
"""

import torch

import src.models  # noqa: F401  触发模型注册
from src.losses.causal import SelfNormalizer, dr_loss, vr_dr_loss
from src.models.base import build_model
from src.models.uc_escm import IMP_MAX


def _batch(n: int = 2048, seed: int = 0, e_value: float = 0.05):
    """构造点击/转化稀疏的合成 batch，e 取真实量级的 5%。"""
    g = torch.Generator().manual_seed(seed)
    click = (torch.rand(n, generator=g) < e_value).float()
    conv = ((torch.rand(n, generator=g) < 0.2) * click).float()
    e = torch.full((n,), e_value)
    # 模拟已训练的 CVR 头：logit 均值 −4、有区分度；requires_grad 以复现真实计算图
    cvr_logit = (torch.randn(n, generator=g) * 0.5 - 4.0).requires_grad_(True)
    return cvr_logit, click, conv, e


def test_dr_detached_imp_receives_no_gradient():
    """detach_imp=True 时，dr_loss 不得向 imp 回传任何梯度。"""
    cvr_logit, click, conv, e = _batch()
    imp = torch.full((click.numel(),), 3.0, requires_grad=True)
    loss = dr_loss(cvr_logit, imp, click, conv, e, detach_imp=True)
    loss.backward()
    assert imp.grad is None or float(imp.grad.abs().max()) == 0.0


def test_vrdr_detached_imp_receives_no_gradient():
    """vr_dr_loss 的 dr 部分同理；imp_loss 里 imp 是主变量，会正常回传。"""
    cvr_logit, click, conv, e = _batch()
    imp = torch.full((click.numel(),), 3.0, requires_grad=True)
    dr, imp_loss = vr_dr_loss(cvr_logit, imp, click, conv, e, detach_imp=True)
    dr.backward(retain_graph=True)
    assert imp.grad is None or float(imp.grad.abs().max()) == 0.0
    # imp_loss 必须能给 imp 梯度，否则插补塔学不到东西
    imp.grad = None
    imp_loss.backward()
    assert imp.grad is not None and float(imp.grad.abs().max()) > 0.0


def test_detach_switch_is_not_hardcoded():
    """detach_imp=False 时必须有梯度，保证消融开关真的可用。"""
    cvr_logit, click, conv, e = _batch()
    imp = torch.full((click.numel(),), 3.0, requires_grad=True)
    dr_loss(cvr_logit, imp, click, conv, e, detach_imp=False).backward()
    assert imp.grad is not None and float(imp.grad.abs().max()) > 0.0


def test_uc_escm_imp_is_nonnegative_and_bounded():
    """模型层保证 δ̂ ∈ [0, IMP_MAX]，与 BCE 目标的非负性一致。"""
    model = build_model("uc_escm", {}, [200, 200, 150])
    ids = torch.randint(0, 150, (64, 3, 8))
    imp = model(ids)["imp"]
    assert float(imp.min()) >= 0.0
    assert float(imp.max()) <= IMP_MAX + 1e-6


def test_vrdr_300_steps_keeps_imp_bounded_and_cvr_alive():
    """端到端：300 步后在合成的稀疏点击数据上 imp 不漂移、CVR 头不坍缩。"""
    torch.manual_seed(0)
    model = build_model("uc_escm", {}, [200, 200, 150])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cvr_logit = imp_out = None

    for _ in range(300):
        ids = torch.randint(0, 150, (512, 3, 8))
        click = (torch.rand(512) < 0.05).float()
        conv = ((torch.rand(512) < 0.2) * click).float()
        out = model(ids)
        e = torch.full((512,), 0.05)
        ctr = torch.nn.functional.binary_cross_entropy_with_logits(out["ctr_logit"], click)
        dr, imp_loss = vr_dr_loss(out["cvr_logit"], out["imp"], click, conv, e)
        total = ctr + dr + imp_loss
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        cvr_logit, imp_out = out["cvr_logit"].detach(), out["imp"].detach()

    assert not torch.isnan(total).item(), "训练损失出现 NaN"
    assert float(imp_out.min()) >= 0.0, "imp 出现负值"
    assert float(imp_out.max()) <= IMP_MAX + 1e-6, f"imp 越界: {float(imp_out.max())}"
    assert float(cvr_logit.std()) > 1e-3, "CVR 输出坍缩为常数"
    assert cvr_logit.unique().numel() > 100, "CVR 输出几乎无区分度"


def test_self_normalizer_smooths_heavy_batches():
    """SNIPS 的 Z 必须跨 batch 平滑：单批重尾（一个 e=1e-4 的点击）不得让 w 爆炸。"""
    normalizer = SelfNormalizer(decay=0.99)
    e = torch.full((1000,), 0.05)
    click = torch.zeros(1000)
    click[:50] = 1                      # 5% CTR → Z = 1.0
    normalizer.normalize(click / e)
    assert abs(normalizer.value - 1.0) < 1e-6

    heavy = torch.zeros(1000)
    heavy[0] = 1e4                      # 相当于 e=1e-4 的单个点击
    w = normalizer.normalize(heavy)
    assert normalizer.value < 5.0, "EMA 未压住重尾批"
    assert float(w.max()) < 1e4, "单批 Z 直接重估会让 w 跨数量级"


def test_self_normalizer_z_has_no_gradient():
    """Z 必须 detach：否则归一化项会引入与主项反号的梯度（SNIPS 塌陷的候选成因）。"""
    e = torch.full((10,), 0.5)
    click = torch.zeros(10)
    click[:2] = 1
    w = (click / e).requires_grad_(True)
    normalizer = SelfNormalizer()
    out = normalizer.normalize(w)
    out.sum().backward()
    # Z 作为常数处理时，每个元素的梯度应恰为 1/Z；若 Z 参与反传会多出反号项
    expected = 1.0 / normalizer.value
    assert torch.allclose(w.grad, torch.full_like(w, expected))
