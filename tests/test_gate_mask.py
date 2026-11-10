"""门控掩码：CTR 门控对转化专属专家的权重恒为 0。"""

import torch

from src.models.experts import DisentangledExperts


def test_conversion_experts_do_not_affect_ctr():
    torch.manual_seed(0)
    experts = DisentangledExperts(in_dim=8, expert_dim=4, n_p=2, n_c=2, n_s=2)
    x = torch.randn(16, 8)
    out_before = experts(x)

    # 扰动第一个转化专属专家的输出层偏置（索引 n_p 起为转化专属）
    with torch.no_grad():
        experts.experts[2].net[2].bias += 1.0
    out_after = experts(x)

    assert torch.allclose(out_before["ctr"], out_after["ctr"], atol=1e-6)
    assert not torch.allclose(out_before["cvr"], out_after["cvr"], atol=1e-6)


def test_masks_and_shapes():
    experts = DisentangledExperts(in_dim=6, expert_dim=4, n_p=2, n_c=1, n_s=3)
    assert experts.mask_ctr[2].item() == 0.0 and experts.mask_ctr[5].item() == 1.0
    assert experts.mask_cvr[0].item() == 0.0 and experts.mask_cvr[2].item() == 1.0

    out = experts(torch.randn(5, 6))
    assert out["ctr"].shape == (5, 4)
    assert out["u_p"].shape == (5, 2 * 4)
    assert out["u_c"].shape == (5, 1 * 4)
