"""EmbeddingBank 的 K=8 masked mean pooling 契约。"""

import torch

from src.features.embedding import EmbeddingBank


def test_masked_mean_pooling_ignores_pad_and_averages_real():
    bank = EmbeddingBank([5, 3], dim=4, max_len=8)
    # 构造可控权重：第 0 域 id=2 的向量为全 1，id=3 为全 3
    with torch.no_grad():
        bank.tables[0].weight.zero_()
        bank.tables[0].weight[2] = 1.0
        bank.tables[0].weight[3] = 3.0
        bank.tables[1].weight.fill_(0.5)

    # 域 0：真实值 [2,3] + 6 个 PAD → 均值 (1+3)/2 = 2；域 1：全 PAD → 0（防除零）
    ids = torch.zeros(1, 2, 8, dtype=torch.long)
    ids[0, 0, 0] = 2
    ids[0, 0, 1] = 3
    out = bank(ids)
    assert out.shape == (1, 2 * 4)
    assert torch.allclose(out[0, :4], torch.full((4,), 2.0))
    assert torch.allclose(out[0, 4:], torch.zeros(4))


def test_pad_row_zero_after_init():
    bank = EmbeddingBank([10], dim=8)
    assert torch.allclose(bank.tables[0].weight[0], torch.zeros(8))


def test_shape_validation():
    bank = EmbeddingBank([5, 3], dim=4)
    # 2D 合法：视为单值 K=1，输出与 [B, F, 1] 一致
    out2d = bank(torch.zeros(2, 2, dtype=torch.long))
    out3d = bank(torch.zeros(2, 2, 1, dtype=torch.long))
    assert torch.allclose(out2d, out3d)
    try:
        bank(torch.zeros(2, 3, dtype=torch.long))  # 域数不匹配仍应拒绝
    except ValueError as exc:
        assert "期望形状" in str(exc)
    else:
        raise AssertionError("应拒绝域数不匹配的输入")
