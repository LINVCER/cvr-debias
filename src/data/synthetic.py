"""合成数据：只为冒烟测试与单元测试服务，不产生任何论文数字。

构造方式刻意保留了 SSB 的结构：点击概率由潜在分 g 决定，转化概率与 g 正相关，
因此「点击样本」的转化率天然高于全空间转化率——朴素 CVR 必然系统性高估。
"""

import torch


def _bernoulli(prob: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return (torch.rand(prob.shape, generator=generator) < prob).float()


def generate(cfg: dict) -> dict:
    n = int(cfg.get("n_samples", 20000))
    field_dims = [int(d) for d in cfg["field_dims"]]
    shift = float(cfg.get("shift", 2.0))
    g = torch.Generator().manual_seed(int(cfg.get("seed", 0)))

    latent = torch.randn(n, generator=g)
    p_click = torch.sigmoid(shift * latent - 4.6)          # 整体点击率约 4%
    conv_prob = torch.sigmoid(1.2 * latent - 1.5)          # 与 latent 正相关 => 强 SSB
    conv_full = _bernoulli(conv_prob, g)
    click = _bernoulli(p_click, g)
    conv = conv_full * click                               # 转化标签仅在点击时可观测

    # 特征必须携带 latent 信号，否则模型学不到东西、冒烟无法验证建模链路是否接通。
    # 每个域把 (latent + 独立噪声) 分桶成该域的取值，噪声控制该域的可分性。
    noise = float(cfg.get("feature_noise", 0.5))
    ids_list = []
    for d in field_dims:
        noisy = latent + noise * torch.randn(n, generator=g)
        order = noisy.argsort()
        ranks = torch.empty(n, dtype=torch.long)
        ranks[order] = torch.arange(n)
        ids_list.append(((ranks * d) // n).clamp(max=d - 1))
    ids = torch.stack(ids_list, dim=1)
    user_ids = torch.randint(0, max(n // 20, 1), (n,), generator=g)

    data = {
        "ids": ids,
        "click": click,
        "conv": conv,
        "conv_full": conv_full,     # 半合成评估用：全空间真实转化标签
        "p_star": conv_prob,        # 伪真值：用于直接度量真实偏差
        "user_ids": user_ids,
    }

    order = torch.randperm(n, generator=g)
    n_test = max(int(n * 0.2), 1)
    n_val = max(int(n * 0.05), 1)
    test_idx, val_idx, train_idx = order[:n_test], order[n_test : n_test + n_val], order[n_test + n_val:]

    def subset(idx):
        return {k: v[idx] for k, v in data.items()}

    return {"train": subset(train_idx), "val": subset(val_idx), "test": subset(test_idx)}
