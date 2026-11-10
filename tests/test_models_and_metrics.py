"""模型输出契约与指标实现自检。"""

import numpy as np
import torch

import src.models  # noqa: F401
from src.evaluation.metrics import auc_score, average_precision, ece, gauc, pcoc
from src.models.base import MODEL_REGISTRY, build_model
from src.models.uc_escm import IMP_MAX


def test_all_models_follow_output_contract():
    field_dims = [50, 50, 30]
    ids = torch.randint(0, 30, (7, 3, 8))  # [B, F, K] 多值定长契约
    for name in ("mlp_cvr", "esmm", "mmoe", "uc_escm"):
        model = build_model(name, {}, field_dims)
        out = model(ids)
        assert set(out) == {"ctr_logit", "cvr_logit", "imp", "extras"}
        assert out["cvr_logit"].shape == (7,)
        # 防坍缩回归：CVR 头输出不能是常数（12 号 E30 曾坍缩，AUC 恒为 0.5）
        assert float(out["cvr_logit"].std()) > 1e-6
        if name == "mlp_cvr":
            assert out["ctr_logit"] is None
        else:
            assert out["ctr_logit"].shape == (7,)
        if name == "uc_escm":
            # δ̂ 拟合非负 BCE，必须落在 [0, IMP_MAX]
            imp = out["imp"]
            assert imp.shape == (7,)
            assert float(imp.min()) >= 0.0
            assert float(imp.max()) <= IMP_MAX + 1e-6
    assert "uc_escm" in MODEL_REGISTRY


def test_metrics_perfect_and_inverted_ranking():
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
    assert auc_score(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc_score(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert average_precision(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_ece_and_pcoc():
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    probs = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert ece(labels, probs, n_bins=2) == 0.0
    # 预测翻倍 => PCOC 约为 2
    assert abs(pcoc(labels, probs * 2) - 2.0) < 1e-6


def test_gauc_drops_pure_groups():
    labels = torch.tensor([0, 0, 1, 1, 0, 0])
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9, 0.3, 0.5])
    groups = torch.tensor([0, 0, 0, 0, 1, 1])
    value, kept = gauc(labels, scores, groups)
    assert value == 1.0
    assert abs(kept - 4 / 6) < 1e-6


def test_gauc_vectorized_matches_per_group_auc():
    """向量化 GAUC 必须与逐用户 auc_score（平均秩口径）逐位一致，含并列分数。"""
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = 3000
        groups = rng.integers(0, 40, n)
        labels = rng.integers(0, 2, n)
        # 刻意制造大量并列：分数只取 7 个离散值
        scores = rng.choice(np.array([0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]), n)

        total_w, total_auc = 0.0, 0.0
        for gid in np.unique(groups):
            idx = groups == gid
            yy = labels[idx]
            if yy.sum() == 0 or (1 - yy).sum() == 0:
                continue
            total_auc += auc_score(yy, scores[idx]) * idx.sum()
            total_w += idx.sum()
        expect = total_auc / total_w
        expect_kept = total_w / n

        got, got_kept = gauc(labels, scores, groups)
        assert abs(got - expect) < 1e-9, (got, expect)
        assert abs(got_kept - expect_kept) < 1e-9
