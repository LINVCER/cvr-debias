"""评估指标：全局唯一实现，所有模型脚本不得各写一份。

指标口径（对应总案 §4.2）：
- CVR AUC / PR-AUC：点击空间，与已有文献对齐的主指标，不能单报；
- CTCVR AUC：全曝光空间，最接近线上排序效果；
- CTR AUC：确认纠偏没牺牲 CTR；
- GAUC：点击空间按用户分组，组内无正或无负的用户剔除并报剔除比例；
- ECE（10 等频桶）：校准质量，C1 的直接证据；
- PCOC = Σŷ / Σy：整体高估倍数，论文最直观的一张图。
"""

import numpy as np

EPS = 1e-9


def _rankdata(x: np.ndarray) -> np.ndarray:
    """平均秩（1 基），向量化处理并列。"""
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = cum - (counts - 1) / 2.0
    return avg[inv]


def auc_score(labels, scores) -> float:
    y = np.asarray(labels).ravel()
    s = np.asarray(scores).ravel().astype(float)
    pos = y > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(s)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels, scores) -> float:
    y = np.asarray(labels).ravel()
    s = np.asarray(scores).ravel().astype(float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    prec = tp / np.maximum(tp + fp, EPS)
    rec = tp / max(tp[-1], EPS)
    d_rec = np.diff(np.concatenate([[0.0], rec]))
    return float((d_rec * prec).sum())


def gauc(labels, scores, groups) -> tuple:
    """返回 (gauc, 保留样本比例)。

    组内 AUC 用秩和（Mann-Whitney U）一次向量化：
    先按 (用户, 分数) 字典序排序，得到组内升序；组内并列分数取平均秩，
    再用 auc = (Σ正样本秩 - n_pos(n_pos+1)/2) / (n_pos·n_neg) 按组聚合。
    复杂度 O(n log n)，不再是逐用户 O(n·U)。
    """
    y = np.asarray(labels).ravel()
    s = np.asarray(scores).ravel().astype(np.float64)
    g = np.asarray(groups).ravel()
    n = y.size
    if n == 0:
        return float("nan"), 0.0

    # 主键用户、次键分数（升序）；组内名次即秩的基础
    order = np.lexsort((s, g))
    gs, ss = g[order], s[order]
    ys = y[order] > 0.5

    # 组边界
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    np.not_equal(gs[1:], gs[:-1], out=new_group[1:])
    group_id = np.cumsum(new_group) - 1
    n_groups = int(group_id[-1]) + 1
    sizes = np.bincount(group_id, minlength=n_groups)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))

    # 并列块：同一用户且同分数 → 取平均秩（与 _rankdata 口径一致）
    tie_start = np.empty(n, dtype=bool)
    tie_start[0] = True
    np.logical_or(new_group[1:], ss[1:] != ss[:-1], out=tie_start[1:])
    tie_id = np.cumsum(tie_start) - 1
    tie_sizes = np.bincount(tie_id)
    tie_first = np.concatenate(([0], np.cumsum(tie_sizes)[:-1]))
    # 组内 0 基起始位置 + 平均偏移 + 1 基秩
    avg_rank = tie_first[tie_id] - starts[group_id] + (tie_sizes[tie_id] - 1) / 2.0 + 1.0

    pos_float = ys.astype(np.float64)
    n_pos = np.bincount(group_id, weights=pos_float, minlength=n_groups)
    n_all = sizes.astype(np.float64)
    n_neg = n_all - n_pos
    sum_rank_pos = np.bincount(group_id, weights=avg_rank * pos_float, minlength=n_groups)

    valid = (n_pos > 0) & (n_neg > 0)
    if not valid.any():
        return float("nan"), 0.0
    aucs = (sum_rank_pos[valid] - n_pos[valid] * (n_pos[valid] + 1) / 2.0) / (
        n_pos[valid] * n_neg[valid]
    )
    w = n_all[valid]
    return float((aucs * w).sum() / w.sum()), float(w.sum() / n)


def ece(labels, probs, n_bins: int = 10) -> float:
    """等频分桶校准误差（不用等宽）。"""
    y = np.asarray(labels).ravel().astype(float)
    p = np.asarray(probs).ravel().astype(float)
    if len(y) == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    y, p = y[order], p[order]
    bins = np.array_split(np.arange(len(y)), min(n_bins, len(y)))
    total = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        total += len(b) * abs(p[b].mean() - y[b].mean())
    return float(total / len(y))


def pcoc(labels, probs) -> float:
    """Σŷ / Σy：> 1 为高估。"""
    y = np.asarray(labels).ravel().astype(float)
    p = np.asarray(probs).ravel().astype(float)
    denom = y.sum()
    return float(p.sum() / denom) if denom > 0 else float("nan")


def binary_metrics(labels, probs, groups=None, n_bins: int = 10) -> dict:
    """单空间指标一次性输出。"""
    out = {
        "auc": auc_score(labels, probs),
        "pr_auc": average_precision(labels, probs),
        "ece": ece(labels, probs, n_bins),
        "pcoc": pcoc(labels, probs),
        "n": int(np.asarray(labels).size),
        "pos_rate": float(np.asarray(labels).mean()),
    }
    if groups is not None:
        g, ratio = gauc(labels, probs, groups)
        out["gauc"] = g
        out["gauc_kept_ratio"] = ratio
    return out


def bias_and_mse(pred, pseudo_truth) -> dict:
    """半合成全标签集上直接度量真实偏差。

    真实数据上永远拿不到这两个数，因此只做方法间的相对比较，绝对数值不作结论。
    """
    pred = np.asarray(pred).ravel().astype(float)
    truth = np.asarray(pseudo_truth).ravel().astype(float)
    diff = pred - truth
    return {"bias": float(diff.mean()), "mse": float((diff ** 2).mean())}
