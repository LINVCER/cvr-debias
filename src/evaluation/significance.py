"""随机性与显著性检验（对应总案 §4.3）。

转化正样本只有万级，单次实验的 AUC 波动可能超过方法间差距。四条纪律：
每配置 5 个种子；报 mean ± std；与最强基线做配对双尾 t 检验并做 Holm 校正；
主指标额外给 bootstrap 置信区间。
"""

import numpy as np


def paired_t_test(a, b) -> dict:
    """配对双尾 t 检验。依赖 scipy；未安装时只返回 t 统计量。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    if n < 2:
        return {"t": float("nan"), "p": float("nan"), "n": n}
    sd = d.std(ddof=1)
    if sd == 0:
        return {"t": float("nan"), "p": float("nan"), "n": n}
    t = float(d.mean() / (sd / np.sqrt(n)))
    try:
        from scipy import stats

        p = float(stats.t.sf(abs(t), df=n - 1) * 2)
    except ImportError:
        p = float("nan")
    return {"t": t, "p": p, "n": n}


def bootstrap_ci(values, n_boot: int = 1000, level: float = 0.95, seed: int = 0) -> dict:
    """对一组种子结果做 bootstrap，返回均值与置信区间。"""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi)}


def holm_bonferroni(p_values: dict) -> dict:
    """Holm-Bonferroni 校正，返回每个假设的校正后 p 值。"""
    items = [(k, float(v)) for k, v in p_values.items() if not np.isnan(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, p * (m - i)))
        adjusted[k] = running
    return adjusted
