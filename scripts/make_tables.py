"""把 artifacts/runs 下的 metrics.json 聚成论文用的 Markdown 表。

手工搬数字进论文是错误与返工的主要来源，主表、消融表都由本脚本生成。
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

COLUMNS = [
    ("CVR AUC", ("cvr_click_space", "auc")),
    ("CVR PR-AUC", ("cvr_click_space", "pr_auc")),
    ("CTCVR AUC", ("ctcvr_exposure", "auc")),
    ("CTR AUC", ("ctr_exposure", "auc")),
    ("ECE(CVR)", ("cvr_click_space", "ece")),
    ("PCOC(CVR)", ("cvr_click_space", "pcoc")),
    ("Bias vs p*", ("bias_vs_pstar", "bias")),
]


def _get(metrics: dict, path) -> float:
    cur = metrics
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return float("nan")
        cur = cur[key]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return float("nan")


def collect(runs_dir: Path) -> dict:
    groups = defaultdict(lambda: defaultdict(list))
    for path in sorted(runs_dir.rglob("metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        exp = path.parent.parent.parent.name
        model = path.parent.parent.name
        for title, key in COLUMNS:
            groups[(exp, model)][title].append(_get(payload.get("metrics", {}), key))
    return groups


def fmt(values) -> str:
    arr = np.asarray([v for v in values if not np.isnan(v)])
    if arr.size == 0:
        return "—"
    if arr.size == 1:
        return f"{arr[0]:.4f}"
    return f"{arr.mean():.4f} ± {arr.std(ddof=1):.4f}"


def main():
    parser = argparse.ArgumentParser(description="聚表")
    parser.add_argument("--runs", default="artifacts/runs")
    parser.add_argument("--out", default="doc/tables")
    args = parser.parse_args()

    groups = collect(Path(args.runs))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = "| 方法 | " + " | ".join(t for t, _ in COLUMNS) + " |"
    sep = "| --- | " + " | ".join("---" for _ in COLUMNS) + " |"
    lines = [header, sep]
    for (exp, model), values in sorted(groups.items()):
        cells = [fmt(values.get(t, [])) for t, _ in COLUMNS]
        lines.append(f"| {model} ({exp}) | " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n\n每格为 mean ± std（多种子）；与最强基线差异显著者需另行打标记。\n"
    (out_dir / "main_table.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
