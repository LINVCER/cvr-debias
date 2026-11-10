"""统计 L0 原始层四个文件的字节数与行数，结果写入 L4 报告层（90_reports/01_probe）。

分层约定见 data/aliccp/README.md：
    00_raw/train|test/*.csv  ->  90_reports/01_probe/stats_raw.json
"""

import json
from pathlib import Path

RAW = Path("data/aliccp/00_raw")
OUT = Path("data/aliccp/90_reports/01_probe/stats_raw.json")

FILES = [
    ("train", "sample_skeleton_train.csv"),
    ("train", "common_features_train.csv"),
    ("test", "sample_skeleton_test.csv"),
    ("test", "common_features_test.csv"),
]


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        while True:
            block = f.read(1 << 24)
            if not block:
                break
            n += block.count(b"\n")
    return n


out = {}
for split, name in FILES:
    p = RAW / split / name
    rows = count_lines(p)
    out[f"{split}/{name}"] = {"bytes": p.stat().st_size, "lines": rows}
    print(f"{split}/{name}: {rows:,} 行, {p.stat().st_size / 2**30:.2f} GiB", flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已写 {OUT}")
