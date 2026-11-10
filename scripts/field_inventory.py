"""扫描 L0 特征串，统计实际出现的 field_id 及其特征个数（抽样，不全量）。

用途：确认清洗流水线要保留哪些稀疏域，避免照抄博客导致域缺失或重复。
"""

import json
from collections import Counter
from pathlib import Path

RAW = Path("data/aliccp/00_raw")
OUT = Path("data/aliccp/90_reports/01_probe/field_inventory.json")
MAX_LINES = 200_000


def scan_skeleton(path: Path) -> Counter:
    c = Counter()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= MAX_LINES:
                break
            parts = line.rstrip("\r\n").split(",", 5)
            if len(parts) < 6:
                continue
            for item in parts[5].split("\x01"):
                seg = item.split("\x02", 1)
                if seg and seg[0]:
                    c[seg[0]] += 1
    return c


def scan_common(path: Path) -> Counter:
    c = Counter()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= MAX_LINES:
                break
            parts = line.rstrip("\r\n").split(",", 2)
            if len(parts) < 3:
                continue
            for item in parts[2].split("\x01"):
                seg = item.split("\x02", 1)
                if seg and seg[0]:
                    c[seg[0]] += 1
    return c


out = {}
for split in ("train", "test"):
    sk = scan_skeleton(RAW / split / f"sample_skeleton_{split}.csv")
    cm = scan_common(RAW / split / f"common_features_{split}.csv")
    merged = sk + cm
    out[split] = {
        "sampled_lines": MAX_LINES,
        "n_fields": len(merged),
        "fields": {k: v for k, v in sorted(merged.items(), key=lambda x: -x[1])},
        "skeleton_only": sorted(sk),
        "common_only": sorted(cm),
    }
    print(f"{split}: {len(merged)} 个 field_id（前 12 个高频）")
    for k, v in sorted(merged.items(), key=lambda x: -x[1])[:12]:
        print(f"    {k:<10} {v:>10,}")
    print(f"    仅出现在骨架表: {sorted(sk)[:15]}")
    print(f"    仅出现在公共表: {sorted(cm)[:15]}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已写 {OUT}")
