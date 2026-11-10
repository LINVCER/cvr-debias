"""L3：10% 分层采样（按 click × conversion 四格），只用于调参与冒烟。

依据：
- doc/202610-选题/06-数据清洗与构建方案.md §8（分层抽样、固定种子、偏差 < 10%）
- doc/202610-选题/13-路线确认单.md S4

做法：逐分片读 20_processed，按四个标签格子各抽 10%，保留原始 schema 写出。
输出目录结构、列名与 20_processed 完全一致，训练侧可直接把路径指到 30_sample。
"""

import argparse
import json
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROCESSED = Path("data/aliccp/20_processed")
SAMPLE = Path("data/aliccp/30_sample")
REPORTS = Path("data/aliccp/90_reports/03_dataset")

CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
MAX_REL_DEV = 0.10  # 06 §8：采样集三级转化率与全量偏差 < 10%


def _cell_seed(split: str, file_name: str, cell: tuple[int, int]) -> int:
    """每个 (split, 分片, 标签格) 一个稳定种子，保证重跑结果一致。"""
    key = f"{split}|{file_name}|{cell[0]}{cell[1]}".encode("utf-8")
    return zlib.crc32(key) % (2 ** 32)


def _rates(exposure: int, click: int, conversion: int) -> dict:
    return {
        "exposure": exposure,
        "click": click,
        "conversion": conversion,
        "ctr": round(click / exposure, 6) if exposure else 0.0,
        "cvr_click_space": round(conversion / click, 6) if click else 0.0,
        "ctcvr": round(conversion / exposure, 8) if exposure else 0.0,
    }


def sample_split(split: str, ratio: float, out_dir: Path) -> dict:
    files = sorted((PROCESSED / split).glob("part-*.parquet"))
    if not files:
        raise SystemExit(f"{PROCESSED / split} 下没有 parquet 分片，先跑 prepare_aliccp --stage clean")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("part-*.parquet"):
        stale.unlink()

    full_cells = {cell: 0 for cell in CELLS}
    kept_cells = {cell: 0 for cell in CELLS}
    rows_in = rows_out = 0
    for path in files:
        table = pq.read_table(path)
        rows_in += table.num_rows
        click = table.column("click").to_numpy(zero_copy_only=False)
        conv = table.column("conversion").to_numpy(zero_copy_only=False)
        picks = []
        for cell in CELLS:
            idx = np.flatnonzero((click == cell[0]) & (conv == cell[1]))
            full_cells[cell] += len(idx)
            k = int(round(len(idx) * ratio))
            if k == 0:
                continue
            rng = np.random.default_rng(_cell_seed(split, path.name, cell))
            picks.append(rng.choice(idx, size=k, replace=False))
            kept_cells[cell] += k
        if not picks:
            continue
        keep = np.sort(np.concatenate(picks))
        out = table.take(pa.array(keep, type=pa.int64()))
        pq.write_table(out, out_dir / path.name, compression="zstd")
        rows_out += out.num_rows

    # 全量标签：click 只有 (1,0)+(1,1) 两格，conversion 只有 (1,1)
    full_click = full_cells[(1, 0)] + full_cells[(1, 1)]
    full_conv = full_cells[(1, 1)]
    kept_click = kept_cells[(1, 0)] + kept_cells[(1, 1)]
    kept_conv = kept_cells[(1, 1)]
    return {
        "split": split,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "ratio": round(rows_out / rows_in, 6) if rows_in else 0.0,
        "cells_full": {f"c{c[0]}v{c[1]}": n for c, n in full_cells.items()},
        "cells_sample": {f"c{c[0]}v{c[1]}": n for c, n in kept_cells.items()},
        "full": _rates(rows_in, full_click, full_conv),
        "sample": _rates(rows_out, kept_click, kept_conv),
    }


def _deviation(sample: dict, full: dict) -> dict:
    out = {}
    for key in ("ctr", "cvr_click_space", "ctcvr"):
        base = full[key]
        out[key] = round(abs(sample[key] - base) / base, 6) if base else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="10% 分层采样（按 click × conversion）")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--out", default=str(SAMPLE))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    out_root = Path(args.out)
    results = {}
    for split in args.splits:
        stats = sample_split(split, args.ratio, out_root / split)
        stats["relative_deviation"] = _deviation(stats["sample"], stats["full"])
        results[split] = stats
        print(f"[sample] {split}: {stats['rows_in']:,} -> {stats['rows_out']:,} "
              f"({stats['ratio'] * 100:.2f}%) CTR 偏差 {stats['relative_deviation']['ctr'] * 100:.3f}% "
              f"CVR 偏差 {stats['relative_deviation']['cvr_click_space'] * 100:.3f}%")

    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    path = REPORTS / f"sample_report_{ts}.json"
    path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ratio": args.ratio,
        "splits": results,
        "output": str(out_root),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] {path}")

    for split, stats in results.items():
        for key, dev in stats["relative_deviation"].items():
            assert dev < MAX_REL_DEV, f"{split} {key} 偏差 {dev:.2%} 超过 {MAX_REL_DEV:.0%}"


if __name__ == "__main__":
    main()
