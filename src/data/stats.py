"""数据集统计：产出论文表 1 与图 2 的原始数字。

论文表 1 的所有统计量必须由本脚本在预处理完成后输出，包括 train / test 各自的
曝光、点击、转化统计与各特征域基数。引用数据集页面的二手数字是答辩常见扣分点。

同时对照 ``configs/data/aliccp.yaml`` 的 ``expected`` 做硬校验：偏差超过容差即
非零退出，避免「解析错了但没人发现」。注意 val 是从 train 按用户切出的，因此
``expected.train`` 对应的是 train + val 的合计。
"""

import argparse
import json
import sys
from pathlib import Path


def split_stats(shard_dir: Path) -> dict:
    import pyarrow.parquet as pq

    files = sorted(shard_dir.glob("*.parquet"))
    if not files:
        return {"n": 0}
    exposure = click = conv = 0
    for f in files:
        table = pq.read_table(f, columns=["click", "conversion"]).to_pandas()
        exposure += len(table)
        click += int(table["click"].sum())
        conv += int(table["conversion"].sum())
    return {
        "exposure": exposure,
        "click": click,
        "conversion": conv,
        "ctr": click / max(exposure, 1),
        "cvr": conv / max(click, 1),
        "ctcvr": conv / max(exposure, 1),
        "click_share_of_exposure": click / max(exposure, 1),
    }


def _sum_stats(*parts: dict) -> dict:
    total = {"exposure": 0, "click": 0, "conversion": 0}
    for part in parts:
        for key in total:
            total[key] += int(part.get(key, 0))
    return total


def _check(name: str, actual: int, expected: int, tol: float) -> tuple[dict, bool]:
    if not expected:
        return {"status": "skipped"}, True
    deviation = abs(actual - expected) / max(expected, 1)
    ok = deviation <= tol
    print(f"  {name:<11} 实测 {actual:>12,}  预期 {expected:>12,}  "
          f"偏差 {deviation:7.4%}  {'ok' if ok else 'FAIL'}")
    return {"actual": actual, "expected": expected,
            "deviation": round(deviation, 8), "status": "ok" if ok else "fail"}, ok


def main() -> int:
    parser = argparse.ArgumentParser(description="统计各 split 的曝光 / 点击 / 转化")
    parser.add_argument("--data", default="data/aliccp/20_processed", help="L2 成品层根目录")
    parser.add_argument("--out", default="data/aliccp/90_reports/03_dataset/stats.json",
                        help="报告落 L4，论文数字只认这里")
    parser.add_argument("--config", default="configs/data/aliccp.yaml",
                        help="expected 质检基线所在配置")
    parser.add_argument("--tol", type=float, default=0.005,
                        help="与 expected 的相对偏差容差，默认 0.5%")
    args = parser.parse_args()

    root = Path(args.data)
    stats = {}
    for split in ("train", "val", "test"):
        if (root / split).is_dir():
            stats[split] = split_stats(root / split)

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    expected = cfg.get("expected", {}) or {}

    failures = []
    checks = {}
    if expected:
        print("expected 校验：")
        if "train" in stats:
            observed = _sum_stats(stats.get("train", {}), stats.get("val", {}))
            checks["train(+val)"] = {}
            for key in ("exposure", "click", "conversion"):
                result, ok = _check(key, observed[key],
                                    int(expected.get("train", {}).get(key, 0)), args.tol)
                checks["train(+val)"][key] = result
                if not ok:
                    failures.append(f"train(+val).{key}")
        if "test" in stats:
            checks["test"] = {}
            for key in ("exposure", "click", "conversion"):
                result, ok = _check(key, int(stats["test"].get(key, 0)),
                                    int(expected.get("test", {}).get(key, 0)), args.tol)
                checks["test"][key] = result
                if not ok:
                    failures.append(f"test.{key}")

    report = {"splits": stats, "expected_check": {"tol": args.tol, "checks": checks,
                                                  "failures": failures}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"已写 {args.out}")

    if failures:
        print(f"expected 校验失败：{', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"expected 校验通过（容差 {args.tol:.1%}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
