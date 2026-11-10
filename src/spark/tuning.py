"""四项调优实验（02 §5.3）：shuffle 分区、AQE、小文件、列裁剪。

只读 DWD/ODS，不写生产表；小文件实验写到 `15_warehouse/_staging/tuning`。
用法：
    spark-submit --driver-memory 6g src\\spark\\tuning.py --exp all
    spark-submit ... --exp shuffle|aqe|files|columns
输出：`90_reports/04_warehouse/tuning_report_*.json`
"""

import argparse
import json
import shutil
import time
from datetime import datetime
from functools import reduce
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data" / "aliccp"
STAGING = DATA / "15_warehouse" / "_staging" / "tuning"
REPORTS = DATA / "90_reports" / "04_warehouse"
DB = "cvr_debias"
SKELETON_FIELDS = ["205", "206", "207", "210", "216", "301", "508", "509", "702", "853"]


def build_spark(app: str) -> SparkSession:
    spark = (
        SparkSession.builder.appName(app)
        .config("spark.sql.warehouse.dir", "E:/spark/warehouse")
        .config("spark.sql.catalogImplementation", "hive")
        .enableHiveSupport()
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    STAGING.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    return spark


def vocab_agg(spark: SparkSession):
    """代表性聚合负载：train 曝光的 10 个骨架域 explode 后按 (域, 取值) 计数。"""
    df = spark.table(f"{DB}.dwd_aliccp_exposure").where("split = 'train'")
    frames = [
        df.select(F.lit(fid).alias("field_id"), F.explode(F.col(f"f_{fid}")).alias("v"))
        for fid in SKELETON_FIELDS
    ]
    items = reduce(lambda a, b: a.unionByName(b), frames).where(F.col("v").isNotNull())
    return items.groupBy("field_id", "v").count()


def exp_shuffle(spark: SparkSession, partitions=(64, 200, 400)) -> list:
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    out = []
    for p in partitions:
        spark.conf.set("spark.sql.shuffle.partitions", str(p))
        t0 = time.time()
        n = vocab_agg(spark).count()
        dt = round(time.time() - t0, 1)
        out.append({"partitions": p, "sec": dt, "distinct_pairs": n})
        print(f"[shuffle] partitions={p} sec={dt} pairs={n:,}")
    return out


def exp_aqe(spark: SparkSession, partitions: int = 200) -> list:
    spark.conf.set("spark.sql.shuffle.partitions", str(partitions))
    out = []
    for aqe in (False, True):
        spark.conf.set("spark.sql.adaptive.enabled", str(aqe).lower())
        spark.conf.set("spark.sql.adaptive.skewJoin.enabled", str(aqe).lower())
        spark.conf.set("spark.sql.adaptive.optimizeSkewsInRebalancePartitions.enabled", str(aqe).lower())
        t0 = time.time()
        n = vocab_agg(spark).count()
        dt = round(time.time() - t0, 1)
        out.append({"aqe": aqe, "sec": dt, "distinct_pairs": n})
        print(f"[aqe] enabled={aqe} sec={dt} pairs={n:,}")
    return out


def exp_files(spark: SparkSession, rows: int = 5_000_000, counts=(4, 16, 64)) -> list:
    base = (
        spark.table(f"{DB}.dwd_aliccp_exposure").where("split = 'train'")
        .select("row_no", "click", "conversion", "user_id",
                *[f"f_{fid}" for fid in SKELETON_FIELDS])
        .limit(rows)
    )
    out = []
    for n in counts:
        d = STAGING / f"files_{n}"
        shutil.rmtree(d, ignore_errors=True)
        t0 = time.time()
        # 用 repartition 而不是 coalesce：limit 后上游只剩 1 个分区，coalesce 不会增加文件数
        base.repartition(n).write.mode("overwrite").parquet(d.as_uri())
        dt = round(time.time() - t0, 1)
        files = sorted(d.rglob("*.parquet"))
        sizes = [f.stat().st_size for f in files]
        item = {
            "repartition": n, "files": len(files),
            "total_mb": round(sum(sizes) / 1e6, 1),
            "min_mb": round(min(sizes) / 1e6, 1),
            "max_mb": round(max(sizes) / 1e6, 1),
            "sec": dt,
        }
        out.append(item)
        print(f"[files] repartition={n} files={item['files']} "
              f"{item['min_mb']}–{item['max_mb']} MB sec={dt}")
    return out


def _read_schema(df) -> str:
    plan = df._jdf.queryExecution().executedPlan().toString()
    for line in plan.splitlines():
        if "ReadSchema" in line:
            return line.strip()
    return ""


def exp_columns(spark: SparkSession, rows: int = 50_000) -> dict:
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    full = spark.sql(f"SELECT * FROM {DB}.dwd_aliccp_exposure WHERE split = 'train'")
    slim = spark.sql(f"SELECT click, conversion FROM {DB}.dwd_aliccp_exposure WHERE split = 'train'")

    t0 = time.time()
    n_full = len(full.limit(rows).collect())
    full_sec = round(time.time() - t0, 1)
    t0 = time.time()
    n_slim = len(slim.limit(rows).collect())
    slim_sec = round(time.time() - t0, 1)
    out = {
        "full_read_schema": _read_schema(full),
        "slim_read_schema": _read_schema(slim),
        "full_sec": full_sec, "slim_sec": slim_sec,
        "full_rows": n_full, "slim_rows": n_slim,
    }
    print(f"[columns] full={full_sec}s slim={slim_sec}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Spark 四项调优实验")
    ap.add_argument("--exp", default="all", choices=["all", "shuffle", "aqe", "files", "columns"])
    args = ap.parse_args()

    spark = build_spark(f"cvr-debias-tuning-{args.exp}")
    payload = {"generated_at": datetime.now().isoformat(timespec="seconds"), "exp": args.exp}
    try:
        if args.exp in ("all", "shuffle"):
            payload["shuffle_partitions"] = exp_shuffle(spark)
        if args.exp in ("all", "aqe"):
            payload["aqe"] = exp_aqe(spark)
        if args.exp in ("all", "files"):
            payload["small_files"] = exp_files(spark)
        if args.exp in ("all", "columns"):
            payload["column_pruning"] = exp_columns(spark)
    finally:
        spark.stop()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = "tuning_files_report" if args.exp == "files" else "tuning_report"
    path = REPORTS / f"{name}_{ts}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] {path}")


if __name__ == "__main__":
    main()
