"""Hive 四层表（ODS/DWD/DWS/ADS）+ Spark 原生算子解析。

口径来源：
- doc/202610-选题/02-研究方案摘要.md §5（分层、存储格式、原生算子、调优）
- doc/202610-选题/06-数据清洗与构建方案.md §5/§7/§9（清洗规则与对账）
- doc/202610-选题/08-数据探测记录.md（列结构、分隔符）

设计要点：
1. ODS 只做类型转换，ORC + Snappy，按 split 分区；
2. DWD 拆「曝光明细（10 个骨架域）+ 用户维表（12 个公共域）」，避免用户特征在曝光行上
   重复约 58 倍；join 与编码一起放在 ADS；
3. DWS 用 Spark 原生算子从 DWD 独立重算词表，与 20_processed/vocab.json 逐项对账；
   ADS 编码以 vocab.json 的 idx 为准，保证与 pyarrow 产物逐位一致；
4. ADS 用 22 域 × K=8 int32、C2 清洗、md5 哈希 user_id，Parquet，按 split 分区；
5. 全程不用 Python UDF，解析用 split/filter/transform 等高阶函数。

用法（在项目根 E:\\BISHE 下）：
    spark-submit src\\spark\\hive_pipeline.py --stage check
    spark-submit src\\spark\\hive_pipeline.py --stage ods
    spark-submit src\\spark\\hive_pipeline.py --stage dwd
    spark-submit src\\spark\\hive_pipeline.py --stage dws
    spark-submit --driver-memory 6g src\\spark\\hive_pipeline.py --stage ads
冒烟：追加 `--limit 200000`（DWS 词表与 ADS 对账在 limit>0 时只打印、不判定）。
"""

import argparse
import hashlib
import json
import time
from datetime import datetime
from functools import reduce
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data" / "aliccp"
RAW = DATA / "00_raw"
PROCESSED = DATA / "20_processed"
WAREHOUSE = DATA / "15_warehouse"
STAGING = WAREHOUSE / "_staging"
REPORTS = DATA / "90_reports" / "04_warehouse"

DB = "cvr_debias"
K = 8
PAD_ID = 0
OOV_ID = 1
MIN_COUNT = 5
SENTINEL = 2147483647
# 与 prepare_aliccp._val_cutoff 完全一致：int(0.05 * (1 << 63))，float 精度下为 ...816
VAL_CUTOFF = 461168601842738816

SEP_FEATURE = "\x01"
SEP_FIELD = "\x02"
SEP_WEIGHT = "\x03"

# 22 个建模域：12 个公共域 + 10 个骨架域；101 全量 OOV 已剔除（见 12 号记录 E7）
SKELETON_FIELDS = ["205", "206", "207", "210", "216", "301", "508", "509", "702", "853"]
COMMON_FIELDS = ["109_14", "110_14", "121", "122", "124", "125", "126",
                 "127", "127_14", "128", "129", "150_14"]
SPARSE_FIELDS = COMMON_FIELDS + SKELETON_FIELDS

# 对账基线：raw 为清洗前（12 号记录 E0/E1），clean 为清洗后（E13）
EXPECTED_RAW = {
    "train": {"rows": 42_300_135, "click": 1_644_256, "conversion": 9_032, "users": 730_600},
    "test": {"rows": 43_016_840, "click": 1_673_447, "conversion": 9_421, "users": 884_212},
}
EXPECTED_CLEAN = {
    "train": {"rows": 40_184_732, "click": 1_562_388, "conversion": 8_364, "users": 693_709},
    "val": {"rows": 2_115_173, "click": 81_868, "conversion": 438, "users": 36_891},
    "test": {"rows": 43_016_614, "click": 1_673_447, "conversion": 9_195, "users": 884_212},
}


# ---------------------------------------------------------------- 基础设施

def build_spark(app: str) -> SparkSession:
    spark = (
        SparkSession.builder.appName(app)
        .config("spark.sql.warehouse.dir", "E:/spark/warehouse")
        .config("spark.sql.catalogImplementation", "hive")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .enableHiveSupport()
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def ensure_db(spark: SparkSession) -> None:
    WAREHOUSE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {DB} LOCATION '{WAREHOUSE.as_uri()}'")


def save_table(spark: SparkSession, df, table: str, fmt: str = "orc",
               partition_by: str | None = None) -> None:
    """重建整张表（先 DROP，保证重跑幂等）。"""
    full = f"{DB}.{table}"
    spark.sql(f"DROP TABLE IF EXISTS {full}")
    writer = df.write.mode("overwrite").format(fmt)
    if partition_by:
        writer = writer.partitionBy(partition_by)
    writer.saveAsTable(full)


def write_report(name: str, payload: dict) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = REPORTS / f"{name}_{ts}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] {path}")
    return path


def parse_kv(col):
    """特征串 → array<struct<f,v>>；field_id 保留字符串（109_14 等复合串）。"""
    return F.transform(
        F.split(col, SEP_FEATURE),
        lambda x: F.struct(
            F.split(x, SEP_FIELD).getItem(0).alias("f"),
            F.split(F.split(x, SEP_FIELD).getItem(1), SEP_WEIGHT).getItem(0).alias("v"),
        ),
    )


def values_of(pairs, fid: str):
    return F.transform(
        F.filter(pairs, lambda p: p["f"] == fid),
        lambda p: p["v"],
    )


def with_user_hash(df):
    """等价于 prepare_aliccp._hash_user：md5 前 8 字节小端，保留 63 位。

    md5 只算一次并落成 _md5 列；按字节逆序拼 hex → decimal → 取模 2^63。
    """
    df = df.withColumn("_md5", F.substring(F.md5(F.col("user_id")), 1, 16))
    # 字节序 b0..b7 → 逆序 b7..b0，得到 int.from_bytes(b0..b7, "little") 的大端 hex
    parts = [F.substring(F.col("_md5"), k, 2) for k in (15, 13, 11, 9, 7, 5, 3, 1)]
    dec = F.conv(F.concat(*parts), 16, 10).cast("decimal(38,0)")
    h = F.pmod(dec, F.lit("9223372036854775808").cast("decimal(38,0)")).cast("long")
    return df.withColumn("_uhash", h).drop("_md5")


def _py_hash(uid: str) -> int:
    digest = hashlib.md5(uid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def read_raw_csv(spark: SparkSession, path: Path, cols: list[str], limit: int = 0):
    df = (
        spark.read.option("header", False)
        .option("inferSchema", False)
        .option("sep", ",")
        .csv(path.as_posix())
    )
    renamed = df.toDF(*cols)
    return renamed.limit(limit) if limit else renamed


def limit_per_split(df, limit: int):
    """冒烟时每个 split 各取 limit 行，避免 limit 只落到某个分区。"""
    if not limit:
        return df
    parts = [df.where(F.col("split") == s).limit(limit) for s in ("train", "test")]
    return reduce(lambda a, b: a.unionByName(b), parts)


# ---------------------------------------------------------------- ODS

def stage_check(spark: SparkSession, n: int = 2000) -> dict:
    """哈希表达式与 Python 实现逐值比对；不一致则后续全部无意义。"""
    path = RAW / "train" / "common_features_train.csv"
    uids = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            uids.append((line.split(",", 1)[0],))
    df = spark.createDataFrame(uids, ["user_id"])
    df = with_user_hash(df)
    bad = [(r["user_id"], r["_uhash"], _py_hash(r["user_id"])) for r in df.collect()
           if r["_uhash"] != _py_hash(r["user_id"])]
    print(f"[check] 哈希比对 {len(uids)} 个 user_id，不一致 {len(bad)} 个")
    if bad:
        print("[check] 前 5 个不一致:", bad[:5])
    return {"checked": len(uids), "mismatch": len(bad), "samples": bad[:5]}


def stage_ods(spark: SparkSession, limit: int = 0) -> dict:
    t0 = time.time()
    # 两个 split 先 union 再一次性写表；逐 split 写会把上一个分区 DROP 掉
    skel_parts, common_parts = [], []
    for split in ("train", "test"):
        skel_parts.append(read_raw_csv(
            spark, RAW / split / f"sample_skeleton_{split}.csv",
            ["row_no", "click", "conversion", "user_id", "extra_int", "feature_str"],
            limit=limit,
        ).select(
            F.col("row_no").cast("long"),
            F.col("click").cast("int"),
            F.col("conversion").cast("int"),
            F.col("user_id"),
            F.col("extra_int").cast("int"),
            F.col("feature_str"),
            F.lit(split).alias("split"),
        ))
        common_parts.append(read_raw_csv(
            spark, RAW / split / f"common_features_{split}.csv",
            ["user_id", "feature_count", "feature_str"], limit=limit,
        ).select(
            F.col("user_id"),
            F.col("feature_count").cast("int"),
            F.col("feature_str"),
            F.lit(split).alias("split"),
        ))
    save_table(spark, reduce(lambda a, b: a.unionByName(b), skel_parts),
               "ods_aliccp_skeleton", "orc", "split")
    save_table(spark, reduce(lambda a, b: a.unionByName(b), common_parts),
               "ods_aliccp_common", "orc", "split")

    splits = {}
    for split in ("train", "test"):
        row = spark.sql(f"""
            SELECT split, COUNT(*) AS rows, SUM(click) AS clicks,
                   SUM(conversion) AS convs
            FROM {DB}.ods_aliccp_skeleton WHERE split = '{split}' GROUP BY split
        """).collect()[0]
        common_row = spark.sql(f"""
            SELECT COUNT(*) AS rows, COUNT(DISTINCT user_id) AS users
            FROM {DB}.ods_aliccp_common WHERE split = '{split}'
        """).collect()[0]
        splits[split] = {
            "rows": row["rows"], "click": row["clicks"], "conversion": row["convs"],
            "common_rows": common_row["rows"], "users": common_row["users"],
        }
        print(f"[ods] {split}: rows={row['rows']:,} click={row['clicks']:,} "
              f"conv={row['convs']:,} users={common_row['users']:,}")
        if limit == 0:
            exp = EXPECTED_RAW[split]
            assert row["rows"] == exp["rows"], f"{split} 行数 {row['rows']} != {exp['rows']}"
            assert row["clicks"] == exp["click"], f"{split} 点击 {row['clicks']} != {exp['click']}"
            assert row["convs"] == exp["conversion"], f"{split} 转化 {row['convs']} != {exp['conversion']}"
            assert common_row["rows"] == exp["users"], f"{split} 用户行数 {common_row['rows']} != {exp['users']}"
            assert common_row["users"] == exp["users"], f"{split} 唯一用户 {common_row['users']} != {exp['users']}"
    payload = {"stage": "ods", "limit": limit, "splits": splits,
               "elapsed_sec": round(time.time() - t0, 1)}
    write_report("ods_report", payload)
    return payload


# ---------------------------------------------------------------- DWD

def stage_dwd(spark: SparkSession, limit: int = 0) -> dict:
    t0 = time.time()
    skel = spark.table(f"{DB}.ods_aliccp_skeleton")
    skel = limit_per_split(skel, limit)
    skel = skel.withColumn("_pairs", parse_kv(F.col("feature_str")))
    for fid in SKELETON_FIELDS:
        skel = skel.withColumn(f"f_{fid}", values_of(F.col("_pairs"), fid))
    exposure = skel.select(
        "row_no", "click", "conversion", "user_id", "split",
        *[f"f_{fid}" for fid in SKELETON_FIELDS],
    )
    save_table(spark, exposure, "dwd_aliccp_exposure", "orc", "split")

    common = spark.table(f"{DB}.ods_aliccp_common")
    # 公共特征是用户维表，冒烟时不截断，否则曝光侧 join 会大面积丢行
    common = common.withColumn("_pairs", parse_kv(F.col("feature_str")))
    for fid in COMMON_FIELDS:
        common = common.withColumn(f"f_{fid}", values_of(F.col("_pairs"), fid))
    user_feat = common.select(
        "user_id", "split", *[f"f_{fid}" for fid in COMMON_FIELDS]
    )
    save_table(spark, user_feat, "dwd_aliccp_user_feature", "orc", "split")

    stats = {}
    for split in ("train", "test"):
        e = spark.sql(f"""
            SELECT COUNT(*) AS rows, SUM(click) AS clicks, SUM(conversion) AS convs
            FROM {DB}.dwd_aliccp_exposure WHERE split = '{split}'
        """).collect()[0]
        u = spark.sql(f"""
            SELECT COUNT(*) AS rows, COUNT(DISTINCT user_id) AS users
            FROM {DB}.dwd_aliccp_user_feature WHERE split = '{split}'
        """).collect()[0]
        stats[split] = {"rows": e["rows"], "click": e["clicks"], "conversion": e["convs"],
                        "user_rows": u["rows"], "users": u["users"]}
        print(f"[dwd] {split}: rows={e['rows']:,} click={e['clicks']:,} "
              f"conv={e['convs']:,} users={u['users']:,}")
        if limit == 0:
            exp = EXPECTED_RAW[split]
            assert e["rows"] == exp["rows"], f"{split} DWD 行数 {e['rows']} != {exp['rows']}"
            assert e["clicks"] == exp["click"]
            assert e["convs"] == exp["conversion"]
            assert u["users"] == exp["users"], f"{split} 用户数 {u['users']} != {exp['users']}"
    payload = {"stage": "dwd", "limit": limit, "splits": stats,
               "elapsed_sec": round(time.time() - t0, 1)}
    write_report("dwd_report", payload)
    return payload


# ---------------------------------------------------------------- DWS

def _vocab_ndjson() -> Path:
    """vocab.json → NDJSON（DWS 对账与 ADS 编码的 idx 权威来源）。"""
    STAGING.mkdir(parents=True, exist_ok=True)
    out = STAGING / "vocab.ndjson"
    if out.exists():
        return out
    payload = json.loads((PROCESSED / "vocab.json").read_text(encoding="utf-8"))
    with out.open("w", encoding="utf-8") as f:
        for fid, mapping in payload["vocab"].items():
            for value, idx in mapping.items():
                f.write(json.dumps({"field_id": fid, "feat_val": value, "idx": idx},
                                   ensure_ascii=False) + "\n")
    return out


def _read_vocab(spark: SparkSession):
    """强制 feat_val 为 string：取值形如数字时，inferSchema 会把它们推断成数值。"""
    schema = T.StructType([
        T.StructField("field_id", T.StringType()),
        T.StructField("feat_val", T.StringType()),
        T.StructField("idx", T.IntegerType()),
    ])
    return spark.read.schema(schema).json(_vocab_ndjson().as_posix())


def stage_dws(spark: SparkSession, limit: int = 0) -> dict:
    t0 = time.time()
    # 词表：只统计 train，按「频次降序、同频取值字典序」编号，idx 从 2 开始
    exp_df = spark.table(f"{DB}.dwd_aliccp_exposure").where("split = 'train'")
    usr_df = spark.table(f"{DB}.dwd_aliccp_user_feature").where("split = 'train'")
    if limit:
        exp_df = exp_df.limit(limit)

    def explode_fields(df, fields):
        parts = [
            df.select(F.lit(fid).alias("field_id"), F.explode(F.col(f"f_{fid}")).alias("v"))
            for fid in fields
        ]
        return reduce(lambda a, b: a.unionByName(b), parts)

    items = explode_fields(exp_df, SKELETON_FIELDS).unionByName(
        explode_fields(usr_df, COMMON_FIELDS)
    ).where(F.col("v").isNotNull())
    counts = items.groupBy("field_id", "v").count().where(F.col("count") >= MIN_COUNT)
    w = Window.partitionBy("field_id").orderBy(F.desc("count"), F.asc("v"))
    vocab = (counts.withColumn("idx", (F.row_number().over(w) + 1).cast("int"))
             .select("field_id", F.col("v").alias("feat_val"), "count", "idx"))
    save_table(spark, vocab.coalesce(1), "dws_aliccp_vocab", "orc", "field_id")

    # 用户级聚合（C8 异常用户排查与论文画像）
    user_stat = spark.sql(f"""
        SELECT user_id, split, COUNT(*) AS exposures, SUM(click) AS clicks,
               SUM(conversion) AS conversions
        FROM {DB}.dwd_aliccp_exposure GROUP BY user_id, split
    """)
    save_table(spark, user_stat, "dws_aliccp_user_stat", "orc")

    # 与 vocab.json 逐项对账：读回落盘结果，避免重新计算整条 explode 链路
    vocab_tbl = spark.table(f"{DB}.dws_aliccp_vocab").select("field_id", "feat_val", "idx")
    vj = _read_vocab(spark).select("field_id", "feat_val",
                                   F.col("idx").alias("idx_ref"))
    a, b = vocab_tbl.alias("a"), vj.alias("b")
    joined = a.join(b, ["field_id", "feat_val"], "full_outer")
    bad = joined.where(
        F.col("a.idx").isNull() | F.col("b.idx_ref").isNull()
        | (F.col("a.idx") != F.col("b.idx_ref"))
    )
    n_bad = bad.count()
    bad_samples = bad.limit(20).collect()
    vocab_rows = vocab_tbl.count()
    dims_a = {r["field_id"]: r["count"] for r in vocab_tbl.groupBy("field_id").count().collect()}
    dims_b = {r["field_id"]: r["count"] for r in vj.groupBy("field_id").count().collect()}
    dim_diff = {k: (dims_a.get(k), dims_b.get(k)) for k in set(dims_a) | set(dims_b)
                if dims_a.get(k) != dims_b.get(k)}
    print(f"[dws] 词表 {vocab_rows:,} 行，与 vocab.json 不一致 {n_bad:,}，域维度差异 {len(dim_diff)}")
    payload = {
        "stage": "dws", "limit": limit,
        "vocab_rows": vocab_rows, "mismatch_vs_vocab_json": n_bad,
        "mismatch_samples": [r.asDict() for r in bad_samples],
        "field_dim_diff": {k: list(v) for k, v in dim_diff.items()},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    write_report("dws_report", payload)
    if limit == 0:
        assert n_bad == 0, f"词表与 vocab.json 有 {n_bad} 处不一致"
        assert not dim_diff, f"域维度不一致：{dim_diff}"
    return payload


# ---------------------------------------------------------------- ADS

def _load_vocab_py() -> dict:
    """driver 端读 vocab.json，按域拆成 dict，供 pandas UDF 广播。"""
    return json.loads((PROCESSED / "vocab.json").read_text(encoding="utf-8"))["vocab"]


def _make_encoder(bc_map):
    """array<string> → array<int>[8] 的 Arrow 批量编码器。

    语义与 prepare_aliccp._encode_values 逐位一致：0=PAD、1=OOV、>=2 真实 id；
    去重后真实 id 升序、OOV 排最后，截断到 K，尾部补 0。

    为何不用原生 element_at(map, v)：Spark 的 map 是 UnsafeArrayData，查找为线性扫描，
    百万级词表（110_14 有 135 万项）下单个值平均几十万次比较，200k 用户跑 11 分钟仍未完成；
    故词表映射改用 Arrow 批量 UDF，解析仍保持原生算子。
    """
    import numpy as np
    import pandas as pd

    pad = [PAD_ID] * K

    def encode_series(s: pd.Series) -> pd.Series:
        mapping = bc_map.value
        out = []
        for vals in s:
            if not isinstance(vals, (list, tuple, np.ndarray)) or len(vals) == 0:
                out.append(pad.copy())
                continue
            ids = []
            for v in vals:
                idx = mapping.get(v)
                ids.append(OOV_ID if idx is None else idx)
            uniq = sorted(set(ids), key=lambda x: (x == OOV_ID, x))[:K]
            if len(uniq) < K:
                uniq.extend([PAD_ID] * (K - len(uniq)))
            out.append(uniq)
        return pd.Series(out, dtype=object)

    return F.pandas_udf(encode_series, T.ArrayType(T.IntegerType()))


def _encode_fields(df, fields: list[str], vocab: dict):
    for fid in fields:
        bc = df.sparkSession.sparkContext.broadcast(vocab[fid])
        df = df.withColumn(f"f_{fid}", _make_encoder(bc)(F.col(f"f_{fid}")))
    return df


def stage_ads(spark: SparkSession, limit: int = 0, files_per_split: int = 16) -> dict:
    t0 = time.time()
    vocab = _load_vocab_py()

    # 1) 用户侧预编码（12 域）：用户特征只依赖 user_id，按用户编码一次与逐行编码等价
    user = spark.table(f"{DB}.dwd_aliccp_user_feature")
    user_enc = _encode_fields(user, COMMON_FIELDS, vocab).select(
        "user_id", "split", *[f"f_{fid}" for fid in COMMON_FIELDS]
    )
    user_path = STAGING / "ads_user_encoded"
    if (user_path / "_SUCCESS").exists():
        print("[ads] 复用已编码的用户向量（_staging/ads_user_encoded）")
    else:
        user_enc.write.mode("overwrite").format("parquet").save(user_path.as_uri())

    # 2) 曝光侧：编码 10 个骨架域 → join 用户向量 → C2 清洗 → 哈希分 split
    exp_df = spark.table(f"{DB}.dwd_aliccp_exposure")
    exp_df = limit_per_split(exp_df, limit)
    exp_df = _encode_fields(exp_df, SKELETON_FIELDS, vocab).select(
        "row_no", "click", "conversion", "user_id", "split",
        *[f"f_{fid}" for fid in SKELETON_FIELDS],
    )
    u = spark.read.parquet(user_path.as_uri())
    joined = exp_df.join(u, on=["user_id", "split"], how="inner")

    # C2：conversion=1 且 click=0 属采集错误，三段都删（数量由写完后行数差反推，
    # 避免为计数单独跑一遍 join）
    joined = joined.where(~((F.col("conversion") == 1) & (F.col("click") == 0)))

    # user_id 哈希（GAUC 分组）+ train/val/test 路由
    joined = with_user_hash(joined)
    joined = joined.withColumn(
        "part",
        F.when(F.col("split") == "test", F.lit("test"))
        .when(F.col("_uhash") < F.lit(VAL_CUTOFF), F.lit("val"))
        .otherwise(F.lit("train")),
    )
    out = joined.select(
        *[f"f_{fid}" for fid in SPARSE_FIELDS],
        "click", "conversion", F.col("_uhash").alias("user_id"), "row_no",
        F.col("part").alias("split"),
    ).coalesce(files_per_split)
    save_table(spark, out, "ads_aliccp_train", "parquet", "split")

    stats = {}
    for part in ("train", "val", "test"):
        r = spark.sql(f"""
            SELECT COUNT(*) AS rows, SUM(click) AS clicks, SUM(conversion) AS convs,
                   COUNT(DISTINCT user_id) AS users
            FROM {DB}.ads_aliccp_train WHERE split = '{part}'
        """).collect()[0]
        stats[part] = {"rows": r["rows"], "click": r["clicks"],
                       "conversion": r["convs"], "users": r["users"]}
        print(f"[ads] {part}: rows={r['rows']:,} click={r['clicks']:,} "
              f"conv={r['convs']:,} users={r['users']:,}")
        if limit == 0:
            exp = EXPECTED_CLEAN[part]
            assert r["rows"] == exp["rows"], f"{part} 行数 {r['rows']} != {exp['rows']}"
            assert r["clicks"] == exp["click"], f"{part} 点击 {r['clicks']} != {exp['click']}"
            assert r["convs"] == exp["conversion"], f"{part} 转化 {r['convs']} != {exp['conversion']}"
            assert r["users"] == exp["users"], f"{part} 用户 {r['users']} != {exp['users']}"
    payload = {"stage": "ads", "limit": limit,
               "c2_removed": {
                   "train": EXPECTED_RAW["train"]["rows"] - stats["train"]["rows"] - stats["val"]["rows"],
                   "test": EXPECTED_RAW["test"]["rows"] - stats["test"]["rows"],
               },
               "splits": stats, "elapsed_sec": round(time.time() - t0, 1)}
    write_report("ads_report", payload)
    if limit == 0:
        assert payload["c2_removed"] == {"train": 230, "test": 226}, \
            f"C2 行数与 E13 不一致：{payload['c2_removed']}"
    return payload


# ---------------------------------------------------------------- profile

def stage_verify(spark: SparkSession, limit: int = 20000, shard: str = "part-00000") -> dict:
    """抽 pyarrow 产物分片，与 ADS 逐列逐位比对，确认两条管道编码完全一致。"""
    t0 = time.time()
    ref = spark.read.parquet((PROCESSED / "train" / f"{shard}.parquet").as_uri()).limit(limit)
    ads = spark.table(f"{DB}.ads_aliccp_train").where("split = 'train'")
    j = ref.alias("r").join(ads.alias("a"), on=["row_no"], how="inner")
    rows = j.count()
    cols = [f"f_{fid}" for fid in SPARSE_FIELDS] + ["click", "conversion", "user_id"]
    diff = {}
    for c in cols:
        n = j.where(~(F.col(f"r.{c}") == F.col(f"a.{c}"))).count()
        if n:
            diff[c] = n
    print(f"[verify] 抽样 {limit}，命中 {rows}，差异列 {len(diff)}")
    payload = {"stage": "verify", "shard": shard, "limit": limit, "matched_rows": rows,
               "diff_columns": diff, "elapsed_sec": round(time.time() - t0, 1)}
    write_report("verify_report", payload)
    if rows == 0:
        raise AssertionError("verify 未命中任何行，检查 row_no 或分片")
    if diff:
        raise AssertionError(f"与 pyarrow 产物存在差异：{diff}")
    return payload


def stage_profile(spark: SparkSession, source: str = "ads") -> dict:
    """论文表 1 画像。source=ads 读 ADS；source=processed 读 pyarrow 成品层。

    两者行与标签等价：ADS 是同一批曝光的 Hive 物化，processed 是 pyarrow 产物。
    ADS 未落表时先用 processed 出数，ADS 建好后可用 ads 源重算复核。
    """
    t0 = time.time()
    rows = []
    for split in ("train", "val", "test"):
        if source == "ads":
            df = spark.table(f"{DB}.ads_aliccp_train").where(F.col("split") == split)
        else:
            df = spark.read.parquet((PROCESSED / split).as_uri()).select("click", "conversion")
        r = df.agg(
            F.count("*").alias("exposure"),
            F.sum("click").alias("click"),
            F.sum("conversion").alias("conversion"),
        ).collect()[0]
        exposure, click, conv = r["exposure"], r["click"], r["conversion"]
        rows.append((split, exposure, click, conv,
                     round(click / exposure, 6), round(conv / click, 6),
                     round(conv / exposure, 8)))
    schema = ["split", "exposure", "click", "conversion", "ctr", "cvr_click_space", "ctcvr"]
    profile = spark.createDataFrame(rows, schema)
    save_table(spark, profile, "ads_aliccp_profile", "orc")
    profile.show(truncate=False)
    payload = {"stage": "profile", "source": source,
               "rows": [list(r) for r in rows],
               "elapsed_sec": round(time.time() - t0, 1)}
    write_report("profile_report", payload)
    return payload


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Ali-CCP Hive 四层表 Spark 管道")
    ap.add_argument("--stage", default="all",
                    choices=["check", "ods", "dwd", "dws", "ads", "verify", "profile", "all"])
    ap.add_argument("--limit", type=int, default=0, help="冒烟行数，0 为全量")
    ap.add_argument("--files", type=int, default=16, help="ADS 每个 split 的输出文件数")
    ap.add_argument("--source", choices=["ads", "processed"], default="ads",
                    help="profile 的数据源：ads 或 pyarrow 成品层 processed")
    args = ap.parse_args()

    spark = build_spark(f"cvr-debias-hive-{args.stage}")
    ensure_db(spark)
    try:
        if args.stage in ("check", "all"):
            stage_check(spark)
        if args.stage in ("ods", "all"):
            stage_ods(spark, limit=args.limit)
        if args.stage in ("dwd", "all"):
            stage_dwd(spark, limit=args.limit)
        if args.stage in ("dws", "all"):
            stage_dws(spark, limit=args.limit)
        if args.stage in ("ads", "all"):
            stage_ads(spark, limit=args.limit, files_per_split=args.files)
        if args.stage in ("verify", "all"):
            stage_verify(spark)
        if args.stage in ("profile", "all"):
            stage_profile(spark, source=args.source)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
