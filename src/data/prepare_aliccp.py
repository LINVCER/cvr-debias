"""Ali-CCP 预处理流水线：L0 原始 → L1 中间 → L2 成品。

分层约定见 data/aliccp/README.md，逐字节事实见 doc/202610-选题/08-数据探测记录.md。
每个 stage 只写自己那一层，可单独重跑：

    python -m src.data.prepare_aliccp --stage index    # L1：user_id -> 特征字典
    python -m src.data.prepare_aliccp --stage join     # L2：流式 join，落字符串宽表（冒烟用）
    python -m src.data.prepare_aliccp --stage vocab    # L2：只用 train 建词表
    python -m src.data.prepare_aliccp --stage encode   # L2：截断到 K=8 + 映射 int32
    python -m src.data.prepare_aliccp --stage clean    # L2：清洗规则 C1–C8
    python -m src.data.prepare_aliccp --stage all
"""

import argparse
import hashlib
import json
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# 稀疏域清单：实测自 90_reports/01_probe/field_inventory.json（train/test 完全一致）
# 注意：109_14 / 110_14 / 127_14 / 150_14 是复合串，不能当整数处理
SKELETON_FIELDS = ["205", "206", "207", "210", "216", "301", "508", "509", "702", "853"]
# 23 个稀疏域：实测自 90_reports/01_probe/field_inventory.json（train/test 一致）
# 101 = 用户 ID（train 294,882 / test 295,787 人，并集 444,861，见 12-E33b）。
# 它必须按曝光级计数，否则会被 MIN_COUNT=5 误剔：去重表上每值仅 1–4 次，
# join 到曝光后平均 143 次（train 89.18% 的取值 ≥5）。
COMMON_FIELDS = ["101", "109_14", "110_14", "121", "122", "124", "125", "126",
                 "127", "127_14", "128", "129", "150_14"]
SPARSE_FIELDS = COMMON_FIELDS + SKELETON_FIELDS          # 23 个，顺序固定，不可插入
SKELETON_SET = set(SKELETON_FIELDS)

# 真实用户 ID 所在的域：既作为稀疏特征之一，也单独哈希成 `uid` 列供 GAUC 分组。
# `user_id` 列仍是公共特征组索引（历史命名），只负责 val 切分与 join 对账。
USER_ID_FIELD = "101"

# 特征串分隔符（实测）
SEP_FEATURE = "\x01"    # 特征之间
SEP_FIELD = "\x02"      # field_id 与 value
SEP_WEIGHT = "\x03"     # value 与 weight

MIN_COUNT = 5           # 频次 < 5 归 OOV
SHARD_ROWS = 250_000    # K=32 下单片约 0.74 GB 原始 int32，500k 会顶到 1.5 GB
VAL_RATIO = 0.05

ENCODE_MAX_LEN = 32     # 每域最多保留 K 个取值，不足补 0
# K=8 时 110_14（每行平均 250 个取值）只保留 3%，且取的是全局最热门的 8 个，
# 页内个性化被池化抹平（15 号 A2）。K=32 后保留 13%，存储约 +2–4 GB。
PAD_ID = 0              # 缺失/补齐
OOV_ID = 1              # 词表外
ENCODE_BATCH_ROWS = 5_000   # 纯 Python 编码的批大小，控制峰值内存

INDEX_BUCKETS = 32      # 公共特征按 user_id 哈希分桶；单桶约 270 MB，L2 可整桶载入
INDEX_BUCKET_ROWS = 256 # 每桶一次写出的行数，控制峰值内存（32 × 256 行 ≈ 460 MB）

RAW = Path("data/aliccp/00_raw")
INTERIM = Path("data/aliccp/10_interim")
PROCESSED = Path("data/aliccp/20_processed")
REPORTS = Path("data/aliccp/90_reports")

# L1 列：field 是 field_id 字符串（如 109_14），value 是取值字符串原值。
# 这里只做结构化，不做词表映射——OOV 判定必须只用 train 统计。
INDEX_SCHEMA = pa.schema([
    ("user_id", pa.string()),
    ("field", pa.list_(pa.string())),
    ("value", pa.list_(pa.string())),
])

# L2 冒烟宽表：标签 + 22 个字符串列表列，不做映射、不截断
JOIN_SCHEMA = pa.schema(
    [
        ("user_id", pa.string()),
        ("click", pa.int8()),
        ("conversion", pa.int8()),
        ("row_no", pa.int64()),
    ]
    + [pa.field(f"f_{fid}", pa.list_(pa.string())) for fid in SPARSE_FIELDS]
)


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raw() -> dict:
    """L0 只读校验：对照 manifest.json 的字节数与 md5，不符立即停止。"""
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))
    ok = True
    for rec in manifest["files"]:
        p = RAW / rec["file"]
        if not p.exists():
            print(f"[缺文件] {rec['file']}")
            ok = False
            continue
        size, digest = p.stat().st_size, md5sum(p)
        if size != rec["bytes"] or digest != rec["md5"]:
            print(f"[校验失败] {rec['file']} 大小 {size} md5 {digest}")
            ok = False
    if not ok:
        raise SystemExit("L0 原始层与 manifest 不一致，禁止继续——先重新解压，不要手工修补")
    print("[ok] L0 四个文件 md5 与 manifest 一致")
    return manifest


def _parse_feature_items(text: str) -> tuple[list[str], list[str]]:
    """把 `field<02>value<03>weight<01>...` 拆成两个平行数组，weight 丢弃。"""
    fields: list[str] = []
    values: list[str] = []
    for item in text.split(SEP_FEATURE):
        if not item:
            continue
        fid, sep, rest = item.partition(SEP_FIELD)
        if not sep:
            raise ValueError(f"缺少 field/value 分隔符：{item[:40]!r}")
        fields.append(fid)
        values.append(rest.partition(SEP_WEIGHT)[0])
    return fields, values


def _flush_index_bucket(writer: pq.ParquetWriter, rows: list[tuple]) -> None:
    """把桶内缓冲的行写成 Parquet，然后清空缓冲。"""
    table = pa.table({
        "user_id": pa.array([r[0] for r in rows], type=pa.string()),
        "field": pa.array([r[1] for r in rows], type=pa.list_(pa.string())),
        "value": pa.array([r[2] for r in rows], type=pa.list_(pa.string())),
    })
    writer.write_table(table)
    rows.clear()


def _build_index_split(split: str, limit: int = 0) -> dict:
    """单切分：流式读公共特征表，按 user_id 哈希写入 32 个分桶。"""
    src = RAW / split / f"common_features_{split}.csv"
    out_dir = INTERIM / "common_index" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("part-*.parquet"):
        stale.unlink()

    t0 = time.time()
    buffers: list[list[tuple]] = [[] for _ in range(INDEX_BUCKETS)]
    writers = [
        pq.ParquetWriter(out_dir / f"part-{b:02d}.parquet", INDEX_SCHEMA, compression="zstd")
        for b in range(INDEX_BUCKETS)
    ]

    rows_in = rows_out = malformed = count_mismatch = empty_rows = duplicate = 0
    feature_total = 0
    seen: set[str] = set()
    field_row_hist: dict[str, int] = {}
    unexpected: dict[str, int] = {}
    per_bucket = [0] * INDEX_BUCKETS
    common_set = set(COMMON_FIELDS)

    try:
        with src.open("rb") as f:
            for raw in f:
                if limit and rows_in >= limit:
                    break
                rows_in += 1

                parts = raw.rstrip(b"\r\n").split(b",", 2)
                if len(parts) != 3:
                    malformed += 1
                    continue
                uid_b, cnt_b, feat_b = parts

                try:
                    uid = uid_b.decode("ascii")
                    declared = int(cnt_b)
                    fields, values = _parse_feature_items(feat_b.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    malformed += 1
                    continue

                # 08 文档第 8 节：声明特征数与实际解析数不一致 → 记录并跳过
                if declared != len(fields):
                    count_mismatch += 1
                    continue

                if not fields:
                    empty_rows += 1
                if uid in seen:
                    duplicate += 1
                else:
                    seen.add(uid)

                feature_total += len(fields)
                for fid in set(fields):
                    field_row_hist[fid] = field_row_hist.get(fid, 0) + 1
                    if fid not in common_set:
                        unexpected[fid] = unexpected.get(fid, 0) + 1

                bucket = zlib.crc32(uid_b) % INDEX_BUCKETS
                buffers[bucket].append((uid, fields, values))
                per_bucket[bucket] += 1
                rows_out += 1

                if len(buffers[bucket]) >= INDEX_BUCKET_ROWS:
                    _flush_index_bucket(writers[bucket], buffers[bucket])
    finally:
        for b, writer in enumerate(writers):
            if buffers[b]:
                _flush_index_bucket(writer, buffers[b])
            writer.close()

    files = sorted(out_dir.glob("part-*.parquet"))
    return {
        "split": split,
        "source": str(src),
        "rows_in": rows_in,
        "rows_written": rows_out,
        "malformed_rows": malformed,
        "count_mismatch_skipped": count_mismatch,
        "empty_feature_rows": empty_rows,
        "duplicate_user_rows": duplicate,
        "unique_users": len(seen),
        "features_parsed": feature_total,
        "field_row_histogram": dict(sorted(field_row_hist.items(), key=lambda kv: -kv[1])),
        "unexpected_field_occurrences": unexpected,
        "per_bucket_rows": per_bucket,
        "output_files": len(files),
        "output_bytes": sum(p.stat().st_size for p in files),
        "elapsed_sec": round(time.time() - t0, 1),
    }


def build_common_index(splits: list[str] | None = None, limit: int = 0) -> dict:
    """L1：把公共特征表建成磁盘字典，按 user_id 分桶写 Parquet。

    禁止两表全载入内存：common_features_train.csv 单行可达 12 KB、1025 个特征。
    落 10_interim/common_index/{split}/part-{桶号}.parquet，列为
    (user_id, field[], value[])：field 是 field_id 字符串，value 是取值原值。
    词表在 L2 才映射，保证 OOV 判定只用 train 统计。weight 字段在此丢弃。

    limit > 0 时每个文件只读前 limit 行，用于冒烟验证；正式产出一律全量。
    """
    splits = splits or ["train", "test"]
    print(f"[index] 分桶 {INDEX_BUCKETS} 个，每桶每次写 {INDEX_BUCKET_ROWS} 行，split={splits}")
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "buckets": INDEX_BUCKETS,
        "bucket_rows": INDEX_BUCKET_ROWS,
        "limit": limit,
        "splits": {},
    }
    for split in splits:
        stats = _build_index_split(split, limit=limit)
        report["splits"][split] = stats
        skipped = stats["malformed_rows"] + stats["count_mismatch_skipped"]
        print(
            f"[index] {split}: 读 {stats['rows_in']:,} → 写 {stats['rows_written']:,} 行，"
            f"跳过 {skipped:,}，{stats['output_bytes'] / 2**30:.2f} GiB，"
            f"{stats['elapsed_sec']}s"
        )

    REPORTS.joinpath("02_cleaning").mkdir(parents=True, exist_ok=True)
    out = REPORTS / "02_cleaning" / f"index_report_{datetime.now():%Y%m%d-%H%M}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[index] 报告 {out}")
    return report


def _hash_user(uid: str) -> int:
    """稳定哈希到 int64：md5 前 8 字节小端，保留 63 位，供 GAUC 分组。"""
    digest = hashlib.md5(uid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


_HASH_CACHE: dict[str, int] = {}


def _hash_user_cached(uid: str) -> int:
    """带缓存的版本：一件样本要哈希组索引与用户 ID 两列，42M 行不能重复算 md5。

    组索引约 73 万、用户约 29 万，缓存后每个进程只算百万次以内。
    """
    value = _HASH_CACHE.get(uid)
    if value is None:
        value = _hash_user(uid)
        _HASH_CACHE[uid] = value
    return value


def _load_vocab(path: Path) -> dict:
    """加载并校验词表；id 约定 0=缺失、1=OOV、>=2 真实值，且 id 按频次降序分配。

    `max_len`（K）是编码参数、不是词表属性，故不在此校验——改 K 不需要重建词表。
    """
    if not path.exists():
        raise SystemExit(f"[encode] 缺少词表 {path}，先运行 --stage vocab")
    data = json.loads(path.read_text(encoding="utf-8"))
    vocab = data.get("vocab", {})
    dims = data.get("field_dims", [])
    if len(dims) != len(SPARSE_FIELDS):
        raise ValueError(f"field_dims 长度 {len(dims)} != {len(SPARSE_FIELDS)}")
    missing = [fid for fid in SPARSE_FIELDS if fid not in vocab]
    if missing:
        raise ValueError(f"词表缺少域：{missing}")
    fields = data.get("fields")
    if fields is not None and list(fields) != list(SPARSE_FIELDS):
        raise ValueError(f"词表 fields 与代码 SPARSE_FIELDS 顺序不一致：{fields}")
    return data


def _encode_values(
    values: list[str], field_map: dict[str, int], field_dim: int
) -> tuple[list[int], int, int, bool]:
    """单域编码：返回 (定长 id 列表, 输入值数, OOV 值数, 是否截断)。

    词表 id 按频次降序分配，因此 id 越小频次越高；排序后取前 K 个。
    多值域先去重，OOV 排在真实值之后；不足 K 个补 0。
    """
    if not values:
        return [PAD_ID] * ENCODE_MAX_LEN, 0, 0, False

    ids: list[int] = []
    oov = 0
    for value in values:
        idx = field_map.get(value)
        if idx is None:
            oov += 1
            ids.append(OOV_ID)
        else:
            if idx < 2 or idx >= field_dim:
                raise ValueError(f"词表 id 越界：{value!r} -> {idx}, dim={field_dim}")
            ids.append(idx)

    unique = sorted(set(ids), key=lambda x: (x == OOV_ID, x))
    truncated = len(unique) > ENCODE_MAX_LEN
    kept = unique[:ENCODE_MAX_LEN]
    kept.extend([PAD_ID] * (ENCODE_MAX_LEN - len(kept)))
    return kept, len(values), oov, truncated


ENCODE_SCHEMA = pa.schema(
    [pa.field(f"f_{fid}", pa.list_(pa.int32(), ENCODE_MAX_LEN)) for fid in SPARSE_FIELDS]
    + [
        ("click", pa.int8()),
        ("conversion", pa.int8()),
        ("user_id", pa.int64()),
        ("row_no", pa.int64()),
        ("uid", pa.int64()),          # 哈希后的真实用户 ID（域 101），只用于 GAUC
    ]
)


def _encode_batch(
    batch: pa.RecordBatch, vocab: dict, field_stats: dict
) -> tuple[pa.Table, dict]:
    """编码一个 batch；返回输出表和本批的标签/用户统计。"""
    columns: dict[str, pa.Array] = {}
    user_values: list | None = None
    for pos, fid in enumerate(SPARSE_FIELDS):
        name = f"f_{fid}"
        idx = batch.schema.get_field_index(name)
        if idx < 0:
            raise ValueError(f"_join 产物缺少列 {name}，请确认 join 输出契约")
        raw_values = batch.column(idx).to_pylist()
        if fid == USER_ID_FIELD:
            user_values = raw_values
        flat: list[int] = []
        stat = field_stats[fid]
        field_map = vocab["vocab"][fid]
        field_dim = int(vocab["field_dims"][pos])
        for values in raw_values:
            values = values or []
            kept, n_values, oov, truncated = _encode_values(values, field_map, field_dim)
            flat.extend(kept)
            stat["missing_rows"] += 1 if n_values == 0 else 0
            stat["input_values"] += n_values
            stat["oov_values"] += oov
            stat["truncated_rows"] += 1 if truncated else 0
            stat["kept_values"] += sum(1 for x in kept if x != PAD_ID)
        columns[name] = pa.FixedSizeListArray.from_arrays(
            pa.array(flat, type=pa.int32()), ENCODE_MAX_LEN
        )

    clicks = batch.column(batch.schema.get_field_index("click")).to_pylist()
    conversions = batch.column(batch.schema.get_field_index("conversion")).to_pylist()
    uids = batch.column(batch.schema.get_field_index("user_id")).to_pylist()
    row_nos = batch.column(batch.schema.get_field_index("row_no")).to_pylist()

    hashed = [_hash_user_cached(str(uid)) for uid in uids]
    # 真实用户 ID：101 取首个值；缺失时退回组索引（口径退化但不会崩）
    if user_values is None:
        user_hashed = hashed
    else:
        user_hashed = [
            _hash_user_cached(str(v[0])) if v else g
            for v, g in zip(user_values, hashed)
        ]
    columns["click"] = pa.array(clicks, type=pa.int8())
    columns["conversion"] = pa.array(conversions, type=pa.int8())
    columns["user_id"] = pa.array(hashed, type=pa.int64())
    columns["row_no"] = pa.array(row_nos, type=pa.int64())
    columns["uid"] = pa.array(user_hashed, type=pa.int64())

    batch_stats = {
        "rows": len(clicks),
        "click": int(sum(1 for x in clicks if x == 1)),
        "conversion": int(sum(1 for x in conversions if x == 1)),
        "users": set(hashed),
        "real_users": set(user_hashed),
    }
    return pa.table(columns, schema=ENCODE_SCHEMA), batch_stats


def _encode_split(
    split: str, vocab: dict, join_root: Path, out_root: Path, limit: int = 0
) -> dict:
    src_dir = join_root / split
    files = sorted(src_dir.glob("part-*.parquet"))
    if not files:
        raise SystemExit(f"[encode] {src_dir} 下无 _join 产物，先运行 --stage join（冒烟）")

    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("part-*.parquet"):
        stale.unlink()

    field_stats = {
        fid: {"missing_rows": 0, "input_values": 0, "oov_values": 0,
              "truncated_rows": 0, "kept_values": 0}
        for fid in SPARSE_FIELDS
    }
    rows_in = rows_out = clicks = conversions = 0
    users: set[int] = set()
    t0 = time.time()

    for path in files:
        if limit and rows_in >= limit:
            break
        writer = pq.ParquetWriter(out_dir / path.name, ENCODE_SCHEMA, compression="zstd")
        try:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=ENCODE_BATCH_ROWS):
                if limit:
                    remain = limit - rows_in
                    if remain <= 0:
                        break
                    if batch.num_rows > remain:
                        batch = batch.slice(0, remain)
                table, bstats = _encode_batch(batch, vocab, field_stats)
                writer.write_table(table)
                rows_in += bstats["rows"]
                rows_out += table.num_rows
                clicks += bstats["click"]
                conversions += bstats["conversion"]
                users.update(bstats["users"])
        finally:
            writer.close()

    for stat in field_stats.values():
        stat["missing_rate"] = round(stat["missing_rows"] / rows_out, 6) if rows_out else 0.0
        stat["oov_rate"] = round(stat["oov_values"] / stat["input_values"], 6) if stat["input_values"] else 0.0
        stat["truncated_rate"] = round(stat["truncated_rows"] / rows_out, 6) if rows_out else 0.0
        stat["avg_kept"] = round(stat["kept_values"] / rows_out, 4) if rows_out else 0.0

    out_files = sorted(out_dir.glob("part-*.parquet"))
    return {
        "split": split,
        "source_files": len(files),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "unique_users": len(users),
        "labels": {
            "exposure": rows_out,
            "click": clicks,
            "conversion": conversions,
            "ctr": round(clicks / rows_out, 6) if rows_out else 0.0,
            "cvr_click_space": round(conversions / clicks, 6) if clicks else 0.0,
            "ctcvr": round(conversions / rows_out, 8) if rows_out else 0.0,
        },
        "fields": field_stats,
        "max_len": ENCODE_MAX_LEN,      # 编码参数（K），不属于词表
        "output_files": len(out_files),
        "output_bytes": sum(p.stat().st_size for p in out_files),
        "elapsed_sec": round(time.time() - t0, 1),
    }


def encode(
    splits: list[str] | None = None,
    limit: int = 0,
    join_root: Path = PROCESSED / "_join",
    out_root: Path = PROCESSED / "_encoded",
    vocab_path: Path = PROCESSED / "vocab.json",
    report_dir: Path = REPORTS / "02_cleaning",
) -> dict:
    """L2：把 `_join` 字符串宽表编码成定长 int32，每域 K=8。

    输入列：user_id, click, conversion, row_no, f_<field>（list<string>，23 个域）。
    输出列：同标签 + user_id(int64 哈希) + f_<field>（fixed_size_list<int32>[8]）。
    0=缺失/补齐、1=OOV、>=2 真实值；词表 id 按频次降序，截断即取最小 K 个。
    本阶段不删行，rows_in 必须等于 rows_out；变动记录写入 encode_report。
    """
    splits = splits or ["train", "test"]
    vocab = _load_vocab(vocab_path)
    ts = datetime.now()
    report = {
        "generated_at": ts.isoformat(timespec="seconds"),
        "stage": "encode",
        "max_len": ENCODE_MAX_LEN,
        "vocab": str(vocab_path),
        "limit": limit,
        "splits": {},
    }
    for split in splits:
        stats = _encode_split(split, vocab, join_root, out_root, limit=limit)
        report["splits"][split] = stats
        print(
            f"[encode] {split}: {stats['rows_in']:,} 行 → {stats['rows_out']:,}，"
            f"用户 {stats['unique_users']:,}，{stats['output_bytes'] / 2**30:.2f} GiB，"
            f"{stats['elapsed_sec']}s"
        )

    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"encode_report_{ts:%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[encode] 报告 {out}")
    return report


# ----------------------------------------------------------------------------- L2 clean

# clean 只作用在 encode 产物上，规则比 06 §5 收窄：C3/C4/C6/C7 已在 join/encode 上游处理。
VAL_FRAC = 0.05          # train 按 user 哈希切 val 的比例
C5_MAX_MISSING = 11      # 单行缺失域数超过此值（> 半数）即计入 C5 上报
C8_USER_SHARE = 0.005    # 单用户曝光占比超过 0.5% 即上报


def _val_cutoff() -> int:
    """train 切 val 的哈希阈值：user_id(md5 前 8 字节小端) < 阈值 → val。"""
    return int(VAL_FRAC * (1 << 63))


def _clean_labels(exposure: int, click: int, conversion: int) -> dict:
    return {
        "exposure": exposure,
        "click": click,
        "conversion": conversion,
        "ctr": round(click / exposure, 6) if exposure else 0.0,
        "cvr_click_space": round(conversion / click, 6) if click else 0.0,
        "ctcvr": round(conversion / exposure, 8) if exposure else 0.0,
    }


def _clean_split(
    split: str,
    in_dir: Path,
    out_dirs: dict[str, Path],
    train_cutoff: int | None,
) -> dict:
    """清洗一个 split，返回该 split 的变动记录。

    split="train" 时按 user 哈希切成 train/val 两份（out_dirs 含两个键）；
    split="test" 原样清洗后落 test。规则：
      C2 删除 conversion=1 且 click=0；C1 按 (user_id,row_no) 判重（预期 0，作断言）；
      C5/C8 只统计上报，不删除。
    """
    # 只认两位编号的正式分片，避免把冒烟产物 part-00000.parquet 一起读入
    files = sorted(in_dir.glob("part-[0-9][0-9].parquet"))
    if not files:
        raise SystemExit(f"[clean] {in_dir} 下无 encode 产物，先运行 --stage encode")

    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        for stale in d.glob("part-*.parquet"):
            stale.unlink()

    rows_in = click_in = conv_in = 0
    users_in: set[int] = set()
    seen_keys: set[tuple[int, int]] = set()
    c1_dup = c2_bad = 0
    c1_click_removed = c1_conv_removed = 0
    c2_click_removed = c2_conv_removed = 0
    c5_rows = 0
    user_exposure: dict[int, int] = {}
    out_rows: dict[str, int] = {name: 0 for name in out_dirs}
    out_clicks: dict[str, int] = {name: 0 for name in out_dirs}
    out_convs: dict[str, int] = {name: 0 for name in out_dirs}
    out_users: dict[str, set[int]] = {name: set() for name in out_dirs}

    for path in files:
        writers = {name: pq.ParquetWriter(out_dirs[name] / path.name, ENCODE_SCHEMA, compression="zstd")
                   for name in out_dirs}
        try:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=ENCODE_BATCH_ROWS):
                d = batch.to_pydict()
                n = len(d["click"])
                clicks = d["click"]
                convs = d["conversion"]
                uids = d["user_id"]
                rows_in += n
                click_in += sum(clicks)
                conv_in += sum(convs)
                users_in.update(uids)
                for u in uids:
                    user_exposure[u] = user_exposure.get(u, 0) + 1

                keep_idx: list[int] = []
                for i in range(n):
                    key = (uids[i], d["row_no"][i])
                    if key in seen_keys:
                        c1_dup += 1
                        c1_click_removed += clicks[i]
                        c1_conv_removed += convs[i]
                        continue
                    seen_keys.add(key)
                    if convs[i] == 1 and clicks[i] == 0:
                        c2_bad += 1
                        c2_click_removed += clicks[i]
                        c2_conv_removed += convs[i]
                        continue
                    keep_idx.append(i)

                # C5：单行缺失域数 > 阈值（整域全 PAD=0 即缺失）
                feat_lists = [d[f"f_{fid}"] for fid in SPARSE_FIELDS]
                for i in keep_idx:
                    missing = sum(
                        1 for col in feat_lists
                        if not any(col[i])
                    )
                    if missing > C5_MAX_MISSING:
                        c5_rows += 1

                if not keep_idx:
                    continue
                kept = pa.Table.from_batches([batch]).take(pa.array(keep_idx, type=pa.int64()))

                # 路由到输出 split：train 按 user 哈希切 val；test 直达
                if train_cutoff is None:
                    writers["test"].write_table(kept)
                    out_rows["test"] += kept.num_rows
                    kc = kept.column("click").to_pylist()
                    kv = kept.column("conversion").to_pylist()
                    out_clicks["test"] += sum(kc)
                    out_convs["test"] += sum(kv)
                    out_users["test"].update(
                        kept.column("uid" if "uid" in kept.column_names else "user_id").to_pylist()
                    )
                else:
                    uarr = kept.column("user_id").to_pylist()
                    is_val = [u < train_cutoff for u in uarr]
                    for name, mask_flag in (("train", False), ("val", True)):
                        idx = [j for j, f in enumerate(is_val) if f == mask_flag]
                        if not idx:
                            continue
                        sub = kept.take(pa.array(idx, type=pa.int64()))
                        writers[name].write_table(sub)
                        out_rows[name] += sub.num_rows
                        sc = sub.column("click").to_pylist()
                        sv = sub.column("conversion").to_pylist()
                        out_clicks[name] += sum(sc)
                        out_convs[name] += sum(sv)
                        out_users[name].update(
                            sub.column("uid" if "uid" in sub.column_names else "user_id").to_pylist()
                        )
        finally:
            for w in writers.values():
                w.close()

    rows_out = sum(out_rows.values())
    removed = c1_dup + c2_bad
    labels_after = {
        "exposure": rows_out,
        "click": sum(out_clicks.values()),
        "conversion": sum(out_convs.values()),
    }
    # 恒等式校验
    assert rows_in - removed == rows_out, f"行数对账失败 {rows_in}-{removed}!={rows_out}"
    assert click_in - c1_click_removed - c2_click_removed == labels_after["click"], (
        f"点击对账失败 {click_in}-{c1_click_removed}-{c2_click_removed}"
        f"!={labels_after['click']}")
    assert conv_in - c1_conv_removed - c2_conv_removed == labels_after["conversion"], (
        f"转化对账失败 {conv_in}-{c1_conv_removed}-{c2_conv_removed}"
        f"!={labels_after['conversion']}")

    top_users = sorted(user_exposure.items(), key=lambda kv: -kv[1])[:10]
    top_share = [
        {"user_id": u, "exposure": c, "share": round(c / rows_in, 6)}
        for u, c in top_users
    ]

    splits_out = {}
    for name in out_dirs:
        files_out = sorted(out_dirs[name].glob("part-*.parquet"))
        lab = _clean_labels(out_rows[name], out_clicks[name], out_convs[name])
        splits_out[name] = {
            "rows": out_rows[name],
            "unique_users": len(out_users[name]),
            "labels": lab,
            "output_files": len(files_out),
            "output_bytes": sum(p.stat().st_size for p in files_out),
        }

    return {
        "split": split,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "users_in": len(users_in),
        "removed": removed,
        "removed_rate": round(removed / rows_in, 8) if rows_in else 0.0,
        "labels_before": _clean_labels(rows_in, click_in, conv_in),
        "labels_after": labels_after,
        "rules": {
            "C1_dup_rows": c1_dup,
            "C1_click_removed": c1_click_removed,
            "C1_conv_removed": c1_conv_removed,
            "C2_logic_rows": c2_bad,
            "C2_click_removed": c2_click_removed,
            "C2_conv_removed": c2_conv_removed,
            "C5_high_missing_rows": c5_rows,
            "C5_max_missing_threshold": C5_MAX_MISSING,
            "C8_top10_user_share": top_share,
            "C8_report_threshold": C8_USER_SHARE,
        },
        "outputs": splits_out,
    }


def clean(
    splits: list[str] | None = None,
    encoded_root: Path = PROCESSED / "_encoded",
    out_root: Path = PROCESSED,
    report_dir: Path = REPORTS / "02_cleaning",
) -> dict:
    """L2 clean：C2 删除 + C1 断言 + C5/C8 上报，train 按 user 哈希切 val。

    输入 `_encoded/{train,test}`，输出 `{train,val,test}`。
    每条规则按 split 记录前后变动，写 cleaning_report_{ts}.json，只追加不覆盖。
    """
    splits = splits or ["train", "test"]
    ts = datetime.now()
    cutoff = _val_cutoff()
    report = {
        "generated_at": ts.isoformat(timespec="seconds"),
        "stage": "clean",
        "val_frac": VAL_FRAC,
        "val_hash_cutoff": cutoff,
        "splits": {},
    }
    for split in splits:
        in_dir = encoded_root / split
        if split == "train":
            out_dirs = {"train": out_root / "train", "val": out_root / "val"}
            stats = _clean_split(split, in_dir, out_dirs, train_cutoff=cutoff)
        else:
            out_dirs = {"test": out_root / "test"}
            stats = _clean_split(split, in_dir, out_dirs, train_cutoff=None)
        report["splits"][split] = stats
        print(
            f"[clean] {split}: {stats['rows_in']:,} → {stats['rows_out']:,}"
            f"（C1 {stats['rules']['C1_dup_rows']}、C2 {stats['rules']['C2_logic_rows']}），"
            f"C5 上报 {stats['rules']['C5_high_missing_rows']:,} 行"
        )
        for name, o in stats["outputs"].items():
            print(f"  → {name}: {o['rows']:,} 行，用户 {o['unique_users']:,}")

    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"cleaning_report_{ts:%Y%m%d-%H%M%S}.json"
    report_path = out
    # 只追加不覆盖：若同名则加序号
    seq = 1
    while report_path.exists():
        report_path = report_dir / f"cleaning_report_{ts:%Y%m%d-%H%M%S}_{seq}.json"
        seq += 1
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[clean] 报告 {report_path}")
    return report


def _parse_skeleton_line(raw: bytes) -> tuple[int, int, int, str, bytes, dict[str, list[str]]] | None:
    """解析一行骨架表：行号, click, conversion, user_id, 整数列, 特征串。"""
    parts = raw.rstrip(b"\r\n").split(b",", 5)
    if len(parts) != 6:
        return None
    try:
        row_no = int(parts[0])
        click = int(parts[1])
        conversion = int(parts[2])
        uid_b = parts[3]
        uid = uid_b.decode("ascii")
        fields, values = _parse_feature_items(parts[5].decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not uid:
        return None

    skel: dict[str, list[str]] = {}
    for fid, value in zip(fields, values):
        if fid in SKELETON_SET:
            skel.setdefault(fid, []).append(value)
    return row_no, click, conversion, uid, uid_b, skel


def stream_join(split: str = "train", limit: int = 0) -> dict:
    """L2：流式扫骨架表，逐行 join L1 字典，落字符串宽表供冒烟与 encode 使用。

    输出 `20_processed/_join/{split}/part-*.parquet`，列为
    (user_id, click, conversion, row_no, f_<field> × 23)，每个 f_ 列是 list<string>。
    本阶段不做词表映射、不截断、不删行；映射与 K=8 截断统一在 encode 阶段完成。

    当前只支持 `--limit N` 冒烟：先按 `crc32(user_id)%32` 把骨架行分组，
    再逐桶载入 L1 并用 Arrow 过滤出目标用户。全量路径需要先把骨架表落成 32 个桶，
    否则内存里放不下 4230 万行。
    """
    if limit <= 0:
        raise NotImplementedError(
            "全量 join 需要先把骨架表按 crc32(user_id)%32 分桶；"
            "当前仅支持 --limit N 冒烟"
        )

    src = RAW / split / f"sample_skeleton_{split}.csv"
    out_dir = PROCESSED / "_join" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("part-*.parquet"):
        stale.unlink()

    ts = datetime.now()
    t0 = time.time()
    rows_in = malformed = 0
    rows_by_bucket: dict[int, list[tuple]] = {}

    with src.open("rb") as f:
        for raw in f:
            if rows_in >= limit:
                break
            rows_in += 1
            parsed = _parse_skeleton_line(raw)
            if parsed is None:
                malformed += 1
                continue
            row_no, click, conversion, uid, uid_b, skel = parsed
            bucket = zlib.crc32(uid_b) % INDEX_BUCKETS
            rows_by_bucket.setdefault(bucket, []).append(
                (row_no, click, conversion, uid, skel)
            )

    buckets_used = len(rows_by_bucket)
    labels = {"user_id": [], "click": [], "conversion": [], "row_no": []}
    field_values: dict[str, list[list[str]]] = {fid: [] for fid in SPARSE_FIELDS}
    nonempty = {fid: 0 for fid in SPARSE_FIELDS}
    total_values = {fid: 0 for fid in SPARSE_FIELDS}
    joined_rows = 0

    for bucket in sorted(rows_by_bucket):
        part = INTERIM / "common_index" / split / f"part-{bucket:02d}.parquet"
        if not part.exists():
            raise SystemExit(f"[join] 缺少 L1 分桶 {part}，先运行 --stage index")

        rows = rows_by_bucket[bucket]
        target = pa.array(sorted({r[3] for r in rows}), type=pa.string())
        table = pq.read_table(part, columns=["user_id", "field", "value"])
        filtered = table.filter(pc.is_in(table["user_id"], value_set=target))
        d = filtered.to_pydict()
        # 同名域在 L1 里重复出现（用户行为序列），必须聚成 list，不能 dict 覆盖
        common: dict[str, dict[str, list[str]]] = {}
        for uid, fs, vs in zip(d["user_id"], d["field"], d["value"]):
            feats = common.setdefault(uid, {})
            for fid, value in zip(fs, vs):
                feats.setdefault(fid, []).append(value)
        del table, filtered, d

        for row_no, click, conversion, uid, skel in rows:
            merged = dict(skel)
            feats = common.get(uid)
            if feats is not None:
                merged.update(feats)
                joined_rows += 1
            for fid in SPARSE_FIELDS:
                values = merged.get(fid, [])
                field_values[fid].append(values)
                if values:
                    nonempty[fid] += 1
                    total_values[fid] += len(values)
            labels["user_id"].append(uid)
            labels["click"].append(click)
            labels["conversion"].append(conversion)
            labels["row_no"].append(row_no)

        del common
        rows_by_bucket[bucket] = []

    rows_out = len(labels["user_id"])
    if rows_out != rows_in - malformed:
        raise RuntimeError(f"[join] 行数对账失败：{rows_in} - {malformed} != {rows_out}")

    columns: dict[str, pa.Array] = {
        "user_id": pa.array(labels["user_id"], type=pa.string()),
        "click": pa.array(labels["click"], type=pa.int8()),
        "conversion": pa.array(labels["conversion"], type=pa.int8()),
        "row_no": pa.array(labels["row_no"], type=pa.int64()),
    }
    for fid in SPARSE_FIELDS:
        columns[f"f_{fid}"] = pa.array(field_values[fid], type=pa.list_(pa.string()))
    out_path = out_dir / "part-00000.parquet"
    pq.write_table(pa.table(columns, schema=JOIN_SCHEMA), out_path, compression="zstd")

    clicks = sum(labels["click"])
    conversions = sum(labels["conversion"])
    elapsed = time.time() - t0
    field_stats = {
        fid: {
            "nonempty_rows": nonempty[fid],
            "coverage": round(nonempty[fid] / rows_out, 6) if rows_out else 0.0,
            "values": total_values[fid],
            "avg_values": round(total_values[fid] / rows_out, 3) if rows_out else 0.0,
        }
        for fid in SPARSE_FIELDS
    }
    report = {
        "generated_at": ts.isoformat(timespec="seconds"),
        "stage": "join_smoke",
        "split": split,
        "limit": limit,
        "source": str(src),
        "rows_in": rows_in,
        "malformed_rows": malformed,
        "rows_out": rows_out,
        "unique_users": len(set(labels["user_id"])),
        "buckets_used": buckets_used,
        "joined_rows": joined_rows,
        "missing_user_rows": rows_out - joined_rows,
        "join_hit_rate": round(joined_rows / rows_out, 6) if rows_out else 0.0,
        "labels": {
            "exposure": rows_out,
            "click": clicks,
            "conversion": conversions,
            "ctr": round(clicks / rows_out, 6) if rows_out else 0.0,
            "cvr_click_space": round(conversions / clicks, 6) if clicks else 0.0,
            "ctcvr": round(conversions / rows_out, 8) if rows_out else 0.0,
        },
        "fields": field_stats,
        "max_len": ENCODE_MAX_LEN,      # 编码参数（K），不属于词表
        "output": str(out_path),
        "output_bytes": out_path.stat().st_size,
        "elapsed_sec": round(elapsed, 1),
        "rows_per_sec": round(rows_out / elapsed, 1) if elapsed else 0.0,
    }

    report_dir = REPORTS / "02_cleaning"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"join_smoke_report_{ts:%Y%m%d-%H%M%S}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[join] {split} 冒烟 {rows_in:,} 行 → {rows_out:,}，"
        f"命中 {report['join_hit_rate'] * 100:.4f}%，"
        f"{report['rows_per_sec']:,} 行/秒，"
        f"输出 {report['output_bytes'] / 2**20:.1f} MiB"
    )
    print(f"[join] 报告 {report_path}")
    return report


# ----------------------------------------------------------------------------- L2 全量：分桶 + join+encode 一趟

# 骨架分桶产物：每行是 (user_id, click, conversion, row_no) + 10 个骨架域 list<string>
SKEL_BUCKET_SCHEMA = pa.schema(
    [("user_id", pa.string()), ("click", pa.int8()),
     ("conversion", pa.int8()), ("row_no", pa.int64())]
    + [pa.field(f"f_{fid}", pa.list_(pa.string())) for fid in SKELETON_FIELDS]
)


def bucket_skeleton(
    split: str,
    out_root: Path = PROCESSED / "_skel_buckets",
    report_dir: Path = REPORTS / "02_cleaning",
    flush_rows: int = 200_000,
) -> dict:
    """把骨架 CSV 按 crc32(user_id)%32 落成 32 个磁盘桶，供全量 join 逐桶消费。

    流式读 CSV，每攒满 flush_rows 行就把属于各桶的行追加写入对应
    `bucket-XX.parquet`（用 row group 追加而非整文件重写）。只读一遍 CSV。
    """
    src = RAW / split / f"sample_skeleton_{split}.csv"
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("bucket-*.parquet"):
        stale.unlink()

    ts = datetime.now()
    t0 = time.time()
    rows_in = malformed = 0
    # 每桶一个 ParquetWriter，按需惰性打开；按桶缓冲到 flush 再写
    writers: dict[int, pq.ParquetWriter] = {}
    buffers: dict[int, list[tuple]] = {b: [] for b in range(INDEX_BUCKETS)}
    bucket_rows = [0] * INDEX_BUCKETS

    def _flush(bucket: int) -> None:
        rows = buffers[bucket]
        if not rows:
            return
        cols: dict[str, list] = {k: [] for k in
                                 ("user_id", "click", "conversion", "row_no")}
        for fid in SKELETON_FIELDS:
            cols[f"f_{fid}"] = []
        for row_no, click, conversion, uid, skel in rows:
            cols["user_id"].append(uid)
            cols["click"].append(click)
            cols["conversion"].append(conversion)
            cols["row_no"].append(row_no)
            for fid in SKELETON_FIELDS:
                cols[f"f_{fid}"].append(skel.get(fid, []))
        arrays = {
            "user_id": pa.array(cols["user_id"], type=pa.string()),
            "click": pa.array(cols["click"], type=pa.int8()),
            "conversion": pa.array(cols["conversion"], type=pa.int8()),
            "row_no": pa.array(cols["row_no"], type=pa.int64()),
        }
        for fid in SKELETON_FIELDS:
            arrays[f"f_{fid}"] = pa.array(cols[f"f_{fid}"], type=pa.list_(pa.string()))
        table = pa.table(arrays, schema=SKEL_BUCKET_SCHEMA)
        if bucket not in writers:
            writers[bucket] = pq.ParquetWriter(
                out_dir / f"bucket-{bucket:02d}.parquet", SKEL_BUCKET_SCHEMA, compression="zstd")
        writers[bucket].write_table(table)
        bucket_rows[bucket] += table.num_rows
        buffers[bucket] = []

    pending = 0
    with src.open("rb") as f:
        for raw in f:
            rows_in += 1
            parsed = _parse_skeleton_line(raw)
            if parsed is None:
                malformed += 1
                continue
            row_no, click, conversion, uid, uid_b, skel = parsed
            bucket = zlib.crc32(uid_b) % INDEX_BUCKETS
            buffers[bucket].append((row_no, click, conversion, uid, skel))
            pending += 1
            if pending >= flush_rows:
                for b in range(INDEX_BUCKETS):
                    _flush(b)
                pending = 0
                print(f"[bucket] {split} 已处理 {rows_in:,} 行 …", flush=True)
    for b in range(INDEX_BUCKETS):
        _flush(b)
    for w in writers.values():
        w.close()

    out_files = sorted(out_dir.glob("bucket-*.parquet"))
    report = {
        "generated_at": ts.isoformat(timespec="seconds"),
        "stage": "bucket_skeleton",
        "split": split,
        "source": str(src),
        "rows_in": rows_in,
        "malformed_rows": malformed,
        "rows_bucketed": sum(bucket_rows),
        "buckets": len(out_files),
        "bucket_rows": {f"{b:02d}": bucket_rows[b] for b in range(INDEX_BUCKETS) if bucket_rows[b]},
        "output_bytes": sum(p.stat().st_size for p in out_files),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    if report["rows_bucketed"] != rows_in - malformed:
        raise RuntimeError(
            f"[bucket] 行数对账失败：{rows_in} - {malformed} != {report['rows_bucketed']}")
    report_dir.mkdir(parents=True, exist_ok=True)
    rp = report_dir / f"bucket_skeleton_report_{split}_{ts:%Y%m%d-%H%M%S}.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bucket] {split}: {rows_in:,} 行 → {report['rows_bucketed']:,} 入桶，"
          f"0 malformed={malformed}，{report['elapsed_sec']}s，报告 {rp}")
    return report


def _join_encode_bucket(
    split: str,
    b: int,
    bucket_root: str,
    out_root: str,
    vocab_path: str,
) -> dict:
    """单桶 join+encode：读骨架桶与同号 L1 桶，边 merge 边映射 K=8，写 part-{b}.parquet。

    桶与桶之间按 user 完全隔离（bucket = crc32(user_id)%32），因此各桶的
    行数、标签、用户数、逐域统计都可直接相加，供父进程汇总。
    """
    vocab = _load_vocab(Path(vocab_path))
    skel_path = Path(bucket_root) / split / f"bucket-{b:02d}.parquet"
    l1_path = INTERIM / "common_index" / split / f"part-{b:02d}.parquet"
    if not skel_path.exists():
        raise SystemExit(f"[join_encode] 缺骨架桶 {skel_path}，先运行 --stage bucket")
    if not l1_path.exists():
        raise SystemExit(f"[join_encode] 缺 L1 分桶 {l1_path}，先运行 --stage index")

    # L1 桶 → dict[uid][field] = [values]；同名域必须聚成 list，不能覆盖
    l1 = pq.read_table(l1_path, columns=["user_id", "field", "value"]).to_pydict()
    common: dict[str, dict[str, list[str]]] = {}
    for uid, fs, vs in zip(l1["user_id"], l1["field"], l1["value"]):
        feats = common.setdefault(uid, {})
        for fid, value in zip(fs, vs):
            feats.setdefault(fid, []).append(value)
    del l1

    field_stats = {
        fid: {"missing_rows": 0, "input_values": 0, "oov_values": 0,
              "truncated_rows": 0, "kept_values": 0}
        for fid in SPARSE_FIELDS
    }
    out_path = Path(out_root) / split / f"part-{b:02d}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    b_in = b_out = b_miss = clicks = conversions = 0
    users: set[int] = set()

    writer = pq.ParquetWriter(out_path, ENCODE_SCHEMA, compression="zstd")
    try:
        for batch in pq.ParquetFile(skel_path).iter_batches(batch_size=ENCODE_BATCH_ROWS):
            d = batch.to_pydict()
            n = len(d["user_id"])
            b_in += n
            merged_cols: dict[str, list] = {
                "user_id": [], "click": [], "conversion": [], "row_no": []}
            for fid in SPARSE_FIELDS:
                merged_cols[f"f_{fid}"] = []
            for i in range(n):
                uid = d["user_id"][i]
                feats = common.get(uid)
                if feats is None:
                    b_miss += 1
                merged: dict[str, list[str]] = {}
                for fid in SKELETON_FIELDS:
                    v = d[f"f_{fid}"][i]
                    if v:
                        merged[fid] = v
                if feats:
                    merged.update(feats)
                merged_cols["user_id"].append(uid)
                merged_cols["click"].append(d["click"][i])
                merged_cols["conversion"].append(d["conversion"][i])
                merged_cols["row_no"].append(d["row_no"][i])
                for fid in SPARSE_FIELDS:
                    merged_cols[f"f_{fid}"].append(merged.get(fid, []))
            join_arrays = {
                "user_id": pa.array(merged_cols["user_id"], type=pa.string()),
                "click": pa.array(merged_cols["click"], type=pa.int8()),
                "conversion": pa.array(merged_cols["conversion"], type=pa.int8()),
                "row_no": pa.array(merged_cols["row_no"], type=pa.int64()),
            }
            for fid in SPARSE_FIELDS:
                join_arrays[f"f_{fid}"] = pa.array(
                    merged_cols[f"f_{fid}"], type=pa.list_(pa.string()))
            join_batch = pa.table(join_arrays, schema=JOIN_SCHEMA).to_batches(
                max_chunksize=ENCODE_BATCH_ROWS)[0]
            table, bstats = _encode_batch(join_batch, vocab, field_stats)
            writer.write_table(table)
            b_out += table.num_rows
            clicks += bstats["click"]
            conversions += bstats["conversion"]
            users.update(bstats["users"])
    finally:
        writer.close()
    del common

    if b_out != b_in:
        raise RuntimeError(f"[join_encode] 桶 {b} 行数对账失败：{b_in} != {b_out}")
    if b_miss:
        raise RuntimeError(f"[join_encode] 桶 {b} 有 {b_miss} 行 L1 未命中，命中率必须 100%")

    result = {
        "bucket": b,
        "rows_in": b_in,
        "rows_out": b_out,
        "miss": b_miss,
        "click": clicks,
        "conversion": conversions,
        "users": len(users),
        "fields": field_stats,
        "bytes": out_path.stat().st_size,
    }
    # 完成标记：父进程据此断点续跑，中途被杀也不会重跑已完成的桶
    (Path(out_root) / split / f"part-{b:02d}.stats.json").write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def join_encode_full(
    split: str,
    vocab_path: Path = PROCESSED / "vocab.json",
    bucket_root: Path = PROCESSED / "_skel_buckets",
    out_root: Path = PROCESSED / "_encoded",
    buckets: list[int] | None = None,
    report_dir: Path = REPORTS / "02_cleaning",
    workers: int = 1,
    resume: bool = True,
) -> dict:
    """逐桶 join L1 + 当场编码：不落字符串宽表，直接写 _encoded/{split}/part-*.parquet。

    workers>1 时用进程池并行处理互不相干的桶（桶按 user 哈希划分，天然无交集）。
    resume=True 时，有 part-*.parquet + 同名 .stats.json 的桶直接跳过（断点续跑）。
    逐桶对账 rows_in==rows_out；命中率必须 100%，任何 miss 即停。
    """
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket_list = buckets if buckets is not None else list(range(INDEX_BUCKETS))

    # 断点续跑：只保留「产物 + 完成标记」齐全的桶，残缺桶清掉重做
    pending: list[int] = []
    skipped: list[dict] = []
    for b in bucket_list:
        part = out_dir / f"part-{b:02d}.parquet"
        side = out_dir / f"part-{b:02d}.stats.json"
        if resume and part.exists() and side.exists():
            try:
                skipped.append(json.loads(side.read_text(encoding="utf-8")))
                continue
            except (OSError, json.JSONDecodeError):
                pass
        part.unlink(missing_ok=True)
        side.unlink(missing_ok=True)
        pending.append(b)

    ts = datetime.now()
    t0 = time.time()
    total_in = total_out = total_miss = 0
    all_clicks = all_convs = 0
    total_users = 0
    field_stats = {
        fid: {"missing_rows": 0, "input_values": 0, "oov_values": 0,
              "truncated_rows": 0, "kept_values": 0}
        for fid in SPARSE_FIELDS
    }
    done = 0

    def _merge(result: dict) -> None:
        nonlocal total_in, total_out, total_miss, all_clicks, all_convs, total_users, done
        total_in += result["rows_in"]
        total_out += result["rows_out"]
        total_miss += result["miss"]
        all_clicks += result["click"]
        all_convs += result["conversion"]
        total_users += result["users"]
        for fid, st in result["fields"].items():
            for k, v in st.items():
                field_stats[fid][k] += v
        done += 1
        print(f"[join_encode] {split} 桶 {result['bucket']:02d}: "
              f"{result['rows_in']:,} 行 → {result['rows_out']:,}（{done}/{len(bucket_list)}），"
              f"累计 {time.time() - t0:.0f}s", flush=True)

    for r in skipped:
        _merge(r)

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_join_encode_bucket, split, b, str(bucket_root),
                          str(out_root), str(vocab_path))
                for b in pending
            ]
            for fut in as_completed(futures):
                _merge(fut.result())
    else:
        for b in pending:
            _merge(_join_encode_bucket(split, b, str(bucket_root),
                                       str(out_root), str(vocab_path)))

    for stat in field_stats.values():
        stat["missing_rate"] = round(stat["missing_rows"] / total_out, 6) if total_out else 0.0
        stat["oov_rate"] = round(stat["oov_values"] / stat["input_values"], 6) if stat["input_values"] else 0.0
        stat["truncated_rate"] = round(stat["truncated_rows"] / total_out, 6) if total_out else 0.0
        stat["avg_kept"] = round(stat["kept_values"] / total_out, 4) if total_out else 0.0

    out_files = sorted(out_dir.glob("part-*.parquet"))
    report = {
        "generated_at": ts.isoformat(timespec="seconds"),
        "stage": "join_encode_full",
        "split": split,
        "workers": workers,
        "buckets": bucket_list,
        "resumed_buckets": len(skipped),
        "recomputed_buckets": len(pending),
        "rows_in": total_in,
        "rows_out": total_out,
        "miss_rows": total_miss,
        "unique_users": total_users,
        "labels": _clean_labels(total_out, all_clicks, all_convs),
        "fields": field_stats,
        "max_len": ENCODE_MAX_LEN,      # 编码参数（K），不属于词表
        "output_files": len(out_files),
        "output_bytes": sum(p.stat().st_size for p in out_files),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    rp = report_dir / f"join_encode_report_{split}_{ts:%Y%m%d-%H%M%S}.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[join_encode] {split}: {total_in:,} 行 → {total_out:,}，用户 {total_users:,}，"
          f"{report['output_bytes'] / 2**30:.2f} GiB，{report['elapsed_sec']}s，报告 {rp}")
    return report


def _count_common_vocab(
    group_exposure: dict[str, int] | None = None, src: Path | None = None
) -> tuple[dict[str, dict[str, int]], dict]:
    """扫 L1 train 统计 13 个公共域的取值频次（按曝光加权）。

    L1 每组一行，而一组公共特征会被该页**所有曝光**复用（train 平均 57.9 条/组）。
    `MIN_COUNT` 的语义是「在训练样本中出现次数」，因此必须乘上组内的曝光数；
    否则 101（用户 ID，每组只出现 1 次）会被判成低频 → 全量 OOV（12-E33b）。

    :param group_exposure: common_feature_index → 该组曝光数；None 表示按组计（不推荐）
    :param src: L1 目录，默认 `10_interim/common_index/train`，测试可传临时目录
    """
    src = src or INTERIM / "common_index" / "train"
    if not src.exists():
        raise SystemExit(f"[vocab] 缺少 L1 {src}，先运行 --stage index")

    counts: dict[str, dict[str, int]] = {fid: {} for fid in COMMON_FIELDS}
    rows_in = skipped = 0
    t0 = time.time()
    for part in sorted(src.glob("part-*.parquet")):
        for batch in pq.ParquetFile(part).iter_batches(
            batch_size=10_000, columns=["user_id", "field", "value"]
        ):
            d = batch.to_pydict()
            rows_in += len(d["user_id"])
            for gid, fields, values in zip(d["user_id"], d["field"], d["value"]):
                w = 1 if group_exposure is None else group_exposure.get(gid, 0)
                if w <= 0:  # 该组在骨架里没有曝光，对训练样本没有贡献
                    skipped += 1
                    continue
                for fid, value in zip(fields, values):
                    counter = counts.get(fid)
                    if counter is not None:  # 只统计公共域，意外域忽略
                        counter[value] = counter.get(value, 0) + w
    meta = {
        "source": str(src),
        "rows_in": rows_in,
        "rows_skipped_no_exposure": skipped,
        "weighted_by_exposure": group_exposure is not None,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    return counts, meta


def _count_skeleton_vocab(
    limit: int = 0,
) -> tuple[dict[str, dict[str, int]], dict, dict[str, int]]:
    """流式扫骨架 train 统计 10 个骨架域的取值频次，同时返回每组曝光数。

    曝光数用于给公共域词表加权：骨架一行 = 一条曝光，按 common_feature_index 聚合
    即得每组的曝光条数（12-E33b）。
    """
    src = RAW / "train" / "sample_skeleton_train.csv"
    counts: dict[str, dict[str, int]] = {fid: {} for fid in SKELETON_FIELDS}
    group_exposure: dict[str, int] = {}
    rows_in = malformed = 0
    t0 = time.time()
    with src.open("rb") as f:
        for raw in f:
            if limit and rows_in >= limit:
                break
            rows_in += 1
            parsed = _parse_skeleton_line(raw)
            if parsed is None:
                malformed += 1
                continue
            _, _, _, uid, _, skel = parsed
            group_exposure[uid] = group_exposure.get(uid, 0) + 1
            for fid, values in skel.items():
                counter = counts[fid]
                for value in values:
                    counter[value] = counter.get(value, 0) + 1
    meta = {
        "source": str(src),
        "rows_in": rows_in,
        "malformed_rows": malformed,
        "groups": len(group_exposure),
        "limit": limit,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    return counts, meta, group_exposure


def build_vocab(limit: int = 0) -> dict:
    """L2：只用 train 统计 23 个域的取值频次，写 20_processed/vocab.json。

    顺序：先扫骨架（拿到每组曝光数），再扫 L1（公共域按曝光加权计数）。
    词表 id 约定：0=缺失、1=OOV、>=2 真实取值，id 按频次降序分配（id 越小频次越高，
    encode 截断时取最小 K 个即高频前 K）。频次 < MIN_COUNT 归 OOV、不进词表。
    test 不参与统计，其新值在 encode 时统一映射到 1。

    输出格式（与 _load_vocab 的校验契约一致）：
        {"min_count": 5, "fields": [23 个 field_id],
         "field_dims": [23 个 int], "vocab": {"field_id": {"原值": id}}}
    `max_len`（K）属编码参数，写进 encode 报告，不写进词表。
    """
    ts = datetime.now()
    skel_counts, skel_meta, group_exposure = _count_skeleton_vocab(limit=limit)
    common_counts, common_meta = _count_common_vocab(group_exposure)

    raw_counts = {**common_counts, **skel_counts}
    if set(raw_counts) != set(SPARSE_FIELDS):
        raise RuntimeError(
            f"域集合不一致：{sorted(set(SPARSE_FIELDS) - set(raw_counts))} 缺失"
        )

    vocab: dict[str, dict[str, int]] = {}
    dims: list[int] = []
    field_report: dict[str, dict] = {}
    for fid in SPARSE_FIELDS:
        counter = raw_counts[fid]
        kept = [(v, c) for v, c in counter.items() if c >= MIN_COUNT]
        kept.sort(key=lambda kv: (-kv[1], kv[0]))  # 频次降序，同频按值字典序保证确定性
        field_map = {value: i + 2 for i, (value, _) in enumerate(kept)}
        vocab[fid] = field_map
        dims.append(len(field_map) + 2)  # +2 为 PAD/OOV
        total = sum(counter.values())
        below = sum(c for c in counter.values() if c < MIN_COUNT)
        field_report[fid] = {
            "distinct_values": len(counter),
            "total_occurrences": total,
            "kept_values": len(field_map),
            "field_dim": len(field_map) + 2,
            "oov_occurrences": below,
            "oov_rate": round(below / total, 6) if total else 0.0,
        }
        print(
            f"[vocab] {fid}: 唯一值 {len(counter):,}，保留 {len(field_map):,}，"
            f"OOV 率 {field_report[fid]['oov_rate'] * 100:.3f}%"
        )

    out_path = PROCESSED / "vocab.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": ts.isoformat(timespec="seconds"),
        "split": "train",
        "min_count": MIN_COUNT,
        "fields": SPARSE_FIELDS,
        "field_dims": dims,
        "vocab": vocab,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    report = {
        "generated_at": ts.isoformat(timespec="seconds"),
        "stage": "vocab",
        "split": "train",
        "min_count": MIN_COUNT,
        "limit": limit,
        "sources": {"common": common_meta, "skeleton": skel_meta},
        "fields": field_report,
        "output": str(out_path),
        "output_bytes": out_path.stat().st_size,
    }
    report_dir = REPORTS / "02_cleaning"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"vocab_report_{ts:%Y%m%d-%H%M%S}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[vocab] 23 域共 {sum(dims):,} 个 id，词表 {report['output_bytes'] / 2**20:.1f} MiB，"
        f"报告 {report_path}"
    )
    return report





def main() -> None:
    ap = argparse.ArgumentParser(description="Ali-CCP 预处理（分层，见 data/aliccp/README.md）")
    ap.add_argument("--stage", default="all",
                    choices=["verify", "index", "join", "bucket", "joinencode", "vocab", "encode", "clean", "all"])
    ap.add_argument("--split", default="both", choices=["train", "test", "both"])
    ap.add_argument("--limit", type=int, default=0,
                    help="index/join/vocab/encode 阶段只处理前 N 行（冒烟用，0=全量；join 全量待分桶）")
    ap.add_argument("--workers", type=int, default=1,
                    help="joinencode 进程数（>1 时按桶并行，每进程约 2 GB 内存）")
    args = ap.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]

    if args.stage in ("verify", "all"):
        verify_raw()
    if args.stage in ("index", "all"):
        INTERIM.mkdir(parents=True, exist_ok=True)
        build_common_index(splits, limit=args.limit)
    if args.stage in ("join", "all"):
        for split in splits:
            stream_join(split, limit=args.limit)
    if args.stage in ("bucket", "all"):
        for split in splits:
            bucket_skeleton(split)
    if args.stage in ("joinencode", "all"):
        for split in splits:
            join_encode_full(split, workers=args.workers)
    if args.stage in ("vocab", "all"):
        build_vocab(limit=args.limit)
    if args.stage in ("encode", "all"):
        encode(splits, limit=args.limit)
    if args.stage in ("clean", "all"):
        REPORTS.joinpath("02_cleaning").mkdir(parents=True, exist_ok=True)
        clean()


if __name__ == "__main__":
    main()
