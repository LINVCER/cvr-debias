"""clean 阶段的契约测试：C2 删除、C1 判重、train/val 按 user 整组切、两条恒等式。"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.prepare_aliccp import (
    ENCODE_MAX_LEN,
    ENCODE_SCHEMA,
    SPARSE_FIELDS,
    _hash_user,
    _val_cutoff,
    clean,
)


def _write_encoded(path: Path, rows: list[dict]) -> None:
    """按 ENCODE_SCHEMA 直接写编码产物，绕开 join/encode，专注测 clean。"""
    columns = {
        "click": pa.array([r["click"] for r in rows], type=pa.int8()),
        "conversion": pa.array([r["conversion"] for r in rows], type=pa.int8()),
        "user_id": pa.array([r["user_id"] for r in rows], type=pa.int64()),
        "row_no": pa.array([r["row_no"] for r in rows], type=pa.int64()),
    }
    for fid in SPARSE_FIELDS:
        columns[f"f_{fid}"] = pa.array(
            [r.get(fid, [0] * ENCODE_MAX_LEN) for r in rows],
            type=pa.list_(pa.int32(), ENCODE_MAX_LEN),
        )
    pq.write_table(pa.table(columns, schema=ENCODE_SCHEMA), path)


def _row(uid: int, click: int, conv: int, row_no: int, feat: int = 2) -> dict:
    r = {"user_id": uid, "click": click, "conversion": conv, "row_no": row_no}
    for fid in SPARSE_FIELDS:
        r[fid] = [feat] * ENCODE_MAX_LEN  # 每域都有值，避免触发 C5 之外的影响
    return r


def test_clean_c2_and_c1_and_identities(tmp_path: Path) -> None:
    in_dir = tmp_path / "encoded" / "test"
    in_dir.mkdir(parents=True)
    u1, u2 = _hash_user("u1"), _hash_user("u2")
    _write_encoded(
        in_dir / "part-00.parquet",
        [
            _row(u1, 1, 1, 10),          # 正常点击转化
            _row(u2, 0, 1, 11),          # C2：conversion=1 且 click=0，删
            _row(u1, 0, 0, 12),          # 正常
            _row(u1, 0, 0, 12),          # C1：与上一行同 (user,row_no)，判重删
        ],
    )

    report = clean(
        splits=["test"],
        encoded_root=tmp_path / "encoded",
        out_root=tmp_path / "out",
        report_dir=tmp_path / "reports",
    )

    s = report["splits"]["test"]
    # C1 删 1、C2 删 1，共删 2
    assert s["rules"]["C1_dup_rows"] == 1
    assert s["rules"]["C2_logic_rows"] == 1
    assert s["rules"]["C2_conv_removed"] == 1  # C2 那行 conversion=1
    assert s["rows_in"] == 4
    assert s["rows_out"] == 2
    # 恒等式：rows_in - removed == rows_out
    assert s["rows_in"] - s["removed"] == s["rows_out"]
    # 转化对账：before 2（u1conv + u2conv) - C2删1 == after 1
    assert s["labels_before"]["conversion"] == 2
    assert s["labels_after"]["conversion"] == 1

    table = pq.read_table(tmp_path / "out" / "test" / "part-00.parquet")
    assert table.num_rows == 2
    convs = table.column("conversion").to_pylist()
    clicks = table.column("click").to_pylist()
    # 不再存在 conversion=1 且 click=0 的行
    assert not any(c == 1 and k == 0 for c, k in zip(convs, clicks))


def test_clean_split_val_by_user_no_overlap(tmp_path: Path) -> None:
    in_dir = tmp_path / "encoded" / "train"
    in_dir.mkdir(parents=True)
    cut = _val_cutoff()
    # 构造明确落在 val（<cut）与 train（>=cut）两侧的 user
    val_uid = int(cut * 0.5)
    train_uid = int(cut + (1 << 62))
    rows = []
    for i in range(10):
        rows.append(_row(val_uid, 1, 1 if i < 3 else 0, 100 + i))
    for i in range(30):
        rows.append(_row(train_uid, 1, 0, 200 + i))
    _write_encoded(in_dir / "part-00.parquet", rows)

    report = clean(
        splits=["train"],
        encoded_root=tmp_path / "encoded",
        out_root=tmp_path / "out",
        report_dir=tmp_path / "reports",
    )

    s = report["splits"]["train"]
    assert s["outputs"]["val"]["rows"] == 10
    assert s["outputs"]["train"]["rows"] == 30

    tr = pq.read_table(tmp_path / "out" / "train" / "part-00.parquet").column("user_id").to_pylist()
    vl = pq.read_table(tmp_path / "out" / "val" / "part-00.parquet").column("user_id").to_pylist()
    assert all(u >= cut for u in tr)
    assert all(u < cut for u in vl)
    assert set(tr).isdisjoint(vl)  # 同一 user 不跨 train/val
    # val 带走了它应有的 3 条转化
    assert s["outputs"]["val"]["labels"]["conversion"] == 3
    assert s["outputs"]["train"]["labels"]["conversion"] == 0


def test_clean_c5_and_c8_report_only(tmp_path: Path) -> None:
    in_dir = tmp_path / "encoded" / "test"
    in_dir.mkdir(parents=True)
    u1 = _hash_user("u1")
    rows = []
    # 3 行：每个域全 0（缺失），22 域全缺 → missing=22 > 11，应计入 C5 但不删
    empty = {fid: [0] * ENCODE_MAX_LEN for fid in SPARSE_FIELDS}
    for i in range(3):
        r = {"user_id": u1, "click": 0, "conversion": 0, "row_no": 300 + i, **empty}
        rows.append(r)
    # 1 行正常（每域有值）
    rows.append(_row(u1, 1, 1, 400))
    _write_encoded(in_dir / "part-00.parquet", rows)

    report = clean(
        splits=["test"],
        encoded_root=tmp_path / "encoded",
        out_root=tmp_path / "out",
        report_dir=tmp_path / "reports",
    )

    s = report["splits"]["test"]
    # C5 只上报不删：3 行全缺失被计入，但行数不因此减少
    assert s["rules"]["C5_high_missing_rows"] == 3
    assert s["rows_out"] == 4  # 没有被 C5 删除
    # C8：单用户占比 100%，应出现在 top10 且 share=1.0
    top = s["rules"]["C8_top10_user_share"]
    assert top[0]["user_id"] == u1
    assert top[0]["share"] == 1.0
