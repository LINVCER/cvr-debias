"""encode 阶段的契约测试：K=8 截断、0/1 分离、去重、user_id 哈希。"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.prepare_aliccp import (
    ENCODE_MAX_LEN,
    OOV_ID,
    PAD_ID,
    SPARSE_FIELDS,
    _hash_user,
    encode,
)


def _write_join(path: Path, rows: list[dict]) -> None:
    columns = {
        "user_id": pa.array([r["user_id"] for r in rows], type=pa.string()),
        "click": pa.array([r["click"] for r in rows], type=pa.int8()),
        "conversion": pa.array([r["conversion"] for r in rows], type=pa.int8()),
        "row_no": pa.array([r["row_no"] for r in rows], type=pa.int64()),
    }
    for fid in SPARSE_FIELDS:
        columns[f"f_{fid}"] = pa.array(
            [r.get(fid, []) for r in rows], type=pa.list_(pa.string())
        )
    pq.write_table(pa.table(columns), path)


def _write_vocab(path: Path) -> None:
    vocab: dict[str, dict[str, int]] = {}
    dims: list[int] = []
    for fid in SPARSE_FIELDS:
        if fid == "109_14":
            vocab[fid] = {"a": 2, "b": 3}
            dims.append(4)
        elif fid == "121":
            vocab[fid] = {f"v{i}": i + 2 for i in range(40)}
            dims.append(42)
        else:
            vocab[fid] = {}
            dims.append(2)
    path.write_text(
        json.dumps(
            {"max_len": ENCODE_MAX_LEN, "field_dims": dims, "vocab": vocab},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_encode_contract(tmp_path: Path) -> None:
    join_root = tmp_path / "join"
    (join_root / "train").mkdir(parents=True)
    _write_join(
        join_root / "train" / "part-00.parquet",
        [
            {"user_id": "u1", "click": 1, "conversion": 1, "row_no": 10,
             "109_14": ["a", "b", "c"], "121": ["v0", "v1"]},
            {"user_id": "u2", "click": 0, "conversion": 0, "row_no": 11,
             "109_14": ["b"] * 10, "121": [f"v{i}" for i in range(40)]},
            {"user_id": "u1", "click": 0, "conversion": 0, "row_no": 12},
        ],
    )
    vocab_path = tmp_path / "vocab.json"
    _write_vocab(vocab_path)

    report = encode(
        splits=["train"],
        join_root=join_root,
        out_root=tmp_path / "encoded",
        vocab_path=vocab_path,
        report_dir=tmp_path / "reports",
    )

    table = pq.read_table(tmp_path / "encoded" / "train" / "part-00.parquet")
    assert table.num_rows == 3
    assert table.schema.field("f_109_14").type == pa.list_(pa.int32(), ENCODE_MAX_LEN)

    f109 = table.column("f_109_14").to_pylist()
    assert f109[0] == [2, 3, OOV_ID] + [PAD_ID] * (ENCODE_MAX_LEN - 3)
    assert f109[1] == [3] + [PAD_ID] * (ENCODE_MAX_LEN - 1)   # 重复值只留一个
    assert f109[2] == [PAD_ID] * ENCODE_MAX_LEN

    f121 = table.column("f_121").to_pylist()
    assert f121[1] == list(range(2, 2 + ENCODE_MAX_LEN))       # 超过 K 时截断
    assert f121[2] == [PAD_ID] * ENCODE_MAX_LEN

    users = table.column("user_id").to_pylist()
    assert users[0] == users[2] == _hash_user("u1")
    assert users[1] == _hash_user("u2")

    stats = report["splits"]["train"]
    assert stats["rows_in"] == stats["rows_out"] == 3
    assert stats["unique_users"] == 2
    assert stats["labels"]["click"] == 1
    assert stats["labels"]["conversion"] == 1
    assert stats["fields"]["109_14"]["oov_values"] == 1
    assert stats["fields"]["109_14"]["missing_rows"] == 1
    assert stats["fields"]["121"]["truncated_rows"] == 1


def test_hash_user_is_stable_and_in_int64() -> None:
    value = _hash_user("u1")
    assert value == _hash_user("u1")
    assert 0 <= value < 2**63
