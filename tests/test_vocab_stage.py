"""vocab 阶段的契约测试：id 语义、降序分配、min_count、域完整性。

build_vocab 内部用模块级常量 RAW/INTERIM/PROCESSED 定位文件，测试用
monkeypatch 把它们指到 tmp_path，避免依赖真实数据。
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import src.data.prepare_aliccp as prep
from src.data.prepare_aliccp import (
    COMMON_FIELDS,
    ENCODE_MAX_LEN,
    MIN_COUNT,
    SKELETON_FIELDS,
    SPARSE_FIELDS,
    _count_common_vocab,
    _load_vocab,
    build_vocab,
)


def _write_common_index(root: Path) -> None:
    """合成 L1 train：频次可推，id 顺序确定。

    公共域按曝光加权（骨架里 u1 有 2 条曝光、u2 有 1 条），所以 L1 里的每个计数 ×2：
      109_14: a×7, b×6, c×1  -> 加权 a=14, b=12, c=2  -> a(2), b(3), c 归 OOV
      121:    x×5, y×2       -> 加权 x=10, y=4       -> x 保留，y 归 OOV
    """
    (root / "train").mkdir(parents=True)
    fields = [
        ["109_14"] * 14 + ["121"] * 7,
    ]
    values = [
        ["a"] * 7 + ["b"] * 6 + ["c"] + ["x"] * 5 + ["y"] * 2,
    ]
    table = pa.table({
        "user_id": pa.array(["u1"], type=pa.string()),
        "field": pa.array(fields, type=pa.list_(pa.string())),
        "value": pa.array(values, type=pa.list_(pa.string())),
    })
    pq.write_table(table, root / "train" / "part-00.parquet")


def _write_skeleton_csv(root: Path) -> None:
    """合成骨架 train：205/206 频次可推，id 顺序确定。

    205: p1×6, p2×1 -> p1(2), p2 归 OOV
    206: q1×6, q2×4 -> q1(2), q2 归 OOV
    """
    (root / "train").mkdir(parents=True)
    sep_f, sep_v, sep_w = "\x01", "\x02", "\x03"
    def feat(fid: str, *vals: str) -> str:
        return sep_f.join(f"{fid}{sep_v}{v}{sep_w}1.0" for v in vals)
    lines = [
        f"0,0,0,u1,0,{feat('205', *(['p1'] * 5))}{sep_f}{feat('206', 'q1','q1','q1','q1','q2')}",
        f"1,1,1,u2,0,{feat('205', 'p1')}{sep_f}{feat('206', 'q1','q1','q2','q2','q2')}",
        f"2,0,0,u1,0,{feat('205', 'p2')}",
    ]
    (root / "train" / "sample_skeleton_train.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_build_vocab_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(prep, "RAW", tmp_path / "raw")
    monkeypatch.setattr(prep, "INTERIM", tmp_path / "interim")
    monkeypatch.setattr(prep, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(prep, "REPORTS", tmp_path / "reports")
    _write_common_index(tmp_path / "interim" / "common_index")
    _write_skeleton_csv(tmp_path / "raw")

    report = build_vocab()

    vocab_path = tmp_path / "processed" / "vocab.json"
    assert vocab_path.exists()
    data = json.loads(vocab_path.read_text(encoding="utf-8"))

    # 结构契约：23 域齐全、field_dims 与 SPARSE_FIELDS 顺序一致、max_len 正确
    assert len(data["field_dims"]) == len(SPARSE_FIELDS) == 23
    assert set(data["vocab"]) == set(SPARSE_FIELDS)
    assert "max_len" not in data            # K 是编码参数，不写进词表
    assert "101" in data["vocab"]  # 101 = 用户 ID，必须回到域清单（12-E33b）
    assert data["fields"] == SPARSE_FIELDS  # 列顺序唯一权威来源

    # 频次与 id 契约：id 从 2 开始，按频次降序；公共域按曝光加权（u1 有 2 条曝光）
    v109 = data["vocab"]["109_14"]
    assert v109 == {"a": 2, "b": 3}                # 加权 a=14 > b=12，c=2 归 OOV
    v121 = data["vocab"]["121"]
    assert v121 == {"x": 2}                        # 加权 x=10 保留，y=4 < MIN_COUNT 归 OOV
    assert data["vocab"]["205"] == {"p1": 2}       # p1×6 保留，p2×1 归 OOV
    assert data["vocab"]["206"] == {"q1": 2}       # q1×5 保留，q2×4 归 OOV

    # field_dims = kept + 2，位置与 SPARSE_FIELDS 对齐
    idx109 = SPARSE_FIELDS.index("109_14")
    assert data["field_dims"][idx109] == 2 + 2

    # 报告：OOV 率与对账（以加权后的计数为准）
    f109 = report["fields"]["109_14"]
    assert f109["total_occurrences"] == 28        # (a×7 + b×6 + c×1) × 2 条曝光
    assert f109["kept_values"] == 2
    assert f109["oov_occurrences"] == 2
    f121 = report["fields"]["121"]
    assert f121["total_occurrences"] == 14        # (x×5 + y×2) × 2
    assert f121["oov_occurrences"] == 4
    assert report["sources"]["common"]["rows_in"] == 1
    assert report["sources"]["common"]["weighted_by_exposure"] is True
    assert report["sources"]["skeleton"]["rows_in"] == 3
    assert report["sources"]["skeleton"]["groups"] == 2
    assert report["sources"]["skeleton"]["malformed_rows"] == 0

    # 产物能被下游 encode 的 _load_vocab 直接消费
    loaded = _load_vocab(vocab_path)
    assert loaded["vocab"]["109_14"]["a"] == 2


def test_load_vocab_rejects_bad_dims(tmp_path: Path) -> None:
    bad = tmp_path / "vocab.json"
    bad.write_text(
        json.dumps({
            "max_len": ENCODE_MAX_LEN,
            "field_dims": [2] * (len(SPARSE_FIELDS) - 1),  # 少一维
            "vocab": {fid: {} for fid in SPARSE_FIELDS},
        }),
        encoding="utf-8",
    )
    try:
        _load_vocab(bad)
    except ValueError as exc:
        assert "field_dims" in str(exc)
    else:
        raise AssertionError("应拒绝 field_dims 长度不符的词表")


def test_common_vocab_weights_by_group_exposure(tmp_path: Path) -> None:
    """101 类用户 ID 的回归测试（12-E33b）。

    按组计数时每值只出现 1 次，会被 MIN_COUNT 剔光；按曝光加权后 g1 的取值保留。
    """
    l1 = tmp_path / "l1"
    l1.mkdir()
    pq.write_table(
        pa.table({
            "user_id": pa.array(["g1", "g2", "g3"], type=pa.string()),
            "field": pa.array([["101"]] * 3, type=pa.list_(pa.string())),
            "value": pa.array([["u1"], ["u2"], ["u3"]], type=pa.list_(pa.string())),
        }),
        l1 / "part-00.parquet",
    )

    counts, meta = _count_common_vocab({"g1": 60, "g2": 1}, src=l1)
    assert counts["101"] == {"u1": 60, "u2": 1}
    assert meta["weighted_by_exposure"] is True
    assert meta["rows_skipped_no_exposure"] == 1   # g3 在骨架里没有曝光

    # 旧口径（不传权重）会把三个用户全部判成低频
    old, old_meta = _count_common_vocab(src=l1)
    assert old["101"] == {"u1": 1, "u2": 1, "u3": 1}
    assert all(c < MIN_COUNT for c in old["101"].values())
    assert old_meta["weighted_by_exposure"] is False


def test_min_count_boundary() -> None:
    """MIN_COUNT 边界：恰好等于阈值的保留，少一次的归 OOV。"""
    assert MIN_COUNT >= 2  # 阈值必须大于 1，否则 OOV 桶永不触发
