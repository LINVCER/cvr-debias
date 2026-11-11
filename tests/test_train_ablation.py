"""消融开关：`train.drop_fields` 只把指定域的 id 置 0，不改 field_dims 与模型结构。"""

import json
from pathlib import Path

import torch

from src.train import _zero_columns, resolve_drop_cols


def _cfg(tmp_path: Path, drop) -> dict:
    vocab = tmp_path / "vocab.json"
    vocab.write_text(
        json.dumps(
            {"fields": ["101", "109_14", "205"], "field_dims": [2, 2, 2], "vocab": {}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"data": {"vocab": str(vocab)}, "train": {"drop_fields": drop}}


def test_drop_fields_resolves_by_vocab_order(tmp_path: Path) -> None:
    assert resolve_drop_cols(_cfg(tmp_path, [101])) == [0]       # CLI 里 [101] 解析成 int
    assert resolve_drop_cols(_cfg(tmp_path, ["109_14"])) == [1]
    assert resolve_drop_cols(_cfg(tmp_path, [])) == []


def test_zero_columns_only_touches_targets(tmp_path: Path) -> None:
    ids = torch.ones((2, 3, 4), dtype=torch.long)
    _zero_columns({"ids": ids}, [0, 2])
    assert int(ids[:, 0].sum()) == 0
    assert int(ids[:, 2].sum()) == 0
    assert int(ids[:, 1].sum()) == 2 * 4          # 其余域保持原值


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    try:
        resolve_drop_cols(_cfg(tmp_path, ["101", "999"]))
    except ValueError as exc:
        assert "999" in str(exc)
    else:
        raise AssertionError("未知域应报错，不能静默忽略")
