"""推理服务：把训练产物（state_dict + metrics.json 里的 config/校准参数）挂成一个 HTTP 服务。

只做一件事：给定 23 个稀疏域的 id，返回 pCTR / pCVR / pCTCVR。
契约由 tests 之外的 smoke 脚本验证，不在服务内部做任何训练或数据读取。

启动：
    python -m uvicorn deploy.serve:app --host 127.0.0.1 --port 8000
    # 或指定产物目录
    RUN_DIR=artifacts/runs/main/uc_escm/seed0 python -m uvicorn deploy.serve:app --port 8000
"""

import json
import os
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.models  # noqa: F401  触发模型注册
from src.models.base import build_model

RUN_DIR = os.environ.get("RUN_DIR", "artifacts/runs/smoke/uc_escm/seed0")
MAX_BATCH = 2048


def load_bundle(run_dir: Path) -> dict:
    meta_path = run_dir / "metrics.json"
    if not meta_path.exists():
        raise SystemExit(f"找不到 {meta_path}；请先训练，或用 RUN_DIR 指定产物目录")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg, metrics = meta["config"], meta["metrics"]
    field_dims = [int(d) for d in cfg["data"]["field_dims"]]

    model = build_model(cfg["model_name"], cfg["model"], field_dims)
    ckpt = Path(cfg["train"]["ckpt_dir"]) / (
        f"{cfg['model_name']}_{cfg['loss_name']}_seed{int(cfg['train']['seed'])}.pt"
    )
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return {
        "model": model,
        "model_name": cfg["model_name"],
        "field_dims": field_dims,
        "calib": metrics.get("calibration", {"temperature": 1.0, "tau": 0.0, "alpha": 0.0}),
        "ckpt": str(ckpt),
        "run_dir": str(run_dir),
    }


BUNDLE = load_bundle(Path(RUN_DIR))
app = FastAPI(title="cvr-debias serving", version="0.1")


class PredictIn(BaseModel):
    ids: list[list[int]]


def _check(ids: list[list[int]]) -> torch.Tensor:
    dims = BUNDLE["field_dims"]
    if not ids:
        raise HTTPException(status_code=422, detail="ids 不能为空")
    if len(ids) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"单批最多 {MAX_BATCH} 行")
    for row in ids:
        if len(row) != len(dims):
            raise HTTPException(status_code=422, detail=f"每行必须是 {len(dims)} 个特征 id")
        for j, v in enumerate(row):
            if not (0 <= int(v) < dims[j]):
                raise HTTPException(status_code=422, detail=f"第 {j} 列 id={v} 越界（上限 {dims[j] - 1}）")
    return torch.tensor(ids, dtype=torch.long)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model": BUNDLE["model_name"]}


@app.get("/meta")
def meta() -> dict:
    return {
        "model": BUNDLE["model_name"],
        "field_dims": BUNDLE["field_dims"],
        "n_fields": len(BUNDLE["field_dims"]),
        "calibration": BUNDLE["calib"],
        "ckpt": BUNDLE["ckpt"],
        "run_dir": BUNDLE["run_dir"],
    }


@app.post("/predict")
def predict(req: PredictIn) -> dict:
    ids = _check(req.ids)
    with torch.no_grad():
        out = BUNDLE["model"](ids)
    t = float(BUNDLE["calib"].get("temperature", 1.0))
    p_ctr = None if out.get("ctr_logit") is None else torch.sigmoid(out["ctr_logit"] / t).tolist()
    p_cvr = torch.sigmoid(out["cvr_logit"]).tolist()
    p_ctcvr = None if p_ctr is None else [a * b for a, b in zip(p_ctr, p_cvr)]
    return {"p_ctr": p_ctr, "p_cvr": p_cvr, "p_ctcvr": p_ctcvr}
