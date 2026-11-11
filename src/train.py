"""训练入口：一条实验 = 一行命令，所有差异都在配置里。

用法：
    python -m src.train exp=smoke data=synthetic model=uc_escm loss=vrdr train.seed=0
    python -m src.train data=aliccp model=uc_escm loss=vrdr exp=main \\
        model.n_propensity=2 model.n_conversion=2 model.n_shared=4 \\
        loss.alpha=0.005 train.seed=0

注意三个实现坑（对应总案 §2.6）：
1. 倾向得分 ẽ 必须 detach，否则模型会靠推高 pCTR 来转化器损失；
2. 温度 T 是后验拟合，不进主训练图；
3. log_var 参数化并初始化为 0，否则前几百步数值溢出。
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import src.models  # noqa: F401  触发模型注册
from src.calibration.temperature import fit_temperature, select_alpha, select_tau
from src.data import dataset, synthetic
from src.evaluation.metrics import bias_and_mse, binary_metrics
from src.losses.causal import (
    SelfNormalizer,
    bce_logits,
    ctcvr_loss,
    dr_loss,
    imp_mse_loss,
    ips_loss,
    naive_cvr_loss,
    propensity_score,
    vr_dr_loss,
)
from src.losses.regularizer import orthogonal_penalty
from src.losses.weighting import build_weighting
from src.models.base import build_model

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- 配置


def deep_set(cfg: dict, dotted: str, value):
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def load_group(cfg: dict, group: str, name: str) -> None:
    """按组名加载 configs/<group>/<name>.yaml；exp 组额外把 overrides 展开成覆盖项。"""
    sub_path = ROOT / "configs" / group / f"{name}.yaml"
    if not sub_path.exists():
        raise FileNotFoundError(f"配置不存在：{sub_path}")
    loaded = yaml.safe_load(sub_path.read_text(encoding="utf-8"))
    cfg[f"{group}_name"] = name
    if group == "exp":
        cfg["exp"] = loaded
        for k, v in (loaded.get("overrides") or {}).items():
            deep_set(cfg, k, v)
    else:
        cfg[group] = loaded


def load_config(path, overrides) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for group in ("data", "model", "loss", "exp"):
        name = cfg.get(group)
        if isinstance(name, str):
            load_group(cfg, group, name)

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"覆盖项必须形如 key=value：{item}")
        k, v = item.split("=", 1)
        value = yaml.safe_load(v)
        if k in ("data", "model", "loss") and isinstance(value, str):
            load_group(cfg, k, value)
        elif k == "exp" and isinstance(value, str):
            load_group(cfg, "exp", value)
        else:
            deep_set(cfg, k, value)
    return cfg


# ----------------------------------------------------------------------------- 数据


def load_data(cfg: dict) -> dict:
    name = cfg.get("data_name", "synthetic")
    if name == "synthetic":
        return synthetic.generate(cfg["data"])
    if name in ("aliccp", "aliccp_10p"):
        # 真实数据只返回分片目录，训练循环惰性产批；全量拼接会超出内存
        return dataset.aliccp_paths(cfg["data"])
    raise ValueError(f"未知数据集 {name}（semisynth / aliexpress 走 build_semisynth 后单独接入）")


def resolve_field_dims(cfg: dict) -> list:
    """field_dims 优先取配置；真实数据从 vocab.json 读取，避免两处手抄不一致。"""
    dims = cfg["data"].get("field_dims")
    if dims:
        return [int(d) for d in dims]
    vocab_path = cfg["data"].get("vocab")
    if vocab_path:
        meta = json.loads((ROOT / vocab_path).read_text(encoding="utf-8"))
        dims = meta.get("field_dims")
        if dims:
            return [int(d) for d in dims]
    raise KeyError("data.field_dims 缺失，且 vocab 中没有 field_dims")


def iter_batches(split: dict, batch_size: int, shuffle: bool = False, generator=None):
    n = len(split["click"])
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        sel = idx[start : start + batch_size]
        yield {k: v[sel] for k, v in split.items()}


def iter_split(split, batch_size: int, shuffle: bool = False, generator=None,
               num_workers: int = 0, epoch: int = 0):
    """split 既可以是内存张量字典（合成数据），也可以是分片目录（真实数据，惰性）。

    num_workers > 0 时用 DataLoader 多进程并行读取 parquet，配合 pin_memory 让
    GPU 不再等数据。epoch 用于每轮打乱文件顺序的种子偏移。
    """
    if isinstance(split, dict):
        yield from iter_batches(split, batch_size, shuffle=shuffle, generator=generator)
        return
    shard_dir = Path(split)
    files = sorted(shard_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{shard_dir} 下没有 parquet 分片")
    if shuffle and len(files) > 1:
        # 惰性模式无法全局打乱，退而求其次：每个 epoch 打乱分片顺序
        order = torch.randperm(len(files), generator=generator).tolist()
        files = [files[i] for i in order]

    if num_workers and num_workers > 0:
        from src.data.dataset import ParquetIterableDataset
        seed = int(generator.initial_seed() % (2**31)) + epoch if generator else epoch
        ds = ParquetIterableDataset(
            shard_dir, batch_size=batch_size, files=files,
            shuffle=False, seed=seed,  # 文件顺序已在上面用 generator 打乱
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=None, num_workers=num_workers, pin_memory=True,
        )
        yield from loader
    else:
        yield from dataset.iter_batches(shard_dir, batch_size=batch_size, files=files)


# ----------------------------------------------------------------------------- 损失


def task_names_for(loss_name: str) -> list:
    if loss_name == "bce":
        return ["cvr"]
    if loss_name == "esmm":
        return ["ctr", "ctcvr"]
    if loss_name == "ips":
        return ["ctr", "ctcvr", "cvr"]
    if loss_name in ("dr", "vrdr"):
        return ["ctr", "ctcvr", "cvr", "imp"]
    raise ValueError(f"未知损失 {loss_name}")


def compute_loss(out, batch, loss_cfg, calib, weighting, normalizer=None):
    click, conv = batch["click"], batch["conv"]
    name = loss_cfg["name"]
    losses = {}

    e = None
    if out.get("ctr_logit") is not None:
        e = propensity_score(out["ctr_logit"], calib["temperature"], calib["tau"])

    if name == "bce":
        losses["cvr"] = naive_cvr_loss(out["cvr_logit"], click, conv)
    else:
        if out.get("ctr_logit") is None:
            raise ValueError(f"{name} 需要 CTR 塔输出，当前模型不提供")
        losses["ctr"] = bce_logits(out["ctr_logit"], click)
        losses["ctcvr"] = ctcvr_loss(out["ctr_logit"], out["cvr_logit"], click, conv)
        if name == "esmm":
            pass                                  # pCVR 只作为中间量，不直接受监督
        elif name == "ips":
            losses["cvr"] = ips_loss(out["cvr_logit"], click, conv, e)
        elif name == "dr":
            if out.get("imp") is None:
                raise ValueError("dr 需要插补塔输出（imp），当前模型不提供；用 uc_escm")
            losses["cvr"] = dr_loss(
                out["cvr_logit"], out["imp"], click, conv, e,
                detach_imp=bool(loss_cfg.get("detach_imp", True)),
                self_normalized=bool(loss_cfg.get("self_normalized", False)),
                normalizer=normalizer,
            )
            losses["imp"] = imp_mse_loss(out["imp"], out["cvr_logit"], conv, click)
        elif name == "vrdr":
            if out.get("imp") is None:
                raise ValueError("vrdr 需要插补塔输出（imp），当前模型不提供；用 uc_escm")
            losses["cvr"], losses["imp"] = vr_dr_loss(
                out["cvr_logit"],
                out["imp"],
                click,
                conv,
                e,
                gamma=float(loss_cfg.get("gamma", 1.0)),
                detach_imp=bool(loss_cfg.get("detach_imp", True)),
                self_normalized=bool(loss_cfg.get("self_normalized", False)),
                normalizer=normalizer,
            )
        else:
            raise ValueError(f"未知损失 {name}")

    total, detail = weighting(losses)
    orth_lambda = float(loss_cfg.get("orth_lambda", 0.0))
    if orth_lambda > 0 and "u_p" in out.get("extras", {}):
        reg = orthogonal_penalty(out["extras"]["u_p"], out["extras"]["u_c"])
        total = total + orth_lambda * reg
        detail["orth"] = reg.detach()
    return total, detail


# ----------------------------------------------------------------------------- 推理与评估


@torch.no_grad()
def predict(model, split, batch_size: int, device, max_rows: int = 0, num_workers: int = 0,
            drop_cols=()) -> dict:
    """max_rows>0 时只评估前若干行，用于真实数据的小步冒烟。"""
    model.eval()
    chunks = {"ctr": [], "cvr": [], "imp": []}
    labels = {"click": [], "conv": [], "user_ids": []}
    seen = 0
    for batch in iter_split(split, batch_size, shuffle=False, num_workers=num_workers):
        _zero_columns(batch, drop_cols)
        out = model(batch["ids"].to(device, non_blocking=True))
        for src, dst in (("ctr_logit", "ctr"), ("cvr_logit", "cvr"), ("imp", "imp")):
            if out.get(src) is not None:
                chunks[dst].append(out[src].detach().cpu())
        labels["click"].append(batch["click"].cpu())
        labels["conv"].append(batch["conv"].cpu())
        labels["user_ids"].append(batch["user_ids"].cpu())
        seen += batch["click"].numel()
        if max_rows and seen >= max_rows:
            break
    res = {k: torch.cat(v) for k, v in labels.items()}
    for k, v in chunks.items():
        res[k] = torch.cat(v) if v else None
    if max_rows and len(res["click"]) > max_rows:
        for k in ("click", "conv", "user_ids", "ctr", "cvr", "imp"):
            if res[k] is not None:
                res[k] = res[k][:max_rows]
    if isinstance(split, dict) and "p_star" in split:
        res["p_star"] = split["p_star"][: len(res["click"])]
    return res


def _check_numeric_health(pred: dict) -> dict:
    """数值健康巡检：CVR 头坍缩与 imp 越界都不会抛异常，只能主动检查。

    回归背景：dr/vrdr 曾把 imp 当无界线性项，两万余步后漂到 −1e12、
    CVR 塔坍缩为常数（AUC 恒为 0.5），但训练全程无报错（12 号 E30）。
    这里只告警不中断，避免长跑被一个阈值误杀。
    """
    from src.models.uc_escm import IMP_MAX

    out = {}
    cvr = pred.get("cvr")
    if cvr is not None:
        std = float(cvr.std())
        out["cvr_logit_std"] = std
        if std < 1e-4:
            print(f"[warn] CVR 头输出接近常数（std={std:.2e}），可能已坍缩", flush=True)
    imp = pred.get("imp")
    if imp is not None:
        lo, hi = float(imp.min()), float(imp.max())
        out["imp_min"], out["imp_max"] = lo, hi
        if lo < 0.0 or hi > IMP_MAX + 1e-3:
            print(f"[warn] imp 越界 [{lo:.4g}, {hi:.4g}]，应在 [0, {IMP_MAX}]", flush=True)
    return out


def evaluate(pred: dict, calib: dict, n_bins: int = 10) -> dict:
    click = pred["click"]
    conv = pred["conv"]
    users = pred["user_ids"]
    temperature = float(calib.get("temperature", 1.0))

    p_cvr = torch.sigmoid(pred["cvr"]).numpy() if pred["cvr"] is not None else None
    if pred["ctr"] is not None:
        p_ctr = torch.sigmoid(pred["ctr"] / temperature).numpy()
        p_ctcvr = p_ctr * p_cvr
        out = {"ctr_exposure": binary_metrics(click.numpy(), p_ctr, users.numpy(), n_bins)}
    else:
        p_ctcvr = p_cvr
        out = {}

    out["ctcvr_exposure"] = binary_metrics((click * conv).numpy(), p_ctcvr)
    clicked = (click > 0.5).numpy()
    if p_cvr is not None and clicked.any():
        out["cvr_click_space"] = binary_metrics(
            conv.numpy()[clicked], p_cvr[clicked], users.numpy()[clicked], n_bins
        )
    if p_cvr is not None and "p_star" in pred:
        # 半合成口径：全空间 pCVR 与伪真值的偏差。真实数据上拿不到这个数
        out["bias_vs_pstar"] = bias_and_mse(p_cvr, pred["p_star"].numpy())
    out["calibration"] = dict(calib)
    return out


# ----------------------------------------------------------------------------- 主流程


def resolve_field_names(cfg: dict) -> list:
    """vocab.json 里的域顺序（与 ids 张量的第 1 维严格对应），用于 drop_fields 定位列。"""
    vocab_path = cfg["data"].get("vocab")
    if not vocab_path:
        return []
    meta = json.loads((ROOT / vocab_path).read_text(encoding="utf-8"))
    return list(meta.get("fields", []))


def resolve_drop_cols(cfg: dict) -> list:
    """把 `train.drop_fields`（field_id 列表）翻译成 ids 张量的列下标。

    消融用：把某域的 id 全置 0（PAD），等价于「该域整列缺失」，
    不改 field_dims 与 embedding 表结构，因此不会影响其它域的参数形状。
    """
    names = resolve_field_names(cfg)
    drop = [str(x) for x in (cfg["train"].get("drop_fields") or [])]
    if not drop:
        return []
    if not names:
        raise ValueError("train.drop_fields 需要 vocab.json 提供 fields，但当前数据配置没有 vocab")
    unknown = [f for f in drop if f not in names]
    if unknown:
        raise ValueError(f"drop_fields 含未知域 {unknown}，可用域：{names}")
    return [names.index(f) for f in drop]


def _zero_columns(batch: dict, cols) -> None:
    if not cols:
        return
    batch["ids"][:, list(cols), :] = 0


def run(cfg: dict) -> dict:
    train_cfg = cfg["train"]
    seed = int(train_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(train_cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[warn] 未检测到 CUDA，回退到 CPU", flush=True)
        device = torch.device("cpu")
    # 实际生效的设备必须可见，避免"以为在 GPU 上、实际在 CPU"的静默偏差
    if device.type == "cuda":
        print(f"[device] cuda:{torch.cuda.current_device()} "
              f"{torch.cuda.get_device_name(device)}", flush=True)
    else:
        print(f"[device] {device}", flush=True)

    data = load_data(cfg)
    field_dims = resolve_field_dims(cfg)
    drop_cols = resolve_drop_cols(cfg)
    if drop_cols:
        print(f"[ablation] drop_fields={cfg['train']['drop_fields']} → 列下标 {drop_cols}",
              flush=True)
    model = build_model(cfg["model_name"], cfg["model"], field_dims).to(device)

    loss_cfg = cfg["loss"]
    # SNIPS 的归一化因子必须在整个训练过程中共享，逐 batch 重估会让 Z 抖到跨数量级
    normalizer = SelfNormalizer() if loss_cfg.get("self_normalized") else None
    weighting = build_weighting(
        loss_cfg.get("weighting", "uncertainty"), task_names_for(loss_cfg["name"])
    ).to(device)

    calib = {"temperature": 1.0, "tau": 0.0, "alpha": float(loss_cfg.get("alpha", 0.0))}
    cal_cfg = cfg.get("calibration", {})
    alphas = tuple(cal_cfg.get("alphas", [0.001, 0.005, 0.01, 0.05]))
    var_weight = float(cal_cfg.get("var_weight", 0.5))

    params = list(model.parameters()) + list(weighting.parameters())
    opt = torch.optim.Adam(params, lr=float(train_cfg["lr"]))
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))
    generator = torch.Generator().manual_seed(seed)
    # 真实数据上全量一个 epoch 需数小时；max_steps > 0 时只跑固定步数做冒烟
    max_steps = int(train_cfg.get("max_steps", 0))
    eval_max_rows = int(train_cfg.get("eval_max_rows", 0))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 5.0))

    # 运行目录提前建好：每轮落 state.json，便于外部随时查进度
    exp = cfg.get("exp") or {}
    # 目录名带 loss：同一模型有多种损失（如 uc_escm 的 ips/dr/vrdr）时不能共用 seed 目录
    run_dir = (
        Path(train_cfg["run_dir"]) / str(exp.get("name", "run"))
        / f"{cfg['model_name']}_{cfg['loss_name']}" / f"seed{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    def _write_state(payload: dict) -> None:
        (run_dir / "state.json").write_text(
            json.dumps(payload, ensure_ascii=False, default=float), encoding="utf-8"
        )

    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        agg, steps, seen = {}, 0, 0
        started = time.time()
        for batch in iter_split(data["train"], batch_size, shuffle=True, generator=generator,
                                num_workers=num_workers, epoch=epoch):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            _zero_columns(batch, drop_cols)
            out = model(batch["ids"])
            total, detail = compute_loss(out, batch, loss_cfg, calib, weighting, normalizer)
            opt.zero_grad()
            total.backward()
            # 兜底：imp/1e 加权项曾有数值漂移，裁剪防止单步爆炸
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            opt.step()
            for k, v in detail.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            steps += 1
            seen += batch["click"].numel()
            if max_steps and steps >= max_steps:
                break
        elapsed = max(time.time() - started, 1e-9)
        msg = " | ".join(f"{k} {v / max(steps, 1):.4f}" for k, v in agg.items())

        # 校准参数每轮在验证集上后验拟合，不进训练图
        cal_sec = 0.0
        if cal_cfg.get("enabled", True):
            t_cal = time.time()
            val_pred = predict(model, data["val"], batch_size, device, max_rows=eval_max_rows, num_workers=num_workers, drop_cols=drop_cols)
            if val_pred["ctr"] is not None:
                temperature = fit_temperature(val_pred["ctr"], val_pred["click"])
                chosen = select_alpha(val_pred["ctr"], val_pred["click"], alphas, temperature, var_weight)
                calib = {
                    "temperature": temperature,
                    "tau": chosen["tau"],
                    "alpha": chosen["alpha"],
                }
            cal_sec = time.time() - t_cal
            print(f"[epoch {epoch + 1}] {msg} | steps {steps} rows {seen} "
                  f"{seen / elapsed:.0f} 行/秒 | 训练 {elapsed:.1f}s 校准 {cal_sec:.1f}s | "
                  f"T={calib['temperature']:.3f} tau={calib['tau']:.4f}", flush=True)
            # 数值健康巡检：CVR 头坍缩与 imp 越界都不会报错，只能主动盯
            health = _check_numeric_health(val_pred)
            if normalizer is not None:
                health["snips_z"] = normalizer.value
            _write_state({
                "epoch": epoch + 1,
                "epochs": int(train_cfg["epochs"]),
                "steps": steps,
                "rows": seen,
                "rows_per_sec": seen / elapsed,
                "train_sec": elapsed,
                "calib_sec": cal_sec,
                "device": str(device),
                "calib": calib,
                "health": health,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        if max_steps and steps >= max_steps:
            print(f"[smoke] 达到 max_steps={max_steps}，提前结束训练", flush=True)
            break

    t_eval = time.time()
    test_pred = predict(model, data["test"], batch_size, device, max_rows=eval_max_rows, num_workers=num_workers, drop_cols=drop_cols)
    t_metrics = time.time()
    metrics = evaluate(test_pred, calib)
    print(f"[eval] test 推理 {t_metrics - t_eval:.1f}s，指标计算 {time.time() - t_metrics:.1f}s", flush=True)

    (run_dir / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "config": cfg}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    ckpt_dir = Path(train_cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        ckpt_dir / f"{cfg['model_name']}_{cfg['loss_name']}_seed{seed}.pt",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=float))
    print(f"结果已写入 {run_dir / 'metrics.json'}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="CVR 纠偏实验入口")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("overrides", nargs="*", help="形如 data=aliccp train.seed=0 的覆盖项")
    args = parser.parse_args()
    run(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
