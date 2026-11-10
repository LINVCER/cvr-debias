"""30 run 正式矩阵调度器 v2：2 进程并行，重/轻任务交错。

设计要点：
1. 已存在 metrics.json 的 run 直接 SKIP（可安全中断续跑）
2. 重任务（uc_escm，47 min）与轻任务（mlp/esmm/mmoe，28-33 min）交错入队，
   避免两个重任务同时抢 GPU 显存
3. 单卡 6GB：轻+重 组合峰值约 4GB，留 1GB 余量
4. 每个 run 独立日志，主日志记录 START/DONE/PROGRESS
"""
import subprocess, sys, os, time, threading, queue, json, shutil
from pathlib import Path

# 解释器默认用当前 Python；需要指定时设环境变量 PYTHON
PY = os.environ.get("PYTHON", sys.executable)
# 仓库根目录：脚本可在根目录或 scripts/ 下运行
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if _HERE.name == "scripts" else _HERE
os.chdir(ROOT)

# (model, loss, 是否重任务)
MODELS = [
    ("mlp_cvr", "bce", False),
    ("esmm", "esmm", False),
    ("mmoe", "esmm", False),
    ("uc_escm", "ips", True),
    ("uc_escm", "dr", True),
    ("uc_escm", "vrdr", True),
]
SEEDS = list(range(5))
BS = 16384
EPOCHS = 10
NW = 2
MAX_PARALLEL = 2
LOG = ROOT / "artifacts" / "baselines_formal.log"
STALE = ROOT / "artifacts" / "runs" / "main" / "_stale"


def metrics_matches(metrics_path: Path, model: str, loss: str, seed: int) -> bool:
    """已存在且配置与本任务完全一致才算完成。

    仅判断文件是否存在会误跳过：同一模型的不同 loss 曾共用 seed 目录，
    旧的低 bs 产物也会被当成本矩阵结果。
    """
    try:
        cfg = json.loads(metrics_path.read_text(encoding="utf-8"))["config"]
    except Exception:
        return False
    train_cfg = cfg.get("train", {})
    return (
        cfg.get("model_name") == model
        and cfg.get("loss_name") == loss
        and int(train_cfg.get("seed", -1)) == seed
        and int(train_cfg.get("batch_size", -1)) == BS
        and int(train_cfg.get("epochs", -1)) == EPOCHS
    )

# 构建重/轻两个列表
heavy, light = [], []
for model, loss, is_heavy in MODELS:
    for seed in SEEDS:
        (heavy if is_heavy else light).append((model, loss, seed))

# 交错合并：重, 轻, 重, 轻 ...
ordered = []
i = j = 0
while i < len(heavy) or j < len(light):
    if i < len(heavy):
        ordered.append(heavy[i]); i += 1
    if j < len(light):
        ordered.append(light[j]); j += 1

task_q = queue.Queue()
for t in ordered:
    task_q.put(t)

TOTAL = len(ordered)
results = []
lock = threading.Lock()
t_start = time.time()

LOG.parent.mkdir(parents=True, exist_ok=True)
with open(LOG, "a", encoding="utf-8") as f:
    f.write(f"\n[{time.strftime('%H:%M:%S')}] === 正式矩阵启动 共 {TOTAL} runs, 并行={MAX_PARALLEL}, bs={BS} ===\n")
    f.write(f"[{time.strftime('%H:%M:%S')}] 队列: 重任务 {len(heavy)} 个(uc_escm), 轻任务 {len(light)} 个, 交错执行\n")


def worker(wid):
    while True:
        try:
            model, loss, seed = task_q.get_nowait()
        except queue.Empty:
            return
        run_dir = ROOT / "artifacts" / "runs" / "main" / f"{model}_{loss}" / f"seed{seed}"
        metrics = run_dir / "metrics.json"
        if metrics.exists():
            if metrics_matches(metrics, model, loss, seed):
                with lock:
                    results.append(("SKIP", model, loss, seed, 0))
                with open(LOG, "a", encoding="utf-8") as f:
                    f.write(f"[{time.strftime('%H:%M:%S')}] SKIP   {model}+{loss} seed{seed}\n")
                task_q.task_done()
                continue
            # 配置不匹配（旧 bs / 其它 loss 的产物）：移入 _stale 留档后重跑
            STALE.mkdir(parents=True, exist_ok=True)
            dst = STALE / f"{run_dir.parent.name}_seed{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
            shutil.move(str(run_dir), str(dst))
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] STALE  {model}+{loss} seed{seed} 旧产物配置不符，已移入 {dst.name}\n")

        per_log = ROOT / "artifacts" / f"run_{model}_{loss}_seed{seed}.log"
        cmd = [PY, "-u", "-m", "src.train", "data=aliccp", f"model={model}", f"loss={loss}",
               "exp=main", f"train.seed={seed}", f"train.batch_size={BS}",
               f"train.num_workers={NW}", "train.device=cuda"]
        t0 = time.time()
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] START  w{wid} {model}+{loss} seed{seed}\n")
        with open(per_log, "w", encoding="utf-8") as pf:
            rc = subprocess.call(cmd, stdout=pf, stderr=subprocess.STDOUT, cwd=ROOT)
        dt = time.time() - t0
        ok = rc == 0 and metrics.exists()
        with lock:
            results.append(("OK" if ok else f"FAIL(rc={rc})", model, loss, seed, dt))
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {'OK  ' if ok else f'FAIL(rc={rc})'} w{wid} {model}+{loss} seed{seed}  用时 {dt/60:.1f} min\n")
        task_q.task_done()


threads = [threading.Thread(target=worker, args=(i,), daemon=False) for i in range(MAX_PARALLEL)]


def reporter():
    while any(t.is_alive() for t in threads):
        time.sleep(120)
        with lock:
            done = len(results)
            ok = sum(1 for r in results if r[0] == "OK")
            sk = sum(1 for r in results if r[0] == "SKIP")
            fail = sum(1 for r in results if r[0].startswith("FAIL"))
        el = (time.time() - t_start) / 60
        eta = (el / done * (TOTAL - done)) if done else 0
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] PROGRESS  {done}/{TOTAL}  ok={ok} skip={sk} fail={fail}  已耗 {el:.0f} min  预计剩余 {eta:.0f} min\n")


for t in threads:
    t.start()
rep = threading.Thread(target=reporter, daemon=True)
rep.start()
for t in threads:
    t.join()

el = (time.time() - t_start) / 60
with open(LOG, "a", encoding="utf-8") as f:
    f.write(f"[{time.strftime('%H:%M:%S')}] === 全部完成 总耗时 {el:.0f} min ===\n")
print(f"完成，总耗时 {el:.0f} min")
