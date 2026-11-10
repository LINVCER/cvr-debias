#!/usr/bin/env bash
# 主实验：6 个模型配置（3 基线 + 3 变体）× 5 个种子，串行排队
# 3 基线 = MLP-CVR / ESMM / MMoE；3 变体 = ESCM²-IPS / ESCM²-DR / UC-ESCM
set -euo pipefail
cd "$(dirname "$0")/.."

PARAMS="train.batch_size=16384 train.num_workers=2 train.device=cuda"

for seed in 0 1 2 3 4; do
  python -m src.train data=aliccp model=mlp_cvr loss=bce   exp=main train.seed=$seed $PARAMS
  python -m src.train data=aliccp model=esmm    loss=esmm  exp=main train.seed=$seed $PARAMS
  python -m src.train data=aliccp model=mmoe    loss=esmm  exp=main train.seed=$seed $PARAMS
  # ESCM²-IPS：UC-ESCM 结构 + IPS 损失，无插补塔
  python -m src.train data=aliccp model=uc_escm loss=ips   exp=main train.seed=$seed $PARAMS
  # ESCM²-DR：专家池全共享（等价于去 C3）+ DR 损失
  python -m src.train data=aliccp model=uc_escm loss=dr    exp=main train.seed=$seed $PARAMS \
      model.n_propensity=0 model.n_conversion=0 model.n_shared=8
  # UC-ESCM：C1 + C2，正交正则按自检结论留到消融
  python -m src.train data=aliccp model=uc_escm loss=vrdr  exp=main train.seed=$seed $PARAMS loss.orth_lambda=0
done

# 消融（C3 开关）：主实验另起，不混进主表
# python -m src.train data=aliccp model=uc_escm loss=vrdr exp=ablation train.seed=$seed loss.orth_lambda=0.01

python scripts/make_tables.py --runs artifacts/runs --out doc/tables
