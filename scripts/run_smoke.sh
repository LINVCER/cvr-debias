#!/usr/bin/env bash
# 冒烟：5 分钟内必须跑通，不通不往下走
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pytest tests -q
python -m src.train exp=smoke data=synthetic model=uc_escm loss=vrdr train.seed=0
python -m src.train exp=smoke data=synthetic model=esmm loss=esmm train.seed=0
python -m src.train exp=smoke data=synthetic model=mlp_cvr loss=bce train.seed=0
python -m src.train exp=smoke data=synthetic model=mmoe loss=esmm train.seed=0
python scripts/make_tables.py --runs artifacts/runs --out doc/tables
