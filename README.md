# 点击后转化率纠偏（本科毕业设计项目）

面向样本选择偏差（SSB）的电商点击后转化率（CVR）多任务预估研究。主数据集 Ali-CCP，备选 AliExpress 五国。

本项目由《点击后转化率纠偏研究总案》落地而成，按**本科口径**裁剪：数据科学与大数据技术专业，1.8–2.5 万字，6 个模型配置（3 基线 + 3 变体），创新点 C1+C2+C5。选题文档见 `doc/202610-选题/`。

## 一句话问题

CTR 预估里每条曝光都有标签；CVR 预估里只有被点击的样本才带转化标签。训练分布（点击空间）与推理分布（曝光空间）不一致，这就是样本选择偏差。技术主线因此固定为：**有偏训练集 → 全空间建模 → 因果加权纠偏**。

## 项目结构

```
cvr-debias/
├─ doc/202610-选题/           # 选题、开题、方案、部署、数据探测文档
├─ doc/tables/                # make_tables.py 生成的论文表格
├─ data/aliccp/               # 数据域（分层约定见 data/aliccp/README.md）
│  ├─ 00_raw/                 # L0 原始：train/ test/ + manifest，只读
│  ├─ 10_interim/             # L1 中间：公共特征字典，可弃可重建
│  ├─ 15_warehouse/           # L2 数仓：Hive ODS/DWD/DWS/ADS（ORC + Parquet）
│  ├─ 20_processed/           # L2 成品：parquet 分片 + vocab.json，训练唯一输入
│  ├─ 30_sample/              # L3 调参采样：10% 分层抽样
│  ├─ 40_semisynth/           # L3 半合成基准（C5）
│  ├─ 90_reports/             # L4 报告：probe / cleaning / dataset / warehouse，论文数字来源
│  └─ 99_archive/             # 废弃版本留档
├─ configs/                   # 配置驱动：一条实验 = 一行命令
│  ├─ config.yaml             # 主配置（data / model / loss / exp 四个组）
│  ├─ data/    synthetic.yaml  aliccp.yaml  aliccp_10p.yaml  semisynth.yaml  aliexpress.yaml
│  ├─ model/   mlp_cvr.yaml  esmm.yaml  mmoe.yaml  uc_escm.yaml
│  ├─ loss/    bce.yaml  esmm.yaml  ips.yaml  dr.yaml  vrdr.yaml
│  └─ exp/     smoke.yaml  main.yaml  ablation.yaml
├─ src/
│  ├─ data/      synthetic.py  dataset.py  prepare_aliccp.py  sample.py  build_semisynth.py  stats.py
│  ├─ spark/     hive_pipeline.py  tuning.py   # Hive 四层表 + 四项调优
│  ├─ features/  embedding.py            # 共享稀疏特征层
│  ├─ models/    base.py  experts.py  towers.py  baselines.py  uc_escm.py
│  ├─ losses/    causal.py  weighting.py  regularizer.py
│  ├─ calibration/ temperature.py        # 温度标度 + 分位数截断
│  ├─ evaluation/  metrics.py  significance.py
│  └─ train.py
├─ scripts/   download_data.py  count_raw.py  label_stats.py  field_inventory.py
│             run_smoke.sh  run_main.sh  run_formal_baselines.py  make_tables.py
├─ deploy/    serve.py  Dockerfile  docker-compose.yml  requirements.txt
└─ tests/
```

## 部署（演示）

```bash
pip install -r deploy/requirements.txt
RUN_DIR=artifacts/runs/smoke/uc_escm/seed0 python -m uvicorn deploy.serve:app --port 8000
curl -s localhost:8000/healthz
```

容器：`docker compose -f deploy/docker-compose.yml up -d --build`。切换模型只改 `RUN_DIR`，不重建镜像。本机 Docker 装在 WSL2 里，命令要在 WSL 内执行：`wsl -d Ubuntu -u root -- sh -c "cd /mnt/e/cvr-debias && docker compose -f deploy/docker-compose.yml ps"`。完整流程与验收清单见 [07-部署方案.md](doc/202610-选题/07-部署方案.md)。

## 快速开始

```bash
pip install -r requirements.txt

# 1. 冒烟：合成数据上跑通 UC-ESCM + 方差正则双重稳健，CPU 即可
python -m src.train exp=smoke data=synthetic model=uc_escm loss=vrdr train.seed=0

# 2. 单元测试
python -m pytest tests -q

# 3. 真实数据：L0 已就位，按层往下走（完整口径见 doc/202610-选题/10-复现清单.md）
python -m src.data.prepare_aliccp --stage verify     # 校验 L0 与 manifest
python -m src.data.prepare_aliccp --stage index      # L0 -> L1
python -m src.data.prepare_aliccp --stage bucket     # 骨架按 user 哈希分 32 桶
python -m src.data.prepare_aliccp --stage joinencode --workers 3   # 全量 join + 编码
python -m src.data.prepare_aliccp --stage vocab      # 只用 train 建词表
python -m src.data.prepare_aliccp --stage clean      # 清洗 C1–C8 + 报告
python -m src.data.stats                             # L2 -> 90_reports/03_dataset/stats.json
python -m src.data.sample --ratio 0.1                # L2 -> 30_sample（仅调参）

# 4. Hive 四层表（Spark 3.5 + Hive，需先设置 PYSPARK_PYTHON）
spark-submit --master "local[8]" --driver-memory 6g src\spark\hive_pipeline.py --stage ods
spark-submit --master "local[8]" --driver-memory 6g src\spark\hive_pipeline.py --stage dwd
spark-submit --master "local[8]" --driver-memory 6g src\spark\hive_pipeline.py --stage dws
spark-submit --master "local[8]" --driver-memory 6g src\spark\hive_pipeline.py --stage ads
spark-submit --master "local[8]" --driver-memory 4g src\spark\hive_pipeline.py --stage profile --source processed

# 5. 训练与聚表（正式配置：bs=16384 + 2 worker，E30）
python -m src.train data=aliccp model=uc_escm loss=vrdr exp=main \
    train.seed=0 train.batch_size=16384 train.num_workers=2 train.device=cuda
python scripts/make_tables.py --runs artifacts/runs --out doc/tables/
```

> 全量训练已在 RTX 3060 上跑通：bs=16384 下 MLP-CVR 稳态 262,800 行/秒；单 run 10 轮约 28–33 min（轻量模型）、uc_escm 约 47 min（E30）。`exp/main.yaml` 默认仍是 4096，跑正式结果必须显式覆盖 `train.batch_size=16384`。若遇到 DataLoader spawn 卡死，把 `train.num_workers` 降为 0。

## 数据获取（Ali-CCP）

数据源：天池数据集 <https://tianchi.aliyun.com/dataset/408>。**必须登录并同意数据协议**，匿名请求只会返回 4 KB 的页面壳（302 / 403），所以脚本不猜直链，也不绕过授权。

> **本项目数据已就位**：`sample_train.tar.gz` / `sample_test.tar.gz` 已解压到 `data/aliccp/00_raw/{train,test}/`，MD5 与随包 `.md5` 一致。下面的下载步骤仅换机器或重取时用。

```bash
# 1) 浏览器登录天池 → 打开数据集页 → 点「下载」→ 复制真实文件链接
# 2) 交给下载器（断点续传 + md5 + 自动解包）
python scripts/download_data.py --out data/aliccp/_incoming --url "<链接>"

# 若返回 302/403，补上登录态
python scripts/download_data.py --out data/aliccp/_incoming --url "<链接>" \
    --cookie "<浏览器 Cookie>" --referer "https://tianchi.aliyun.com/dataset/408"

# 解包后手工归位到 L0（下载器不决定分层）：
#   sample_skeleton_train.csv / common_features_train.csv  -> data/aliccp/00_raw/train/
#   sample_skeleton_test.csv  / common_features_test.csv   -> data/aliccp/00_raw/test/
```

下载完先对量级（本数据集实测）：train 曝光 42,300,135 / 点击 1,644,256 / 转化 9,032，test 曝光 43,016,840 / 点击 1,673,447 / 转化 9,421。偏差 > 0.5% 就别往下跑，先查文件是否截断（`00_raw/manifest.json` 里有字节数与 md5）。

合规：原始日志不进仓库（`.gitignore` 已排除 `data/`），只提交 `manifest.json` 与预处理脚本；所有 user_id 在预处理阶段哈希脱敏。


## 四条必须先落地的纪律

1. **多随机种子**：每配置 5 个种子，报 mean ± std。
2. **显著性检验**：与最强基线做配对双尾 t 检验 + Holm-Bonferroni 校正 + bootstrap 置信区间。
3. **独立验证集**：温度 T、截断分位数 α、早停轮数、损失权重全部在验证集上定，测试集只用于最终报告。
4. **半合成基准**：真实数据上量不出真实偏差，偏差强度必须由可控 logging policy 给出（C5）。

## 代码与总案章节的对应

| 代码 | 总案章节 | 说明 |
| --- | --- | --- |
| `src/features/embedding.py` | §2.2、§6.3 | 22 个稀疏域各一张 embedding 表、统一维度（101 全量 OOV 已剔除） |
| `src/models/experts.py` | §2.2（C3） | 因果解耦专家层，本科进展望章，代码保留 |
| `src/models/uc_escm.py` | §2.2、§2.3 | 4 层 3 塔，输出同时供损失层使用 |
| `src/losses/causal.py` | §2.3（C1/C2） | IPS / DR / 方差正则 DR，`ẽ` 必须 detach |
| `src/losses/weighting.py` | §2.3（C4） | 同方差不确定性加权，本科进展望章 |
| `src/calibration/temperature.py` | §2.3（C1）、§5.3 | 温度标度 + 分位数截断，服务侧可热更新 |
| `src/evaluation/metrics.py` | §4.2 | 指标唯一实现，全部输出到 metrics.json |
| `src/evaluation/significance.py` | §4.3 | 配对 t 检验、bootstrap、Holm 校正 |
| `src/data/build_semisynth.py` | §3.4（C5） | 本科不可删减的最小可发表集合之一 |

## 当前进度

**进度只记在 [doc/202610-选题/09-进度台账.md](doc/202610-选题/09-进度台账.md)，本节不再维护。**

复现命令与验收数字见 [10-复现清单.md](doc/202610-选题/10-复现清单.md)；下一步计划与逐步确认见 [13-路线确认单.md](doc/202610-选题/13-路线确认单.md)。

## 冒烟自检结果（2026-10-30）

合成数据 2 万条（点击率 4.3%、点击样本转化率 54%），8 epoch，CPU，`train.seed=0`。只用于验证链路接通，**不是论文数字**。

| 模型 / 损失 | CTR AUC | CTR PCOC | CTCVR AUC | CVR AUC | CVR PCOC | Bias vs p* | T |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mlp_cvr / bce | — | — | 0.837 | 0.751 | 1.15 | 0.177 | — |
| esmm / esmm | 0.867 | 0.92 | 0.931 | 0.772 | 0.89 | 0.068 | 0.89 |
| mmoe / esmm | 0.868 | 0.91 | 0.931 | 0.767 | 0.61 | -0.014 | 0.89 |
| uc_escm / vrdr | 0.852 | 1.04 | 0.922 | 0.777 | 0.94 | 0.252 | 1.13 |

自检中发现并修掉的三处问题（都已写进注释或文档，避免真实实验重犯）：

1. **正交正则 λ=0.1 会把 CTR 头打死**（AUC 0.16、T 贴到上界）。C3 属展望章，主实验默认 `orth_lambda: 0.0`，消融时用 `loss.orth_lambda=0.01`。
2. **温度拟合不能加正样本权重**。负正比加权把 T 从 0.96 拉到 5.57、预测均值从 0.046 抬到 0.23，PCOC 从 1.0 变成 5.6。`fit_temperature` 已去掉该参数，并加了回归测试。
3. **α 选择准则里的 ECE 必须算在截断后的 ẽ 上**，否则 ECE 对 α 是常数，准则退化成只看方差、必然选最大的 α。

另：合成数据原先的特征是纯随机 id，与点击/转化无关，模型学不到任何东西（AUC≈0.5）。现改为把 latent 分桶后加噪生成特征，故上表 AUC 在 0.75–0.93 之间，能真正检验建模链路。

## 许可证

MIT License，见 [LICENSE](LICENSE)。
