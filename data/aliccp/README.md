# Ali-CCP 数据域

> 本目录是**数据的唯一入口**。任何脚本读写数据都必须按本文的分层约定走，不允许在层之间复制文件。
> 逐字节事实（列结构、分隔符、标签分布）见 [08-数据探测记录.md](../../doc/202610-选题/08-数据探测记录.md)；清洗规则见 [06-数据清洗与构建方案.md](../../doc/202610-选题/06-数据清洗与构建方案.md)。

## 1. 分层结构

```
data/aliccp/
├─ 00_raw/                 L0 原始层：解压原样，只读
│  ├─ train/               sample_skeleton_train.csv  common_features_train.csv
│  ├─ test/                sample_skeleton_test.csv   common_features_test.csv
│  └─ manifest.json        字节数 + md5 + 行数 + 标签量（质检基线）
├─ 10_interim/             L1 中间层：可删可重建
│  └─ common_index/        user_id → 特征分桶字典（Parquet）
├─ 15_warehouse/           L2 数仓层：Hive ODS/DWD/DWS/ADS（ORC + Parquet，可由 L0 重建）
│  ├─ ods_aliccp_skeleton/  ods_aliccp_common/
│  ├─ dwd_aliccp_exposure/  dwd_aliccp_user_feature/
│  ├─ dws_aliccp_vocab/     dws_aliccp_user_stat/
│  └─ ads_aliccp_train/     ads_aliccp_profile/
├─ 20_processed/           L2 成品层：清洗+编码后的训练输入
│  ├─ train/  val/  test/  每片 200 万行的 int32 Parquet
│  └─ vocab.json           只用 train 统计的词表
├─ 30_sample/              L3 采样层：10% 分层抽样，仅调参
├─ 40_semisynth/           L3 半合成层：C5 偏差基准，由 L2 派生
├─ 90_reports/             L4 报告层：统计与质检，论文数字的唯一来源
│  ├─ 01_probe/            字节探测：stats_raw.json、label_stats.json、field_inventory.json
│  ├─ 02_cleaning/         清洗报告：cleaning_report.json
│  └─ 03_dataset/          成品画像：stats.json（论文表 1）
└─ 99_archive/             废弃版本，仅留档
```

编号即数据流方向：**00 → 10 → 20 → 30 / 40**，`90` 旁挂观测，`99` 收容废弃。看到文件在哪个目录，就知道它处于生命周期的哪一步。

## 2. 每层的契约

| 层              | 唯一职责              | 输入                      | 输出                        | 可删           | 可改                   |
| -------------- | ----------------- | ----------------------- | ------------------------- | ------------ | -------------------- |
| `00_raw`       | 保真存储原始数据          | 下载的 `*.tar.gz`          | 4 个 csv + manifest        | 否（重下 9.4 GB） | **禁止**（只读）           |
| `10_interim`   | 把两张表 join 起来      | `00_raw`                | 分桶特征字典                    | 是（可从 L0 重建）  | 允许                   |
| `15_warehouse` | Hive 四层表（ODS/DWD/DWS/ADS） | `00_raw` | ORC（ODS/DWD/DWS）与 Parquet（ADS）表目录 | 是（可从 L0 重建） | 允许（重建而非手改）        |
| `20_processed` | 清洗 + 编码成训练可直接读的格式 | `00_raw` + `10_interim` | Parquet 分片 + `vocab.json` | 是（重建约数十分钟）   | **禁止**（重建而非手改）       |
| `30_sample`    | 提供快速迭代的小数据集       | `20_processed`          | 10% 分层采样                  | 是            | 允许                   |
| `40_semisynth` | 提供偏差强度可控的基准（C5）   | `20_processed` + 教师模型   | 各 β 一份伪真值 + 观测指示          | 是            | 允许                   |
| `90_reports`   | 记录每次运行的统计与质检结果    | 各层                      | json                      | 否（论文证据）      | **禁止**（追加新文件，不覆盖旧结果） |
| `99_archive`   | 存废弃版本，避免误用        | —                       | —                         | 是            | 允许                   |

三条硬规则：

1. **`00_raw` 永不修改**。清洗是"读 raw、写 processed"，不是原地改 raw。这样任何一次清洗都能重跑、能对比。
2. **`20_processed` 不可手改**。发现清洗规则有问题就改代码重建，不允许手改 Parquet——否则论文数字无法复现。
3. **论文里的数字只能来自 `90_reports`**，不能来自终端输出或临时脚本的 stdout。

## 3. 命名规范

| 对象         | 规则                                       | 示例                                   |
| ---------- | ---------------------------------------- | ------------------------------------ |
| Parquet 分片 | `part-{序号:05d}.parquet`                  | `part-00000.parquet`                 |
| 报告 json    | `{内容}_{时间戳}.json`，时间戳 `YYYYmmdd-HHMM`    | `cleaning_report_20261107-2310.json` |
| 词表         | 固定 `vocab.json`（唯一，覆盖前先备份到 `99_archive`） | `20_processed/vocab.json`            |
| 数据版本       | 目录名后缀 `_v{主}.{次}`                        | `20_processed_v1.0`，放 `99_archive`   |

不出现中文文件名、空格、大写字母——避免跨平台与命令行转义问题。

## 4. 当前状态

> 进度与变更记录统一记在 [doc/202610-选题/09-进度台账.md](../../doc/202610-选题/09-进度台账.md)，本表只记录各层数据是否已落盘这一事实。

| 层                     | 状态      | 说明                                                                                                   |
| --------------------- | ------- | ---------------------------------------------------------------------------------------------------- |
| `00_raw`              | **已完成** | 4 个 CSV 共 38.6 GiB，md5 与随包 `.md5` 一致                                                                 |
| `10_interim`          | **已完成** | 64 个分片、3.55 GiB；行数与 L0 完全一致（730,600 / 884,212），0 跳过                                                  |
| `15_warehouse`        | **在建**  | Hive 库 `cvr_debias`；ODS→DWD→DWS→ADS 逐层对账，验收数字见 12 号记录 E18 起                                              |
| `20_processed`        | **已完成** | `_encoded`（K=8，1.61/1.67 GiB）+ `train/val/test`（40.18M / 2.12M / 43.02M 行，合计 3.33 GiB）+ `vocab.json` |
| `30_sample`           | **已完成** | 10% 分层采样，4,018,471 / 211,517 / 4,301,660 行，390 MB；CTR/CVR 偏差 ≤2.73%                              |
| `40_semisynth`        | 空       | 依赖 L2 + 教师模型                                                                                         |
| `90_reports/01_probe` | **已完成** | 行数、标签分布、23 个特征域清单已产出                                                                                 |

## 5. 原始层事实（质检基线）

| 文件                                | 行数         | 字节             |
| --------------------------------- | ---------- | -------------- |
| `train/sample_skeleton_train.csv` | 42,300,135 | 10,891,589,135 |
| `train/common_features_train.csv` | 730,600    | 8,590,621,765  |
| `test/sample_skeleton_test.csv`   | 43,016,840 | 11,018,450,696 |
| `test/common_features_test.csv`   | 884,212    | 10,997,078,207 |

标签量见 `90_reports/01_probe/label_stats.json`。**下游任何一步产出的统计量与这里偏差超过 0.5%，一律视为解析错误，停止实验**。

## 6. 复现顺序

```bash
# L0 → L1 → L2（清洗与编码）
python -m src.data.prepare_aliccp --stage index     # 建公共特征字典
python -m src.data.prepare_aliccp --stage join      # 流式 join + 编码
python -m src.data.prepare_aliccp --stage vocab     # 只用 train 建词表
python -m src.data.prepare_aliccp --stage clean     # 清洗规则 C1–C8 + 报告

# L2 → 报告
python -m src.data.stats --data data/aliccp/20_processed

# L2 → L3（仅调参用）
python -m src.data.sample --ratio 0.1 --out data/aliccp/30_sample

# L2 → L3（半合成基准，C5）
python -m src.data.build_semisynth --teacher artifacts/models/teacher_cvr.pt
```

每条命令都可单独重跑，且只写自己那一层。

## 7. 磁盘预算

| 层              | 实测/预估        | 说明                                                   |
| -------------- | ------------ | ---------------------------------------------------- |
| `00_raw`       | 38.6 GiB     | 已完成                                                  |
| `10_interim`   | 3.55 GiB（实测） | 64 个分片，zstd 压缩                                       |
| `15_warehouse` | 预估 15–20 GiB | ODS/DWD 用 ORC+Snappy；DWD 拆明细+维表，避免用户特征 58 倍冗余 |
| `20_processed` | 6.61 GiB（实测） | `_encoded` 3.28 GiB + `train/val/test` 3.33 GiB，zstd |
| `30_sample`    | 390 MB（实测）   | L2 的 10%，三段合计                                        |
| `40_semisynth` | 约 5 GiB      | 5 个 β，只存伪真值与观测指示                                     |
| 合计             | 约 75 GiB     | 当前 E 盘剩余 115 GiB，够用                                 |

`10_interim` 在 L2 建成后可删——它是唯一"用完即弃"的层。
