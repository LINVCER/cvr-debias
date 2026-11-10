"""Parquet 分片读取：训练 / 验证 / 测试三份，标签列 click 与 conversion。

约定（来自预处理流水线，K=8 定长契约）：
- 每片 ≤50 万行，按 data/aliccp/20_processed/{train,val,test}/part-*.parquet 组织；
- 特征列为 f_<field>，类型 list<int32>[K]，0=缺失/补齐、1=OOV、>=2 真实取值；
- 标签单独存 click / conversion 两列；user_id 已哈希成 int64，只用于 GAUC 分组。

内存约束：K=8 下 train 全量拼接约 62 GB，不可整体 cat。本模块只做两件事——
按文件惰性产批（iter_batches），以及把分片读成 [B, F, K] 张量（load_split，
仅用于 val/test 或抽样后的子集）。训练循环应消费 iter_batches。
"""

from pathlib import Path

import numpy as np
import torch


def _table_to_tensors(table, label_cols) -> dict:
    """把一个 Arrow 表转成 {"ids": [B,F,K], "click", "conv", "user_ids"} 张量字典。"""
    names = table.column_names
    feat_cols = [c for c in names if c.startswith("f_")]
    if not feat_cols:
        raise ValueError(f"未找到 f_<field> 特征列，实际列：{names}")
    cols = []
    for c in feat_cols:
        col = table.column(c)
        # RecordBatch 里是 FixedSizeListArray，Table 里是 ChunkedArray，统一成连续数组
        if hasattr(col, "combine_chunks"):
            col = col.combine_chunks()
        # list<int32>[K] → [B, K]；先拍平再 reshape，规避 to_pandas 的对象列开销
        flat = col.values.to_numpy(zero_copy_only=False)
        cols.append(flat.reshape(len(table), -1))
    ids = torch.as_tensor(np.stack(cols, axis=1), dtype=torch.long)  # [B, F, K]

    # 直接从 Arrow 数组取标签，避免 to_pandas 的对象列开销
    label_np = np.stack(
        [table.column(c).to_numpy(zero_copy_only=False) for c in label_cols], axis=1
    )
    user = (
        torch.as_tensor(table.column("user_id").to_numpy(), dtype=torch.long)
        if "user_id" in names
        else torch.zeros(len(table), dtype=torch.long)
    )
    return {
        "ids": ids,
        "click": torch.as_tensor(label_np[:, 0], dtype=torch.float32),
        "conv": torch.as_tensor(label_np[:, 1], dtype=torch.float32),
        "user_ids": user,
    }


class ParquetIterableDataset(torch.utils.data.IterableDataset):
    """按 parquet 文件分片的 IterableDataset，配合 DataLoader 多 worker 并行读取。

    每个 worker 处理 files 的一个子集（按 worker_id 跨步取），逐文件流式产批，
    峰值内存只含一个 batch。shuffle=True 时打乱文件顺序（不打乱行内顺序，
    与原始 iter_batches 行为一致）。
    """

    def __init__(self, shard_dir, batch_size=8192, label_cols=("click", "conversion"),
                 files=None, shuffle=False, seed=0):
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.label_cols = label_cols
        if files is None:
            files = sorted(self.shard_dir.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"{self.shard_dir} 下没有 parquet 分片，先跑 prepare_aliccp.py")
        self.files = list(files)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        import random
        import pyarrow.parquet as pq

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files = self.files[worker_id::num_workers]
            seed = self.seed + worker_id
        else:
            files = self.files
            seed = self.seed

        if self.shuffle and len(files) > 1:
            rng = random.Random(seed)
            rng.shuffle(files)

        for f in files:
            for rb in pq.ParquetFile(f).iter_batches(batch_size=self.batch_size):
                yield _table_to_tensors(rb, self.label_cols)


def load_split(shard_dir, label_cols=("click", "conversion")) -> dict:
    """读取一个 split 的所有分片，拼成 [N, F, K] 张量字典（只适合 val/test 或子集）。"""
    import pyarrow.parquet as pq  # 仅真实数据路径需要，合成数据不装 pyarrow 也能跑

    shard_dir = Path(shard_dir)
    files = sorted(shard_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{shard_dir} 下没有 parquet 分片，先跑 prepare_aliccp.py")

    parts = [_table_to_tensors(pq.read_table(f), label_cols) for f in files]
    return {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}


def iter_batches(shard_dir, batch_size=8192, label_cols=("click", "conversion"), files=None):
    """逐分片、逐 record batch 产批，峰值内存只含一个 batch。

    files 给定时按其顺序读取（训练侧用于按 epoch 打乱分片顺序）。
    """
    import pyarrow.parquet as pq

    shard_dir = Path(shard_dir)
    if files is None:
        files = sorted(shard_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"{shard_dir} 下没有 parquet 分片，先跑 prepare_aliccp.py")
    for f in files:
        for rb in pq.ParquetFile(f).iter_batches(batch_size=batch_size):
            yield _table_to_tensors(rb, label_cols)


def aliccp_paths(data_cfg: dict) -> dict:
    """只返回三个分片目录，由训练循环惰性产批（train 全量拼接约 62 GB，不可 load_split）。"""
    return {
        "train": Path(data_cfg["train_shards"]),
        "val": Path(data_cfg["val_shards"]),
        "test": Path(data_cfg["test_shards"]),
    }


def load_aliccp(data_cfg: dict) -> dict:
    """train / val 由 train 分片按用户哈希切分，test 分片原样做测试集。

    切分在管道侧完成（按 user_id 哈希，防同一用户跨 train/val 泄漏），
    这里只读现成目录。val 是必选项：温度 T、截断分位数 α、早停、损失权重都在它上面定。
    """
    label_cols = data_cfg.get("label_cols", ("click", "conversion"))
    return {
        "train": load_split(data_cfg["train_shards"], label_cols),
        "val": load_split(data_cfg["val_shards"], label_cols),
        "test": load_split(data_cfg["test_shards"], label_cols),
    }
