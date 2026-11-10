"""半合成偏差基准构造骨架（C5，本科不可删减的最小可发表集合）。

目标：造一个转化标签全部已知、但点击按可控策略选择性暴露的数据集，
从而能直接算出每个模型的真实偏差。

四步：
1. 在 Ali-CCP 点击样本上训教师 CVR 模型 f*，对全部曝光输出 p*(x) 作为伪真值，
   采样得到全标签数据集 D_full；
2. 日志策略 π_β(x) = σ(β·g(x) + b)，β = 0 为完全随机暴露，β 越大 SSB 越强；
   b 用于把整体点击率钉在 4%，保证不同 β 之间可比；
3. 按 π_β 采样观测指示 o，仅 o = 1 时暴露 z，得到有偏训练集；
4. 评估时回到 D_full，直接算 Bias = E[ẑ - p*] 与 MSE。

必须写进论文的局限：伪真值来自模型，带教师模型自身的偏差。
半合成结果只用于方法间相对比较，绝对数值不作结论。
"""

import argparse
import math
from pathlib import Path

import torch


def logging_policy(score: torch.Tensor, beta: float, b: float) -> torch.Tensor:
    """π_β(x) = σ(β · g(x) + b)，score 为与 p*(x) 正相关的单调打分。"""
    return torch.sigmoid(beta * score + b)


def solve_bias_for_target_ctr(score: torch.Tensor, beta: float, target: float = 0.04) -> float:
    """二分求 b，使整体点击率落在 target（默认 4%）。"""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        ctr = float(logging_policy(score, beta, mid).mean())
        if ctr > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def main():
    parser = argparse.ArgumentParser(description="半合成偏差基准构造")
    parser.add_argument("--parquet", default="data/aliccp/20_processed",
                        help="L2 成品层（默认 data/aliccp/20_processed）")
    parser.add_argument("--teacher", required=True, help="教师 CVR 模型 checkpoint")
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--out", default="data/aliccp/40_semisynth",
                        help="半合成基准落 L3 同级新层，不污染 L2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("步骤 1/4 载入教师模型并对全部曝光打分，得到伪真值 p*")
    # TODO(真实数据)：载入 teacher ckpt、对全量曝光推理、采样生成 D_full
    print("步骤 2/4 为每个 β 二分求 b，把点击率钉在 4%")
    # score = p* 的 logit；对每个 β 调 solve_bias_for_target_ctr
    print("步骤 3/4 按 π_β 采样观测指示 o，仅 o=1 暴露 z")
    # TODO(真实数据)：落盘每个 β 的 标签 + 指示（特征复用主集）
    print("步骤 4/4 输出 metadata.json：beta / b / 实际点击率 / 随机种子")
    # TODO(真实数据)：写 metadata，供 evaluate 计算 Bias / MSE
    print(f"输出目录：{out_dir}（每个 β 一个子目录，约 2–4 GB）")
    print("提示：伪真值与偏差强度必须分开存储，β 与种子入参，便于复现。")


if __name__ == "__main__":
    main()
