#!/usr/bin/env python3
"""
ConvNeXt-Pose 训练入口脚本

用法:
    python scripts/train.py --data /path/to/dataset.yaml --backbone tiny --epochs 210

支持的参数:
    --data: YOLO 数据集配置文件 (必需)
    --backbone: tiny/small/base/large
    --img-size: 输入尺寸 (H W)
    --batch-size: 批大小
    --epochs: 训练轮数
    --lr: 学习率
    --ema: 指数移动平均
    --output-dir: 输出目录
"""

import sys
from pathlib import Path

import torch

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from convnext_pose.training.config import get_parser, build_config
from convnext_pose.training.trainer import Trainer


def main():
    # 解析命令行参数
    parser = get_parser()
    parser.add_argument('--debug-nan', action='store_true',
                        help='Enable NaN detection (slower but helps debug)')
    args = parser.parse_args()

    # [调试] 开启异常检测，定位 NaN 的反向传播来源
    if args.debug_nan:
        print("=" * 60)
        print("[DEBUG] NaN detection enabled - training will be slower")
        print("=" * 60)
        torch.autograd.set_detect_anomaly(True)

    # 构建配置
    config = build_config(args)

    # 创建训练器 (resume 在 Trainer.__init__ 中处理)
    trainer = Trainer(config)

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
