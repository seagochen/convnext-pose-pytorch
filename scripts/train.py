#!/usr/bin/env python3
"""
ConvNeXt-Pose 训练入口脚本

用法:
    python scripts/train.py --data /path/to/dataset.yaml --backbone tiny --epochs 210

支持的参数:
    --data: YOLO 数据集配置文件 (必需)
    --backbone: tiny/small/base/large
    --head-type: heatmap/paf/yolo
    --img-size: 输入尺寸 (H W)
    --batch-size: 批大小
    --epochs: 训练轮数
    --lr: 学习率
    --amp: 混合精度训练
    --ema: 指数移动平均
    --output-dir: 输出目录
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from convnext_pose.training.config import get_parser, build_config
from convnext_pose.training.trainer import Trainer


def main():
    # 解析命令行参数
    parser = get_parser()
    args = parser.parse_args()

    # 构建配置
    config = build_config(args)

    # 创建训练器 (resume 在 Trainer.__init__ 中处理)
    trainer = Trainer(config)

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
