"""
配置管理

纯命令行驱动的配置系统
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List


# 可用的 Backbone 配置
AVAILABLE_BACKBONES = {
    'tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
    'small': {'depths': [3, 3, 27, 3], 'dims': [96, 192, 384, 768]},
    'base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
    'large': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536]},
}


def get_parser() -> argparse.ArgumentParser:
    """获取命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='ConvNeXt-Pose Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据相关
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data', type=str, required=True,
                           help='YOLO 数据集配置文件路径 (dataset.yaml)')
    data_group.add_argument('--img-size', type=int, nargs=2, default=[640, 640],
                           metavar=('H', 'W'),
                           help='输入图像大小')
    data_group.add_argument('--output-size', type=int, nargs=2, default=[160, 160],
                           metavar=('H', 'W'),
                           help='输出热图大小 (stride=4 时)')

    # 模型相关
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--backbone', type=str, default='tiny',
                            choices=list(AVAILABLE_BACKBONES.keys()),
                            help='ConvNeXt backbone 类型')
    model_group.add_argument('--head-type', type=str, default='heatmap',
                            choices=['heatmap', 'paf', 'yolo'],
                            help='检测头类型')
    model_group.add_argument('--num-keypoints', type=int, default=17,
                            help='关键点数量')
    model_group.add_argument('--pretrained', type=str, default=None,
                            help='预训练 backbone 权重路径')
    model_group.add_argument('--list-backbones', action='store_true',
                            help='列出所有可用的 backbone')

    # 训练参数
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=210,
                            help='训练轮数')
    train_group.add_argument('--batch-size', type=int, default=32,
                            help='批大小')
    train_group.add_argument('--lr', type=float, default=1e-3,
                            help='初始学习率')
    train_group.add_argument('--weight-decay', type=float, default=0.05,
                            help='权重衰减')
    train_group.add_argument('--lr-scheduler', type=str, default='cosine',
                            choices=['cosine', 'step', 'multi_step'],
                            help='学习率调度器')
    train_group.add_argument('--warmup-epochs', type=int, default=5,
                            help='Warmup 轮数')

    # 高级训练选项
    adv_group = parser.add_argument_group('Advanced Training')
    adv_group.add_argument('--amp', action='store_true',
                          help='使用混合精度训练')
    adv_group.add_argument('--ema', action='store_true',
                          help='使用指数移动平均')
    adv_group.add_argument('--ema-decay', type=float, default=0.9999,
                          help='EMA 衰减率')
    adv_group.add_argument('--freeze-backbone', action='store_true',
                          help='冻结 backbone')
    adv_group.add_argument('--freeze-epochs', type=int, default=10,
                          help='冻结 backbone 的轮数')
    adv_group.add_argument('--gradient-accumulation', type=int, default=1,
                          help='梯度累积步数')

    # 损失函数
    loss_group = parser.add_argument_group('Loss')
    loss_group.add_argument('--loss-type', type=str, default='mse',
                           choices=['mse', 'awing', 'combined'],
                           help='损失函数类型')

    # 输出和日志
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output-dir', type=str, default='./runs/train',
                             help='输出目录')
    output_group.add_argument('--project', type=str, default='convnext-pose',
                             help='项目名称')
    output_group.add_argument('--name', type=str, default='exp',
                             help='实验名称')
    output_group.add_argument('--log-interval', type=int, default=50,
                             help='日志打印间隔')
    output_group.add_argument('--val-interval', type=int, default=5,
                             help='验证间隔')
    output_group.add_argument('--save-interval', type=int, default=10,
                             help='保存间隔')

    # 恢复和加载
    resume_group = parser.add_argument_group('Resume')
    resume_group.add_argument('--resume', type=str, default=None,
                             help='恢复训练: 实验名称(如exp)或检查点路径')
    resume_group.add_argument('--weights', type=str, default=None,
                             help='加载模型权重 (不恢复优化器状态)')

    # 其他
    misc_group = parser.add_argument_group('Misc')
    misc_group.add_argument('--num-workers', type=int, default=4,
                           help='数据加载线程数')
    misc_group.add_argument('--seed', type=int, default=42,
                           help='随机种子')
    misc_group.add_argument('--device', type=str, default='cuda',
                           help='设备 (cuda 或 cpu)')

    return parser


def resolve_resume_path(resume: str, output_dir: str, project: str) -> str:
    """解析恢复训练路径

    支持以下格式:
    - 实验名称: "exp" -> ./runs/train/convnext-pose/exp/weights/last.pt
    - 完整路径: "/path/to/checkpoint.pt"

    Args:
        resume: 用户输入的恢复路径或实验名称
        output_dir: 输出目录
        project: 项目名称

    Returns:
        解析后的完整检查点路径
    """
    resume_path = Path(resume)

    # 如果是完整路径且存在，直接返回
    if resume_path.is_file():
        return str(resume_path)

    # 如果是 .pt 文件路径但不存在，报错
    if resume.endswith('.pt'):
        return resume  # 让后续代码报错

    # 尝试作为实验名称解析
    # 格式: output_dir/project/exp_name/weights/last.pt
    exp_checkpoint = Path(output_dir) / project / resume / 'weights' / 'last.pt'
    if exp_checkpoint.is_file():
        return str(exp_checkpoint)

    # 如果找不到 last.pt，尝试 best.pt
    exp_best = Path(output_dir) / project / resume / 'weights' / 'best.pt'
    if exp_best.is_file():
        return str(exp_best)

    # 都找不到，返回原始输入让后续报错
    return resume


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """从命令行参数构建配置字典

    Args:
        args: 解析后的命令行参数

    Returns:
        配置字典
    """
    # 解析 resume 路径
    resume_path = None
    if args.resume:
        resume_path = resolve_resume_path(args.resume, args.output_dir, args.project)

    config = {
        # 数据配置
        'data': {
            'yaml_path': args.data,
            'input_size': tuple(args.img_size),
            'output_size': tuple(args.output_size),
            'num_workers': args.num_workers,
        },

        # 模型配置
        'model': {
            'backbone': args.backbone,
            'head_type': args.head_type,
            'num_keypoints': args.num_keypoints,
            'pretrained': args.pretrained,
        },

        # 训练配置
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'lr_scheduler': args.lr_scheduler,
            'warmup_epochs': args.warmup_epochs,
            'amp': args.amp,
            'ema': args.ema,
            'ema_decay': args.ema_decay,
            'freeze_backbone': args.freeze_backbone,
            'freeze_epochs': args.freeze_epochs,
            'gradient_accumulation': args.gradient_accumulation,
            'loss_type': args.loss_type,
        },

        # 输出配置
        'output': {
            'dir': args.output_dir,
            'project': args.project,
            'name': args.name,
            'log_interval': args.log_interval,
            'val_interval': args.val_interval,
            'save_interval': args.save_interval,
        },

        # 其他
        'resume': resume_path,
        'weights': args.weights,
        'seed': args.seed,
        'device': args.device,
    }

    return config


def list_backbones():
    """打印所有可用的 backbone"""
    print("\nAvailable Backbones:")
    print("-" * 60)
    print(f"{'Name':<12} {'Depths':<20} {'Dims':<30}")
    print("-" * 60)

    for name, cfg in AVAILABLE_BACKBONES.items():
        depths = str(cfg['depths'])
        dims = str(cfg['dims'])
        print(f"{name:<12} {depths:<20} {dims:<30}")

    print("-" * 60)
    print("\nExample usage:")
    print("  python scripts/train.py --data dataset.yaml --backbone tiny")
    print("  python scripts/train.py --data dataset.yaml --backbone small --amp")


def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置有效性

    Args:
        config: 配置字典

    Returns:
        是否有效
    """
    # 检查必要字段
    if not config['data']['yaml_path']:
        print("Error: --data is required")
        return False

    # 检查 backbone
    if config['model']['backbone'] not in AVAILABLE_BACKBONES:
        print(f"Error: Unknown backbone '{config['model']['backbone']}'")
        print(f"Available: {list(AVAILABLE_BACKBONES.keys())}")
        return False

    # 检查输入输出尺寸比例
    input_h, input_w = config['data']['input_size']
    output_h, output_w = config['data']['output_size']

    if input_h % output_h != 0 or input_w % output_w != 0:
        print(f"Warning: Input size {config['data']['input_size']} should be "
              f"divisible by output size {config['data']['output_size']}")

    return True
