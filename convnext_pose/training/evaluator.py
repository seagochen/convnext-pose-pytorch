"""
评估器类

用于模型评估和测试
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from pathlib import Path

from ..models import ConvNeXtPose
from ..utils.data import build_dataloader
from ..utils.metrics import PoseEvaluator, decode_heatmap, decode_heatmap_dark


class Evaluator:
    """姿态估计评估器

    Args:
        config: 配置字典
        checkpoint: 检查点路径
        use_dark: 是否使用 DARK 解码
    """

    def __init__(self,
                 config: Dict[str, Any],
                 checkpoint: str,
                 use_dark: bool = True):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))
        self.use_dark = use_dark

        # 构建模型
        self.model = self._build_model(checkpoint)

        # 构建数据加载器
        self.dataloader = self._build_dataloader()

        # 评估器
        self.evaluator = PoseEvaluator(
            num_keypoints=config['model']['num_keypoints']
        )

        # 计算步长
        input_size = config['data']['input_size']
        output_size = config['data']['output_size']
        self.stride = input_size[0] // output_size[0]

    def _build_model(self, checkpoint: str) -> nn.Module:
        """构建模型"""
        model_cfg = self.config['model']

        model = ConvNeXtPose(
            backbone=model_cfg['backbone'],
            num_keypoints=model_cfg['num_keypoints'],
            head_type=model_cfg['head_type'],
        )

        # 加载权重
        ckpt = torch.load(checkpoint, map_location=self.device)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        elif 'ema' in ckpt:
            # 优先使用 EMA 权重
            model.load_state_dict(ckpt['ema']['ema'])
        else:
            model.load_state_dict(ckpt)

        model = model.to(self.device)
        model.eval()

        return model

    def _build_dataloader(self) -> DataLoader:
        """构建数据加载器"""
        data_cfg = self.config['data']

        dataloader = build_dataloader(
            data_yaml=data_cfg['yaml_path'],
            split='val',
            input_size=data_cfg['input_size'],
            output_size=data_cfg['output_size'],
            batch_size=self.config['training']['batch_size'],
            num_workers=data_cfg['num_workers'],
        )

        return dataloader

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """评估模型

        Returns:
            评估指标字典
        """
        self.evaluator.reset()

        for images, targets in self.dataloader:
            images = images.to(self.device)

            # 前向传播
            outputs = self.model(images)

            if self.config['model']['head_type'] == 'paf':
                _, hm_outputs = outputs
                outputs = hm_outputs[-1]

            # 解码预测
            if self.use_dark:
                pred_kpts, scores = decode_heatmap_dark(outputs, stride=self.stride)
            else:
                pred_kpts, scores = decode_heatmap(outputs, stride=self.stride)

            # 更新评估器
            target_kpts = targets['keypoints'].numpy()
            self.evaluator.update(
                pred_kpts,
                target_kpts[:, :, :2],
                visibility=target_kpts[:, :, 2]
            )

        metrics = self.evaluator.compute()
        return metrics

    def print_results(self, metrics: Dict[str, float]):
        """打印评估结果"""
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)

        print(f"\nOverall Metrics:")
        print(f"  PCK@0.5: {metrics['PCK']:.4f}")

        if 'AP' in metrics:
            print(f"  AP: {metrics['AP']:.4f}")
            print(f"  AP50: {metrics['AP50']:.4f}")
            print(f"  AP75: {metrics['AP75']:.4f}")

        if 'PCK_per_joint' in metrics:
            print(f"\nPer-Joint PCK:")
            keypoint_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]

            for i, pck in enumerate(metrics['PCK_per_joint']):
                name = keypoint_names[i] if i < len(keypoint_names) else f'joint_{i}'
                print(f"  {name:<15}: {pck:.4f}")

        print("=" * 50)


def evaluate_checkpoint(checkpoint: str,
                       data_yaml: str,
                       backbone: str = 'tiny',
                       head_type: str = 'heatmap',
                       input_size: tuple = (640, 640),
                       output_size: tuple = (160, 160),
                       batch_size: int = 32,
                       num_workers: int = 4,
                       use_dark: bool = True) -> Dict[str, float]:
    """评估检查点

    Args:
        checkpoint: 检查点路径
        data_yaml: 数据集配置文件
        backbone: Backbone 类型
        head_type: 检测头类型
        input_size: 输入大小
        output_size: 输出大小
        batch_size: 批大小
        num_workers: 数据加载线程数
        use_dark: 是否使用 DARK 解码

    Returns:
        评估指标字典
    """
    config = {
        'data': {
            'yaml_path': data_yaml,
            'input_size': input_size,
            'output_size': output_size,
            'num_workers': num_workers,
        },
        'model': {
            'backbone': backbone,
            'head_type': head_type,
            'num_keypoints': 17,
        },
        'training': {
            'batch_size': batch_size,
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    evaluator = Evaluator(config, checkpoint, use_dark=use_dark)
    metrics = evaluator.evaluate()
    evaluator.print_results(metrics)

    return metrics
