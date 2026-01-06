"""
ConvNeXt-Pose: 基于ConvNeXt的端到端多人姿态检测模型

使用 YOLO-Pose 风格的检测头，支持:
- 多尺度目标检测 (bbox)
- 关键点预测 (相对于bbox中心的偏移)
- Anchor-free 设计 (类似 YOLOX/YOLOv8)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from .backbone import ConvNeXt, convnext_tiny, convnext_small, convnext_base, convnext_large
from .pose_head import YOLOPoseHead, build_pose_head


class ConvNeXtPose(nn.Module):
    """ConvNeXt-Pose 端到端多人姿态检测模型

    Args:
        backbone: ConvNeXt backbone类型 ('tiny', 'small', 'base', 'large')
        num_keypoints: 关键点数量
        fpn_channels: FPN 输出通道数
        use_fpn: 是否使用 FPN
        pretrained_backbone: 是否加载预训练backbone
        out_indices: backbone输出的stage索引
        strides: 各尺度的下采样步长
    """
    def __init__(self,
                 backbone: str = 'tiny',
                 num_keypoints: int = 17,
                 fpn_channels: int = 256,
                 use_fpn: bool = True,
                 pretrained_backbone: bool = False,
                 out_indices: List[int] = [0, 1, 2, 3],
                 strides: List[int] = None):
        super().__init__()

        self.num_keypoints = num_keypoints

        # 创建backbone
        backbone_fn = {
            'tiny': convnext_tiny,
            'small': convnext_small,
            'base': convnext_base,
            'large': convnext_large
        }

        if backbone not in backbone_fn:
            raise ValueError(f"Unknown backbone: {backbone}. "
                           f"Choose from {list(backbone_fn.keys())}")

        self.backbone = backbone_fn[backbone](
            pretrained=pretrained_backbone,
            num_classes=0,  # 不使用分类头
            out_indices=out_indices
        )

        # 获取backbone输出通道数
        in_channels = self.backbone.out_channels

        # 默认步长
        if strides is None:
            strides = [8, 16, 32, 64][:len(in_channels)]

        # 创建 YOLO-Pose 检测头
        self.head = YOLOPoseHead(
            in_channels=in_channels,
            num_keypoints=num_keypoints,
            fpn_channels=fpn_channels,
            use_fpn=use_fpn,
            strides=strides
        )

        self.strides = strides

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """前向传播

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            outputs: 多尺度预测结果列表
                每个元素形状: (B, H, W, num_outputs)
                num_outputs = 4 (box) + 1 (obj) + num_keypoints * 3 (kpts)
        """
        features = self.backbone(x)
        outputs = self.head(features)
        return outputs

    def decode(self,
               outputs: List[torch.Tensor],
               conf_thresh: float = 0.25,
               input_size: Tuple[int, int] = (640, 640)):
        """解码网络输出为检测结果

        Args:
            outputs: forward() 的输出
            conf_thresh: 置信度阈值
            input_size: 输入图像大小 (H, W)

        Returns:
            bboxes: (N, 4) [x1, y1, x2, y2] 像素坐标
            scores: (N,) 置信度分数
            keypoints: (N, num_keypoints, 3) [x, y, conf] 像素坐标
        """
        return self.head.decode(outputs, conf_thresh, input_size)

    def load_pretrained_backbone(self, checkpoint_path: str):
        """加载预训练backbone权重"""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']

        # 过滤掉分类头的权重
        backbone_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('head.'):
                continue
            backbone_state_dict[k] = v

        self.backbone.load_state_dict(backbone_state_dict, strict=False)
        print(f"Loaded pretrained backbone from {checkpoint_path}")

    def freeze_backbone(self):
        """冻结backbone参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")

    def unfreeze_backbone(self):
        """解冻backbone参数"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")


def create_model(config):
    """根据配置创建模型

    Args:
        config: 配置字典或对象

    Returns:
        ConvNeXtPose模型
    """
    if hasattr(config, '__dict__'):
        config = vars(config)

    model = ConvNeXtPose(
        backbone=config.get('backbone', 'tiny'),
        num_keypoints=config.get('num_keypoints', 17),
        fpn_channels=config.get('fpn_channels', 256),
        use_fpn=config.get('use_fpn', True),
        pretrained_backbone=config.get('pretrained_backbone', False),
        out_indices=config.get('out_indices', [0, 1, 2, 3]),
        strides=config.get('strides', None)
    )

    # 加载预训练backbone
    if config.get('backbone_checkpoint'):
        model.load_pretrained_backbone(config['backbone_checkpoint'])

    return model


# COCO关键点定义
COCO_KEYPOINTS = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle',    # 16
]

# COCO关键点翻转映射
COCO_FLIP_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# COCO肢体连接 (用于PAF和可视化)
COCO_LIMBS = [
    (0, 1),   # nose -> left_eye
    (0, 2),   # nose -> right_eye
    (1, 3),   # left_eye -> left_ear
    (2, 4),   # right_eye -> right_ear
    (0, 5),   # nose -> left_shoulder
    (0, 6),   # nose -> right_shoulder
    (5, 7),   # left_shoulder -> left_elbow
    (7, 9),   # left_elbow -> left_wrist
    (6, 8),   # right_shoulder -> right_elbow
    (8, 10),  # right_elbow -> right_wrist
    (5, 11),  # left_shoulder -> left_hip
    (6, 12),  # right_shoulder -> right_hip
    (11, 13), # left_hip -> left_knee
    (13, 15), # left_knee -> left_ankle
    (12, 14), # right_hip -> right_knee
    (14, 16), # right_knee -> right_ankle
    (5, 6),   # left_shoulder -> right_shoulder
    (11, 12), # left_hip -> right_hip
]
