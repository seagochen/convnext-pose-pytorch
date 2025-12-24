"""
ConvNeXt-Pose: 基于ConvNeXt的姿态检测模型

支持三种模式:
1. heatmap: 热图回归 (类似SimpleBaseline, HRNet)
2. paf: Part Affinity Fields (类似OpenPose)
3. yolo: YOLO风格检测+关键点 (类似YOLOv8-Pose)
"""

import torch
import torch.nn as nn

from .backbone import ConvNeXt, convnext_tiny, convnext_small, convnext_base, convnext_large
from .pose_head import PoseHead, PAFHead, YOLOPoseHead


class ConvNeXtPose(nn.Module):
    """ConvNeXt-Pose 姿态检测模型

    Args:
        backbone: ConvNeXt backbone类型 ('tiny', 'small', 'base', 'large')
        num_keypoints: 关键点数量
        num_limbs: 肢体连接数量 (仅PAF模式使用)
        head_type: 检测头类型 ('heatmap', 'paf', 'yolo')
        pretrained_backbone: 是否加载预训练backbone
        out_indices: backbone输出的stage索引
    """
    def __init__(self,
                 backbone='tiny',
                 num_keypoints=17,
                 num_limbs=19,
                 head_type='heatmap',
                 pretrained_backbone=False,
                 out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.head_type = head_type

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

        # 创建检测头
        if head_type == 'heatmap':
            self.head = PoseHead(
                in_channels=in_channels,
                num_keypoints=num_keypoints,
                use_fpn=True
            )
        elif head_type == 'paf':
            self.head = PAFHead(
                in_channels=in_channels,
                num_keypoints=num_keypoints,
                num_limbs=num_limbs,
                num_stages=2,
                use_fpn=True
            )
        elif head_type == 'yolo':
            self.head = YOLOPoseHead(
                in_channels=in_channels,
                num_keypoints=num_keypoints
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}. "
                           f"Choose from ['heatmap', 'paf', 'yolo']")

    def forward(self, x):
        """
        Args:
            x: 输入图像 (B, 3, H, W)
        Returns:
            head_type='heatmap': heatmaps (B, num_keypoints, H', W')
            head_type='paf': (paf_outputs, heatmap_outputs) 列表
            head_type='yolo': 多尺度预测结果列表
        """
        features = self.backbone(x)
        outputs = self.head(features)
        return outputs

    def load_pretrained_backbone(self, checkpoint_path):
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
        num_limbs=config.get('num_limbs', 19),
        head_type=config.get('head_type', 'heatmap'),
        pretrained_backbone=config.get('pretrained_backbone', False),
        out_indices=config.get('out_indices', [0, 1, 2, 3])
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
