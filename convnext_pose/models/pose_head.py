"""
Pose Detection Head

包含三种检测头:
1. PoseHead: 直接预测关键点热图 (类似SimpleBaseline, HRNet)
2. PAFHead: 预测热图 + Part Affinity Fields (类似OpenPose)
3. YOLOPoseHead: YOLO风格检测+关键点 (类似YOLOv8-Pose)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """卷积 + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    """反卷积上采样块"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

        self.out_channels = out_channels

    def forward(self, features):
        """
        Args:
            features: list of feature maps from backbone [C1, C2, C3, C4]
        Returns:
            fused feature map
        """
        # 自顶向下
        laterals = [lateral_conv(features[i])
                   for i, lateral_conv in enumerate(self.lateral_convs)]

        # 特征融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='bilinear', align_corners=False
            )

        # 输出卷积
        outs = [fpn_conv(laterals[i]) for i, fpn_conv in enumerate(self.fpn_convs)]

        # 返回最高分辨率特征
        return outs[0]


class PoseHead(nn.Module):
    """简单的姿态检测头 - 预测关键点热图

    类似SimpleBaseline的设计，使用反卷积进行上采样

    Args:
        in_channels: 输入通道数或通道数列表
        num_keypoints: 关键点数量 (COCO=17, MPII=16)
        num_deconv_layers: 反卷积层数量
        num_deconv_filters: 每个反卷积层的输出通道数
        use_fpn: 是否使用FPN融合多尺度特征
        output_stride: 输出相对于输入的步长 (默认4，即输出为输入的1/4)
    """
    def __init__(self, in_channels, num_keypoints=17,
                 num_deconv_layers=3, num_deconv_filters=256,
                 use_fpn=True, output_stride=4):
        super().__init__()
        self.use_fpn = use_fpn
        self.output_stride = output_stride

        if use_fpn and isinstance(in_channels, list):
            self.fpn = FPN(in_channels, out_channels=num_deconv_filters)
            in_ch = num_deconv_filters
            # FPN输出已经是1/4分辨率，不需要太多上采样
            # 只需要保持分辨率的卷积层
            num_deconv_layers = 0
        else:
            self.fpn = None
            in_ch = in_channels[-1] if isinstance(in_channels, list) else in_channels

        # 反卷积上采样层 (从1/32上采样到1/4需要3次2x上采样)
        deconv_layers = []
        for i in range(num_deconv_layers):
            out_ch = num_deconv_filters
            deconv_layers.append(DeconvBlock(in_ch, out_ch))
            in_ch = out_ch

        if deconv_layers:
            self.deconv_layers = nn.Sequential(*deconv_layers)
        else:
            # 如果不需要上采样，使用卷积层进行特征细化
            self.deconv_layers = nn.Sequential(
                ConvBNReLU(in_ch, num_deconv_filters, 3, 1, 1),
                ConvBNReLU(num_deconv_filters, num_deconv_filters, 3, 1, 1),
            )
            in_ch = num_deconv_filters

        # 最终的热图预测层
        self.final_layer = nn.Conv2d(num_deconv_filters, num_keypoints, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        Args:
            features: backbone输出的特征图列表或单个特征图
        Returns:
            heatmaps: (B, num_keypoints, H, W) 关键点热图
        """
        if self.fpn is not None:
            x = self.fpn(features)
        else:
            x = features[-1] if isinstance(features, list) else features

        x = self.deconv_layers(x)
        heatmaps = self.final_layer(x)

        return heatmaps


class PAFHead(nn.Module):
    """Part Affinity Fields检测头 - 类似OpenPose

    同时预测:
    1. 关键点热图 (Confidence Maps)
    2. Part Affinity Fields (肢体方向场)

    Args:
        in_channels: 输入通道数或通道数列表
        num_keypoints: 关键点数量
        num_limbs: 肢体连接数量 (PAF通道数 = num_limbs * 2)
        num_stages: 迭代优化的stage数量
        use_fpn: 是否使用FPN
    """
    def __init__(self, in_channels, num_keypoints=17, num_limbs=19,
                 num_stages=2, use_fpn=True):
        super().__init__()
        self.num_stages = num_stages
        self.num_keypoints = num_keypoints
        self.num_limbs = num_limbs
        self.use_fpn = use_fpn

        # 特征融合
        if use_fpn and isinstance(in_channels, list):
            self.fpn = FPN(in_channels, out_channels=256)
            feature_channels = 256
        else:
            self.fpn = None
            feature_channels = in_channels[-1] if isinstance(in_channels, list) else in_channels

        # 初始特征提取
        self.initial_conv = nn.Sequential(
            ConvBNReLU(feature_channels, 256, 3, 1, 1),
            ConvBNReLU(256, 128, 3, 1, 1),
        )

        # Stage模块
        self.paf_stages = nn.ModuleList()
        self.heatmap_stages = nn.ModuleList()

        for stage_idx in range(num_stages):
            if stage_idx == 0:
                in_ch = 128
            else:
                in_ch = 128 + num_limbs * 2 + num_keypoints

            # PAF分支
            paf_stage = self._make_stage(in_ch, num_limbs * 2)
            self.paf_stages.append(paf_stage)

            # Heatmap分支
            heatmap_stage = self._make_stage(in_ch, num_keypoints)
            self.heatmap_stages.append(heatmap_stage)

        self._init_weights()

    def _make_stage(self, in_channels, out_channels):
        """创建一个stage模块"""
        return nn.Sequential(
            ConvBNReLU(in_channels, 128, 3, 1, 1),
            ConvBNReLU(128, 128, 3, 1, 1),
            ConvBNReLU(128, 128, 3, 1, 1),
            ConvBNReLU(128, 128, 3, 1, 1),
            ConvBNReLU(128, 128, 3, 1, 1),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        Args:
            features: backbone输出的特征图列表或单个特征图
        Returns:
            paf_outputs: list of PAF predictions from each stage
            heatmap_outputs: list of heatmap predictions from each stage
        """
        if self.fpn is not None:
            x = self.fpn(features)
        else:
            x = features[-1] if isinstance(features, list) else features

        # 上采样到更高分辨率
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # 初始特征
        feat = self.initial_conv(x)

        paf_outputs = []
        heatmap_outputs = []

        for stage_idx in range(self.num_stages):
            if stage_idx == 0:
                stage_input = feat
            else:
                # 拼接前一stage的输出
                stage_input = torch.cat([feat, paf_outputs[-1], heatmap_outputs[-1]], dim=1)

            paf = self.paf_stages[stage_idx](stage_input)
            heatmap = self.heatmap_stages[stage_idx](stage_input)

            paf_outputs.append(paf)
            heatmap_outputs.append(heatmap)

        return paf_outputs, heatmap_outputs


class YOLOPoseHead(nn.Module):
    """YOLO风格的姿态检测头 - 类似YOLOv8-Pose

    在检测框的基础上预测关键点坐标

    Args:
        in_channels: 输入通道数列表
        num_keypoints: 关键点数量
        num_classes: 目标类别数 (通常为1表示person)
    """
    def __init__(self, in_channels, num_keypoints=17, num_classes=1):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes

        # 每个尺度的检测头
        self.detect_heads = nn.ModuleList()

        for in_ch in in_channels:
            head = nn.Sequential(
                ConvBNReLU(in_ch, in_ch, 3, 1, 1),
                ConvBNReLU(in_ch, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch,
                         # box(4) + obj(1) + cls + kpts(num_keypoints * 3)
                         4 + 1 + num_classes + num_keypoints * 3,
                         kernel_size=1)
            )
            self.detect_heads.append(head)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        Args:
            features: 多尺度特征图列表
        Returns:
            outputs: 每个尺度的预测结果列表
                每个元素形状: (B, num_anchors, H, W, 4+1+num_classes+num_keypoints*3)
        """
        outputs = []
        for feat, head in zip(features, self.detect_heads):
            out = head(feat)
            B, C, H, W = out.shape
            # 重塑为 (B, H, W, C)
            out = out.permute(0, 2, 3, 1).contiguous()
            outputs.append(out)
        return outputs
