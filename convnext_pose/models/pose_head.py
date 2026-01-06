"""
YOLO-Pose Detection Head

端到端多人姿态估计检测头，支持:
- 多尺度目标检测 (bbox)
- 关键点预测 (相对于bbox中心的偏移)
- Anchor-free 设计 (类似 YOLOX/YOLOv8)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvBNSiLU(nn.Module):
    """卷积 + BatchNorm + SiLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


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
            list of FPN outputs at each scale
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

        return outs


class YOLOPoseHead(nn.Module):
    """YOLO-Pose 端到端多人姿态估计头

    Anchor-free 设计，同时预测:
    - 边界框: (cx, cy, w, h) 相对于 grid cell
    - 目标置信度: objectness
    - 关键点: (dx, dy, visibility) × num_keypoints，相对于 bbox 中心

    Args:
        in_channels: 输入通道数列表 (多尺度)
        num_keypoints: 关键点数量
        fpn_channels: FPN 输出通道数
        use_fpn: 是否使用 FPN 融合
        strides: 每个尺度的步长
    """

    def __init__(self,
                 in_channels: List[int],
                 num_keypoints: int = 17,
                 fpn_channels: int = 256,
                 use_fpn: bool = True,
                 strides: List[int] = None):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.use_fpn = use_fpn

        # 预测通道数: box(4) + obj(1) + kpts(num_keypoints * 3)
        self.num_outputs = 4 + 1 + num_keypoints * 3

        # 默认步长
        if strides is None:
            strides = [8, 16, 32, 64][:len(in_channels)]
        self.strides = strides
        self.num_scales = len(in_channels)

        # FPN 特征融合
        if use_fpn:
            self.fpn = FPN(in_channels, out_channels=fpn_channels)
            head_in_channels = [fpn_channels] * len(in_channels)
        else:
            self.fpn = None
            head_in_channels = in_channels

        # 每个尺度的检测头
        self.detect_heads = nn.ModuleList()
        for in_ch in head_in_channels:
            head = nn.Sequential(
                ConvBNSiLU(in_ch, in_ch, 3, 1, 1),
                ConvBNSiLU(in_ch, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, self.num_outputs, kernel_size=1)
            )
            self.detect_heads.append(head)

        # 初始化权重
        self._init_weights()

        # 缓存的 grid (用于解码)
        self._cached_grids = {}

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 对输出层的 objectness 使用较小的初始偏置 (防止初始输出过大)
        for head in self.detect_heads:
            # 最后一层是 Conv2d
            final_conv = head[-1]
            if final_conv.bias is not None:
                # objectness 偏置初始化为 -4.0，使初始概率接近 0.02
                final_conv.bias.data[4] = -4.0

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: 多尺度特征图列表 [(B, C1, H1, W1), (B, C2, H2, W2), ...]

        Returns:
            outputs: 每个尺度的预测结果列表
                每个元素形状: (B, H, W, num_outputs)
                num_outputs = 4 (box) + 1 (obj) + num_keypoints * 3 (kpts)
        """
        # FPN 融合
        if self.fpn is not None:
            features = self.fpn(features)

        outputs = []
        for i, (feat, head) in enumerate(zip(features, self.detect_heads)):
            out = head(feat)  # (B, num_outputs, H, W)
            B, C, H, W = out.shape
            # 重塑为 (B, H, W, C)
            out = out.permute(0, 2, 3, 1).contiguous()
            outputs.append(out)

        return outputs

    def decode(self,
               outputs: List[torch.Tensor],
               conf_thresh: float = 0.25,
               input_size: Tuple[int, int] = (640, 640)) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        device = outputs[0].device
        batch_size = outputs[0].shape[0]

        all_bboxes = []
        all_scores = []
        all_keypoints = []
        all_batch_idx = []

        for scale_idx, (output, stride) in enumerate(zip(outputs, self.strides)):
            B, H, W, C = output.shape

            # 获取或创建 grid
            grid = self._get_grid(H, W, stride, device)

            # 解析预测
            # box: (B, H, W, 4) -> (cx, cy, w, h)
            box_pred = output[..., :4]
            # obj: (B, H, W, 1)
            obj_pred = output[..., 4:5]
            # kpts: (B, H, W, num_keypoints * 3)
            kpts_pred = output[..., 5:]

            # 解码边界框
            # cx, cy 相对于 grid cell，w, h 是相对于图像的比例
            cx = (grid[..., 0:1] + box_pred[..., 0:1].sigmoid()) * stride
            cy = (grid[..., 1:2] + box_pred[..., 1:2].sigmoid()) * stride
            # 限制 exp 的输入范围，防止数值爆炸
            w = box_pred[..., 2:3].clamp(-10, 10).exp() * stride
            h = box_pred[..., 3:4].clamp(-10, 10).exp() * stride

            # 转换为 [x1, y1, x2, y2]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            bboxes = torch.cat([x1, y1, x2, y2], dim=-1)  # (B, H, W, 4)

            # 置信度
            scores = obj_pred.sigmoid().squeeze(-1)  # (B, H, W)

            # 解码关键点
            kpts_pred = kpts_pred.view(B, H, W, self.num_keypoints, 3)
            # dx, dy 是相对于 bbox 中心的偏移 (已经归一化到 bbox 尺寸)
            # 转换为绝对像素坐标
            kpts_x = cx + kpts_pred[..., 0] * w
            kpts_y = cy + kpts_pred[..., 1] * h
            kpts_conf = kpts_pred[..., 2].sigmoid()
            keypoints = torch.stack([kpts_x.squeeze(-1), kpts_y.squeeze(-1), kpts_conf], dim=-1)

            # 过滤低置信度预测
            for b in range(B):
                mask = scores[b] > conf_thresh  # (H, W)
                if mask.sum() > 0:
                    all_bboxes.append(bboxes[b][mask])
                    all_scores.append(scores[b][mask])
                    all_keypoints.append(keypoints[b][mask])
                    all_batch_idx.append(torch.full((mask.sum(),), b, device=device))

        if len(all_bboxes) == 0:
            return (
                torch.zeros((0, 4), device=device),
                torch.zeros((0,), device=device),
                torch.zeros((0, self.num_keypoints, 3), device=device)
            )

        bboxes = torch.cat(all_bboxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        keypoints = torch.cat(all_keypoints, dim=0)

        return bboxes, scores, keypoints

    def _get_grid(self, H: int, W: int, stride: int, device: torch.device) -> torch.Tensor:
        """获取或创建 grid 坐标

        Args:
            H, W: 特征图大小
            stride: 下采样步长
            device: 设备

        Returns:
            grid: (1, H, W, 2) grid cell 的中心坐标 (未乘以 stride)
        """
        key = (H, W, stride)
        if key not in self._cached_grids:
            yv, xv = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            grid = torch.stack([xv, yv], dim=-1).float()  # (H, W, 2)
            grid = grid.unsqueeze(0)  # (1, H, W, 2)
            self._cached_grids[key] = grid

        return self._cached_grids[key]

    def get_targets_for_scale(self,
                               bboxes: torch.Tensor,
                               keypoints: torch.Tensor,
                               num_persons: torch.Tensor,
                               scale_idx: int,
                               feat_size: Tuple[int, int],
                               input_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """为指定尺度生成训练目标

        Args:
            bboxes: (B, max_persons, 4) [cx, cy, w, h] 归一化
            keypoints: (B, max_persons, num_keypoints, 3) [dx, dy, v] 相对偏移
            num_persons: (B,) 每张图的实际人数
            scale_idx: 尺度索引
            feat_size: 特征图大小 (H, W)
            input_size: 输入图像大小 (H, W)

        Returns:
            target_obj: (B, H, W) 目标置信度
            target_box: (B, H, W, 4) 目标边界框
            target_kpts: (B, H, W, num_keypoints, 3) 目标关键点
        """
        device = bboxes.device
        B = bboxes.shape[0]
        H, W = feat_size
        stride = self.strides[scale_idx]
        input_h, input_w = input_size

        # 初始化目标
        target_obj = torch.zeros((B, H, W), device=device)
        target_box = torch.zeros((B, H, W, 4), device=device)
        target_kpts = torch.zeros((B, H, W, self.num_keypoints, 3), device=device)

        for b in range(B):
            n_persons = num_persons[b].item()
            for p in range(n_persons):
                # 获取 bbox (归一化坐标)
                cx, cy, w, h = bboxes[b, p]

                # 跳过无效的 bbox
                if w <= 0 or h <= 0:
                    continue

                # 转换为特征图坐标
                fx = cx * input_w / stride
                fy = cy * input_h / stride

                # 找到最近的 grid cell
                gi = int(fx.clamp(0, W - 1))
                gj = int(fy.clamp(0, H - 1))

                # 设置目标
                target_obj[b, gj, gi] = 1.0

                # 边界框目标: 相对于 grid cell 的偏移
                target_box[b, gj, gi, 0] = fx - gi  # cx 偏移
                target_box[b, gj, gi, 1] = fy - gj  # cy 偏移
                target_box[b, gj, gi, 2] = torch.log(w * input_w / stride + 1e-6)  # log(w)
                target_box[b, gj, gi, 3] = torch.log(h * input_h / stride + 1e-6)  # log(h)

                # 关键点目标 (直接使用编码后的相对偏移)
                target_kpts[b, gj, gi] = keypoints[b, p]

        return target_obj, target_box, target_kpts


def build_pose_head(in_channels: List[int],
                    num_keypoints: int = 17,
                    fpn_channels: int = 256,
                    use_fpn: bool = True) -> YOLOPoseHead:
    """构建姿态检测头

    Args:
        in_channels: backbone 输出通道数列表
        num_keypoints: 关键点数量
        fpn_channels: FPN 通道数
        use_fpn: 是否使用 FPN

    Returns:
        YOLOPoseHead 实例
    """
    return YOLOPoseHead(
        in_channels=in_channels,
        num_keypoints=num_keypoints,
        fpn_channels=fpn_channels,
        use_fpn=use_fpn
    )
