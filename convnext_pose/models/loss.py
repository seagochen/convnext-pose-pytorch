"""
姿态检测损失函数

注意：YOLO-Pose 多人检测使用 yolo_loss.py 中的损失函数
本文件保留用于热图方式的单人姿态估计（如需要）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapLoss(nn.Module):
    """热图损失 - 使用 BCE Loss with pos_weight

    使用 BCEWithLogitsLoss 并通过 pos_weight 来平衡正负样本。
    BCEWithLogitsLoss 内部包含 sigmoid，数值稳定，
    模型输出 raw logits 即可。

    Args:
        use_target_weight: 是否使用目标权重
        pos_weight: 正样本权重 (用于平衡正负样本，热图中正样本很少)
    """
    def __init__(self, use_target_weight=True, pos_weight=25.0):
        super().__init__()
        self.use_target_weight = use_target_weight
        # pos_weight 用于给正样本更高权重
        # 热图中约 1% 是前景，所以使用 ~25 的权重
        self.register_buffer('pos_weight', torch.tensor([pos_weight]))

    def forward(self, pred, target, target_weight=None):
        """
        Args:
            pred: 预测热图 logits (B, K, H, W)
            target: 目标热图 (B, K, H, W), 值域 [0, 1]
            target_weight: 关键点权重 (B, K) 或 (B, K, 1, 1)
        Returns:
            loss: 标量损失值
        """
        # BCE Loss with pos_weight
        loss = F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=self.pos_weight.to(pred.device),
            reduction='none'
        )

        if self.use_target_weight and target_weight is not None:
            if target_weight.dim() == 2:
                target_weight = target_weight.unsqueeze(-1).unsqueeze(-1)
            loss = loss * target_weight

        # 平均损失
        loss = loss.mean()

        return loss


class AWingLoss(nn.Module):
    """Adaptive Wing Loss

    来自论文: Adaptive Wing Loss for Robust Face Alignment via
    Heatmap Regression
    """
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: 预测热图 (B, K, H, W)
            target: 目标热图 (B, K, H, W)
            weight: 权重
        """
        diff = torch.abs(pred - target)

        # Adaptive Wing Loss
        A = self.omega * (1 / (1 + (self.theta / self.epsilon) ** (self.alpha - target))) * \
            (self.alpha - target) * ((self.theta / self.epsilon) ** (self.alpha - target - 1)) / self.epsilon
        C = self.theta * A - self.omega * torch.log(1 + (self.theta / self.epsilon) ** (self.alpha - target))

        loss = torch.where(
            diff < self.theta,
            self.omega * torch.log(1 + (diff / self.epsilon) ** (self.alpha - target)),
            A * diff - C
        )

        if weight is not None:
            if weight.dim() == 2:
                weight = weight.unsqueeze(-1).unsqueeze(-1)
            loss = loss * weight

        return loss.mean()
