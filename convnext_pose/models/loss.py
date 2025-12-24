"""
姿态检测损失函数

包含多种损失函数:
1. HeatmapLoss: 热图MSE损失
2. PAFLoss: Part Affinity Fields损失
3. JointsMSELoss: 带权重的关节点MSE损失
4. OKSLoss: Object Keypoint Similarity损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HeatmapLoss(nn.Module):
    """热图MSE损失

    Args:
        use_target_weight: 是否使用目标权重
    """
    def __init__(self, use_target_weight=True):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight

    def forward(self, pred, target, target_weight=None):
        """
        Args:
            pred: 预测热图 (B, K, H, W)
            target: 目标热图 (B, K, H, W)
            target_weight: 关键点权重 (B, K) 或 (B, K, 1, 1)
        Returns:
            loss: 标量损失值
        """
        batch_size = pred.size(0)
        num_joints = pred.size(1)

        # 计算MSE
        loss = self.criterion(pred, target)

        if self.use_target_weight and target_weight is not None:
            if target_weight.dim() == 2:
                target_weight = target_weight.unsqueeze(-1).unsqueeze(-1)
            loss = loss * target_weight

        # 平均损失
        loss = loss.mean()

        return loss


class JointsMSELoss(nn.Module):
    """带权重的关节点MSE损失

    支持对不同关键点设置不同权重
    """
    def __init__(self, use_target_weight=True, joint_weights=None):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight

        # 关节点权重 (例如: 躯干关键点权重更高)
        if joint_weights is not None:
            self.register_buffer('joint_weights',
                               torch.tensor(joint_weights, dtype=torch.float32))
        else:
            self.joint_weights = None

    def forward(self, pred, target, target_weight=None):
        """
        Args:
            pred: 预测热图 (B, K, H, W)
            target: 目标热图 (B, K, H, W)
            target_weight: 可见性权重 (B, K)
        """
        batch_size = pred.size(0)
        num_joints = pred.size(1)
        height = pred.size(2)
        width = pred.size(3)

        # reshape为 (B, K, H*W)
        pred = pred.view(batch_size, num_joints, -1)
        target = target.view(batch_size, num_joints, -1)

        # MSE损失
        loss = self.criterion(pred, target)  # (B, K, H*W)

        # 在空间维度求和
        loss = loss.sum(dim=2)  # (B, K)

        # 应用可见性权重
        if self.use_target_weight and target_weight is not None:
            loss = loss * target_weight

        # 应用关节点权重
        if self.joint_weights is not None:
            loss = loss * self.joint_weights.unsqueeze(0)

        # 归一化
        if self.use_target_weight and target_weight is not None:
            num_valid = target_weight.sum()
            if num_valid > 0:
                loss = loss.sum() / num_valid
            else:
                loss = loss.sum() * 0
        else:
            loss = loss.mean()

        return loss


class PAFLoss(nn.Module):
    """Part Affinity Fields损失

    用于OpenPose风格的多stage训练
    """
    def __init__(self, num_stages=2, stage_weights=None):
        super().__init__()
        self.num_stages = num_stages
        self.criterion = nn.MSELoss(reduction='mean')

        # stage权重 (后面的stage权重更高)
        if stage_weights is None:
            stage_weights = [1.0] * num_stages
        self.stage_weights = stage_weights

    def forward(self, paf_preds, heatmap_preds, paf_targets, heatmap_targets, masks=None):
        """
        Args:
            paf_preds: list of PAF预测 [(B, L*2, H, W), ...]
            heatmap_preds: list of 热图预测 [(B, K, H, W), ...]
            paf_targets: PAF目标 (B, L*2, H, W)
            heatmap_targets: 热图目标 (B, K, H, W)
            masks: 有效区域掩码 (B, 1, H, W)
        """
        total_loss = 0
        paf_loss = 0
        heatmap_loss = 0

        for stage_idx in range(self.num_stages):
            weight = self.stage_weights[stage_idx]

            # PAF损失
            if masks is not None:
                paf_pred_masked = paf_preds[stage_idx] * masks
                paf_target_masked = paf_targets * masks
                stage_paf_loss = self.criterion(paf_pred_masked, paf_target_masked)
            else:
                stage_paf_loss = self.criterion(paf_preds[stage_idx], paf_targets)

            # 热图损失
            if masks is not None:
                hm_pred_masked = heatmap_preds[stage_idx] * masks
                hm_target_masked = heatmap_targets * masks
                stage_hm_loss = self.criterion(hm_pred_masked, hm_target_masked)
            else:
                stage_hm_loss = self.criterion(heatmap_preds[stage_idx], heatmap_targets)

            paf_loss += weight * stage_paf_loss
            heatmap_loss += weight * stage_hm_loss
            total_loss += weight * (stage_paf_loss + stage_hm_loss)

        return {
            'total_loss': total_loss,
            'paf_loss': paf_loss,
            'heatmap_loss': heatmap_loss
        }


class OKSLoss(nn.Module):
    """Object Keypoint Similarity (OKS) 损失

    基于COCO评估指标的损失函数
    """

    # COCO关键点的sigma值
    COCO_SIGMAS = np.array([
        0.026, 0.025, 0.025, 0.035, 0.035,
        0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087,
        0.089, 0.089
    ])

    def __init__(self, num_keypoints=17, sigmas=None):
        super().__init__()
        self.num_keypoints = num_keypoints

        if sigmas is None:
            sigmas = self.COCO_SIGMAS[:num_keypoints]
        self.register_buffer('sigmas', torch.tensor(sigmas, dtype=torch.float32))

    def forward(self, pred, target, area, visibility):
        """
        Args:
            pred: 预测关键点 (B, K, 2)
            target: 目标关键点 (B, K, 2)
            area: 目标区域面积 (B,)
            visibility: 关键点可见性 (B, K)
        Returns:
            loss: 1 - OKS
        """
        # 计算欧氏距离
        d = torch.sqrt(((pred - target) ** 2).sum(dim=-1))  # (B, K)

        # 归一化
        s = area.unsqueeze(1)  # (B, 1)
        kappa = 2 * self.sigmas ** 2  # (K,)

        # OKS for each keypoint
        oks = torch.exp(-d ** 2 / (2 * s * kappa.unsqueeze(0)))  # (B, K)

        # 只考虑可见关键点
        oks = oks * visibility

        # 平均OKS
        num_visible = visibility.sum(dim=1, keepdim=True).clamp(min=1)
        oks = oks.sum(dim=1) / num_visible.squeeze(1)

        # 损失 = 1 - OKS
        loss = 1 - oks.mean()

        return loss


class WingLoss(nn.Module):
    """Wing Loss for keypoint regression

    来自论文: Wing Loss for Robust Facial Landmark Localisation with
    Convolutional Neural Networks
    """
    def __init__(self, omega=10, epsilon=2):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * np.log(1 + self.omega / self.epsilon)

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: 预测值 (B, K, 2) 或 (B, K)
            target: 目标值
            weight: 权重
        """
        diff = torch.abs(pred - target)

        # Wing loss公式
        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )

        if weight is not None:
            loss = loss * weight.unsqueeze(-1)

        return loss.mean()


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


class CombinedLoss(nn.Module):
    """组合损失函数

    可以组合多个损失函数，并设置权重
    """
    def __init__(self, losses_config):
        """
        Args:
            losses_config: list of dicts, each with 'type', 'weight', and optional params
                Example: [
                    {'type': 'mse', 'weight': 1.0},
                    {'type': 'oks', 'weight': 0.5}
                ]
        """
        super().__init__()

        self.losses = nn.ModuleList()
        self.weights = []
        self.names = []

        loss_classes = {
            'mse': HeatmapLoss,
            'joints_mse': JointsMSELoss,
            'paf': PAFLoss,
            'oks': OKSLoss,
            'wing': WingLoss,
            'awing': AWingLoss,
        }

        for config in losses_config:
            loss_type = config['type']
            weight = config.get('weight', 1.0)
            params = {k: v for k, v in config.items() if k not in ['type', 'weight']}

            if loss_type not in loss_classes:
                raise ValueError(f"Unknown loss type: {loss_type}")

            self.losses.append(loss_classes[loss_type](**params))
            self.weights.append(weight)
            self.names.append(loss_type)

    def forward(self, pred, target, **kwargs):
        """
        Returns:
            total_loss: 加权总损失
            loss_dict: 各个损失的字典
        """
        total_loss = 0
        loss_dict = {}

        for loss_fn, weight, name in zip(self.losses, self.weights, self.names):
            loss = loss_fn(pred, target, **kwargs)
            total_loss += weight * loss
            loss_dict[name] = loss.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


def build_loss(loss_type='mse', **kwargs):
    """根据类型构建损失函数

    Args:
        loss_type: 损失函数类型
        **kwargs: 损失函数参数
    """
    if loss_type == 'mse':
        return HeatmapLoss(use_target_weight=kwargs.get('use_target_weight', True))
    elif loss_type == 'joints_mse':
        return JointsMSELoss(
            use_target_weight=kwargs.get('use_target_weight', True),
            joint_weights=kwargs.get('joint_weights', None)
        )
    elif loss_type == 'paf':
        return PAFLoss(
            num_stages=kwargs.get('num_stages', 2),
            stage_weights=kwargs.get('stage_weights', None)
        )
    elif loss_type == 'oks':
        return OKSLoss(
            num_keypoints=kwargs.get('num_keypoints', 17),
            sigmas=kwargs.get('sigmas', None)
        )
    elif loss_type == 'wing':
        return WingLoss(
            omega=kwargs.get('omega', 10),
            epsilon=kwargs.get('epsilon', 2)
        )
    elif loss_type == 'awing':
        return AWingLoss(
            omega=kwargs.get('omega', 14),
            theta=kwargs.get('theta', 0.5),
            epsilon=kwargs.get('epsilon', 1),
            alpha=kwargs.get('alpha', 2.1)
        )
    elif loss_type == 'combined':
        return CombinedLoss(kwargs.get('losses', [{'type': 'mse', 'weight': 1.0}]))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
