"""
YOLO-Pose 多人姿态估计损失函数

包含:
- CIoU Loss: 边界框回归损失
- BCEWithLogits: 目标置信度损失 (Focal Loss)
- KeypointLoss: 关键点回归损失 (SmoothL1 + OKS)
- TaskAlignedAssigner: 正负样本动态分配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math


class CIoULoss(nn.Module):
    """Complete IoU Loss

    CIoU = IoU - (ρ²(b, b_gt) / c²) - αv

    其中:
    - ρ²: 预测框和真实框中心点的欧氏距离
    - c²: 最小闭包区域的对角线距离
    - α: 权重系数
    - v: 长宽比一致性
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测框 (N, 4) [x1, y1, x2, y2]
            target: 目标框 (N, 4) [x1, y1, x2, y2]

        Returns:
            loss: CIoU 损失
        """
        # 强制使用 float32 计算，避免 FP16 数值不稳定
        pred = pred.float()
        target = target.float()

        # 计算交集
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # 计算并集 - 确保宽高为正
        pred_w = (pred[:, 2] - pred[:, 0]).clamp(min=self.eps)
        pred_h = (pred[:, 3] - pred[:, 1]).clamp(min=self.eps)
        target_w = (target[:, 2] - target[:, 0]).clamp(min=self.eps)
        target_h = (target[:, 3] - target[:, 1]).clamp(min=self.eps)

        pred_area = pred_w * pred_h
        target_area = target_w * target_h
        union_area = pred_area + target_area - inter_area + self.eps

        # IoU
        iou = inter_area / union_area

        # 中心点距离
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2

        rho2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

        # 最小闭包区域的对角线距离
        enclose_x1 = torch.min(pred[:, 0], target[:, 0])
        enclose_y1 = torch.min(pred[:, 1], target[:, 1])
        enclose_x2 = torch.max(pred[:, 2], target[:, 2])
        enclose_y2 = torch.max(pred[:, 3], target[:, 3])

        c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + self.eps

        # 长宽比一致性 - 使用 clamp 后的宽高，避免除零
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(target_w / target_h) -
            torch.atan(pred_w / pred_h), 2
        )

        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)

        # CIoU
        ciou = iou - rho2 / c2 - alpha * v

        # 确保损失在合理范围内
        loss = 1 - ciou
        loss = loss.clamp(min=0, max=4.0)  # 限制最大损失

        return loss


class FocalLoss(nn.Module):
    """Focal Loss for objectness

    FL(p) = -α(1-p)^γ * log(p)

    用于解决正负样本不平衡问题

    注意：alpha 是正样本的权重。对于 YOLO 这种正负样本比例悬殊的任务，
    需要给正样本更大的权重 (alpha > 0.5)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测 logits (N,)
            target: 目标 (N,) 0 or 1

        Returns:
            loss: Focal Loss
        """
        # 数值稳定的 sigmoid
        pred_prob = torch.sigmoid(pred.float()).clamp(self.eps, 1 - self.eps)

        # 计算 focal 权重
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # BCE loss - 使用 float32 确保数值稳定
        bce = F.binary_cross_entropy_with_logits(
            pred.float(), target.float(), reduction='none'
        )

        loss = (focal_weight * bce).mean()

        # 防止 NaN - 使用 pred.sum() * 0 保持计算图连接
        if torch.isnan(loss) or torch.isinf(loss):
            return pred.sum() * 0

        return loss


class KeypointLoss(nn.Module):
    """关键点损失

    组合 SmoothL1 (坐标) + BCE (可见性) + OKS (可选)
    """

    # COCO关键点的sigma值
    COCO_SIGMAS = torch.tensor([
        0.026, 0.025, 0.025, 0.035, 0.035,
        0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087,
        0.089, 0.089
    ])

    def __init__(self,
                 num_keypoints: int = 17,
                 use_oks: bool = True,
                 coord_weight: float = 1.0,
                 vis_weight: float = 1.0,
                 oks_weight: float = 0.5):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.use_oks = use_oks
        self.coord_weight = coord_weight
        self.vis_weight = vis_weight
        self.oks_weight = oks_weight

        # 注册 sigmas
        sigmas = self.COCO_SIGMAS[:num_keypoints]
        self.register_buffer('sigmas', sigmas)

    def forward(self,
                pred_kpts: torch.Tensor,
                target_kpts: torch.Tensor,
                target_vis: torch.Tensor,
                bbox_area: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_kpts: 预测关键点 (N, K, 2) [dx, dy]
            target_kpts: 目标关键点 (N, K, 2) [dx, dy]
            target_vis: 目标可见性 (N, K)
            bbox_area: bbox 面积 (N,) 用于 OKS

        Returns:
            loss_dict: 包含各部分损失
        """
        loss_dict = {}

        # 只计算可见关键点的损失
        vis_mask = target_vis > 0  # (N, K)

        if vis_mask.sum() == 0:
            # 没有可见关键点
            zero = pred_kpts.sum() * 0
            loss_dict['coord_loss'] = zero
            loss_dict['oks_loss'] = zero
            loss_dict['kpt_loss'] = zero
            return loss_dict

        # 坐标损失 (SmoothL1) - 使用 float32 确保数值稳定
        coord_loss = F.smooth_l1_loss(
            pred_kpts[vis_mask].float(),
            target_kpts[vis_mask].float(),
            reduction='mean',
            beta=0.1
        )
        loss_dict['coord_loss'] = coord_loss * self.coord_weight

        # OKS 损失
        if self.use_oks and bbox_area is not None:
            oks_loss = self._compute_oks_loss(
                pred_kpts, target_kpts, target_vis, bbox_area
            )
            loss_dict['oks_loss'] = oks_loss * self.oks_weight
        else:
            loss_dict['oks_loss'] = coord_loss * 0

        # 总损失
        loss_dict['kpt_loss'] = loss_dict['coord_loss'] + loss_dict['oks_loss']

        return loss_dict

    def _compute_oks_loss(self,
                          pred_kpts: torch.Tensor,
                          target_kpts: torch.Tensor,
                          target_vis: torch.Tensor,
                          bbox_area: torch.Tensor) -> torch.Tensor:
        """计算 OKS 损失

        OKS = Σ exp(-d²/2s²σ²) * δ(v>0) / Σ δ(v>0)
        """
        # 强制使用 float32 计算
        pred_kpts = pred_kpts.float()
        target_kpts = target_kpts.float()
        bbox_area = bbox_area.float()

        # 计算欧氏距离的平方
        d2 = ((pred_kpts - target_kpts) ** 2).sum(dim=-1)  # (N, K)

        # 归一化因子
        s2 = bbox_area.unsqueeze(1).clamp(min=1.0)  # (N, 1) 确保面积不为0
        kappa = 2 * (self.sigmas ** 2).unsqueeze(0)  # (1, K)

        # OKS for each keypoint - 限制 exp 输入范围
        exp_input = -d2 / (2 * s2 * kappa + 1e-6)
        exp_input = exp_input.clamp(min=-50, max=0)  # exp 输入应该是负数，限制范围
        oks = torch.exp(exp_input)  # (N, K)

        # 只考虑可见关键点
        vis_mask = target_vis > 0
        oks = oks * vis_mask.float()

        # 平均 OKS
        num_visible = vis_mask.sum(dim=1, keepdim=True).clamp(min=1)
        oks = oks.sum(dim=1) / num_visible.squeeze(1)

        # 损失 = 1 - OKS
        return (1 - oks).mean()


class TaskAlignedAssigner(nn.Module):
    """Task-Aligned Assigner for YOLO

    基于预测质量和 IoU 动态分配正负样本

    align_metric = IoU^α × score^β
    """

    def __init__(self,
                 topk: int = 10,
                 alpha: float = 0.5,
                 beta: float = 6.0,
                 eps: float = 1e-9,
                 iou_threshold: float = 0.2):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.iou_threshold = iou_threshold

    @torch.no_grad()
    def forward(self,
                pred_scores: torch.Tensor,
                pred_bboxes: torch.Tensor,
                gt_bboxes: torch.Tensor,
                gt_labels: torch.Tensor,
                num_gts: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pred_scores: 预测分数 (num_anchors,)
            pred_bboxes: 预测框 (num_anchors, 4) [x1, y1, x2, y2]
            gt_bboxes: GT 框 (num_gts, 4) [x1, y1, x2, y2]
            gt_labels: GT 标签 (num_gts,) - 这里全是 1 (人)
            num_gts: GT 数量

        Returns:
            assigned_gt_idx: 每个 anchor 分配的 GT 索引 (num_anchors,) -1 表示负样本
            assigned_labels: 分配的标签 (num_anchors,)
            assigned_scores: 分配的目标分数 (num_anchors,)
        """
        num_anchors = pred_bboxes.shape[0]
        device = pred_bboxes.device

        # 初始化
        assigned_gt_idx = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
        assigned_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_scores = torch.zeros(num_anchors, dtype=torch.float, device=device)

        if num_gts == 0:
            return assigned_gt_idx, assigned_labels, assigned_scores

        # 计算 IoU (num_gts, num_anchors)
        ious = self._compute_iou(gt_bboxes, pred_bboxes)

        # 计算 align metric
        align_metric = ious.pow(self.alpha) * pred_scores.unsqueeze(0).pow(self.beta)

        # 为每个 GT 选择 top-k 个 anchor
        topk_metrics, topk_indices = align_metric.topk(
            min(self.topk, num_anchors), dim=1, largest=True
        )

        # 创建 mask
        is_pos = torch.zeros((num_gts, num_anchors), dtype=torch.bool, device=device)
        for gt_idx in range(num_gts):
            is_pos[gt_idx, topk_indices[gt_idx]] = True

        # 过滤 IoU 太低的
        is_pos = is_pos & (ious > self.iou_threshold)

        # 如果一个 anchor 被多个 GT 选中，选择 IoU 最大的
        anchor_max_iou, anchor_gt_idx = ious.max(dim=0)

        # 只保留被选中的
        pos_mask = is_pos.any(dim=0)

        # 分配
        assigned_gt_idx[pos_mask] = anchor_gt_idx[pos_mask]
        assigned_labels[pos_mask] = 1  # 人
        assigned_scores[pos_mask] = anchor_max_iou[pos_mask]

        return assigned_gt_idx, assigned_labels, assigned_scores

    def _compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """计算 IoU

        Args:
            box1: (N, 4) [x1, y1, x2, y2]
            box2: (M, 4) [x1, y1, x2, y2]

        Returns:
            iou: (N, M)
        """
        N, M = box1.shape[0], box2.shape[0]

        # 扩展维度
        box1 = box1.unsqueeze(1).expand(N, M, 4)  # (N, M, 4)
        box2 = box2.unsqueeze(0).expand(N, M, 4)  # (N, M, 4)

        # 交集
        inter_x1 = torch.max(box1[..., 0], box2[..., 0])
        inter_y1 = torch.max(box1[..., 1], box2[..., 1])
        inter_x2 = torch.min(box1[..., 2], box2[..., 2])
        inter_y2 = torch.min(box1[..., 3], box2[..., 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # 各自面积
        area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

        # IoU
        union = area1 + area2 - inter_area
        iou = inter_area / (union + 1e-6)

        return iou


class YOLOPoseLoss(nn.Module):
    """YOLO-Pose 多人姿态估计损失函数

    总损失 = λ_box × L_box + λ_obj × L_obj + λ_kpt × L_kpt

    - L_box: CIoU Loss (边界框回归)
    - L_obj: Focal Loss (前景/背景分类)
    - L_kpt: KeypointLoss (关键点定位)
    """

    def __init__(self,
                 num_keypoints: int = 17,
                 box_weight: float = 0.5,
                 obj_weight: float = 1.0,
                 kpt_weight: float = 1.0,
                 strides: List[int] = None):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.kpt_weight = kpt_weight
        self.strides = strides or [8, 16, 32, 64]

        # 子损失函数
        self.box_loss = CIoULoss()
        self.obj_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.kpt_loss = KeypointLoss(num_keypoints=num_keypoints, use_oks=True)

        # 正负样本分配器
        self.assigner = TaskAlignedAssigner(topk=13)

    def _check_nan(self, tensor: torch.Tensor, name: str, extra_info: str = "") -> bool:
        """检查 tensor 是否包含 NaN 或 Inf，返回 True 表示有问题"""
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        if has_nan or has_inf:
            print(f"\n{'='*60}")
            print(f"[NaN/Inf DETECTED] {name}")
            print(f"  Shape: {tensor.shape}")
            print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
            valid_mask = ~torch.isnan(tensor) & ~torch.isinf(tensor)
            if valid_mask.any():
                print(f"  Valid Min: {tensor[valid_mask].min().item():.6f}")
                print(f"  Valid Max: {tensor[valid_mask].max().item():.6f}")
            if extra_info:
                print(f"  Extra: {extra_info}")
            print(f"{'='*60}\n")
            return True
        return False

    def forward(self,
                outputs: List[torch.Tensor],
                targets: Dict[str, torch.Tensor],
                input_size: Tuple[int, int] = (640, 640)) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: 多尺度预测输出 [(B, H, W, C), ...]
            targets: 目标字典
                - bboxes: (B, max_persons, 4) [cx, cy, w, h] 归一化
                - keypoints: (B, max_persons, K, 3) [dx, dy, v]
                - num_persons: (B,)
            input_size: 输入图像大小 (H, W)

        Returns:
            loss_dict: 损失字典
        """
        device = outputs[0].device
        batch_size = outputs[0].shape[0]
        input_h, input_w = input_size

        # [哨兵 1] 检查模型原始输出
        for i, out in enumerate(outputs):
            if self._check_nan(out, f"Model output scale {i}"):
                # 返回一个大的损失值，让训练器可以检测并跳过这个 batch
                nan_loss = torch.tensor(float('nan'), device=device, requires_grad=True)
                return {
                    'loss': nan_loss,
                    'box_loss': nan_loss,
                    'obj_loss': nan_loss,
                    'kpt_loss': nan_loss,
                    'num_pos': torch.tensor(0, device=device)
                }

        # 使用 output 的 sum() * 0 确保梯度可以反向传播
        # 即使没有正样本，也保持计算图连接
        zero_loss = outputs[0].sum() * 0
        total_box_loss = zero_loss.clone()
        total_obj_loss = zero_loss.clone()
        total_kpt_loss = zero_loss.clone()

        num_pos = 0

        for scale_idx, output in enumerate(outputs):
            B, H, W, C = output.shape
            stride = self.strides[scale_idx]

            # [关键修复] 强制转换为 float32，防止 exp(10)*stride 溢出 FP16
            # FP16 最大值约 65504，而 exp(10)*64 ≈ 1,409,664 会溢出
            output = output.float()

            # 解析预测
            pred_box = output[..., :4]  # (B, H, W, 4)
            pred_obj = output[..., 4]   # (B, H, W)
            pred_kpts = output[..., 5:].view(B, H, W, self.num_keypoints, 3)  # (B, H, W, K, 3)

            # 创建 grid
            yv, xv = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            grid = torch.stack([xv, yv], dim=-1).float()  # (H, W, 2)

            # 解码预测框为像素坐标
            pred_cx = (grid[..., 0] + pred_box[..., 0].sigmoid()) * stride
            pred_cy = (grid[..., 1] + pred_box[..., 1].sigmoid()) * stride
            # 限制 exp 的输入范围，防止数值爆炸导致 NaN
            pred_w = pred_box[..., 2].clamp(-10, 10).exp() * stride
            pred_h = pred_box[..., 3].clamp(-10, 10).exp() * stride

            # 转换为 [x1, y1, x2, y2]
            pred_x1 = pred_cx - pred_w / 2
            pred_y1 = pred_cy - pred_h / 2
            pred_x2 = pred_cx + pred_w / 2
            pred_y2 = pred_cy + pred_h / 2
            pred_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

            # [哨兵 2] 检查解码后的 Box
            if self._check_nan(pred_boxes_xyxy, f"Decoded boxes scale {scale_idx}",
                               f"pred_w range: [{pred_w.min():.2f}, {pred_w.max():.2f}]"):
                if torch.isnan(pred_w).any() or torch.isinf(pred_w).any():
                    print(" -> pred_w is NaN/Inf (likely exp overflow)")
                if torch.isnan(pred_h).any() or torch.isinf(pred_h).any():
                    print(" -> pred_h is NaN/Inf (likely exp overflow)")
                nan_loss = torch.tensor(float('nan'), device=device, requires_grad=True)
                return {
                    'loss': nan_loss, 'box_loss': nan_loss, 'obj_loss': nan_loss,
                    'kpt_loss': nan_loss, 'num_pos': torch.tensor(0, device=device)
                }

            # 对每个 batch 计算损失
            for b in range(B):
                n_persons = targets['num_persons'][b].item()
                if n_persons == 0:
                    # 没有 GT，只有负样本损失
                    obj_target = torch.zeros_like(pred_obj[b])
                    total_obj_loss += self.obj_loss(pred_obj[b].flatten(), obj_target.flatten())
                    continue

                # 获取 GT
                gt_bboxes_norm = targets['bboxes'][b, :n_persons]  # (n, 4) [cx, cy, w, h]
                gt_kpts = targets['keypoints'][b, :n_persons]       # (n, K, 3)

                # 转换 GT 为像素坐标 [x1, y1, x2, y2]
                gt_cx = gt_bboxes_norm[:, 0] * input_w
                gt_cy = gt_bboxes_norm[:, 1] * input_h
                gt_w = gt_bboxes_norm[:, 2] * input_w
                gt_h = gt_bboxes_norm[:, 3] * input_h

                gt_x1 = gt_cx - gt_w / 2
                gt_y1 = gt_cy - gt_h / 2
                gt_x2 = gt_cx + gt_w / 2
                gt_y2 = gt_cy + gt_h / 2
                gt_boxes_xyxy = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=-1)

                # 展平预测
                pred_boxes_flat = pred_boxes_xyxy[b].view(-1, 4)  # (H*W, 4)
                pred_obj_flat = pred_obj[b].view(-1)              # (H*W,)
                pred_kpts_flat = pred_kpts[b].view(-1, self.num_keypoints, 3)  # (H*W, K, 3)

                # 获取预测分数
                pred_scores_flat = pred_obj_flat.sigmoid()

                # 正负样本分配
                assigned_gt_idx, assigned_labels, assigned_scores = self.assigner(
                    pred_scores_flat,
                    pred_boxes_flat,
                    gt_boxes_xyxy,
                    torch.ones(n_persons, device=device),
                    n_persons
                )

                # 正样本 mask
                pos_mask = assigned_gt_idx >= 0
                num_pos_scale = pos_mask.sum().item()
                num_pos += num_pos_scale

                # Objectness 损失
                obj_target = torch.zeros_like(pred_obj_flat)
                obj_target[pos_mask] = assigned_scores[pos_mask].to(obj_target.dtype)
                total_obj_loss += self.obj_loss(pred_obj_flat, obj_target)

                if num_pos_scale > 0:
                    # Box 损失
                    pos_pred_boxes = pred_boxes_flat[pos_mask]
                    pos_gt_idx = assigned_gt_idx[pos_mask]
                    pos_gt_boxes = gt_boxes_xyxy[pos_gt_idx]

                    # [哨兵 3] 检查 CIoU Loss 输入
                    if self._check_nan(pos_pred_boxes, "CIoU pred_boxes") or \
                       self._check_nan(pos_gt_boxes, "CIoU gt_boxes"):
                        nan_loss = torch.tensor(float('nan'), device=device, requires_grad=True)
                        return {
                            'loss': nan_loss, 'box_loss': nan_loss, 'obj_loss': nan_loss,
                            'kpt_loss': nan_loss, 'num_pos': torch.tensor(0, device=device)
                        }

                    box_loss = self.box_loss(pos_pred_boxes, pos_gt_boxes)

                    # [哨兵 4] 检查 CIoU Loss 输出
                    if self._check_nan(box_loss, "CIoU Loss output"):
                        nan_loss = torch.tensor(float('nan'), device=device, requires_grad=True)
                        return {
                            'loss': nan_loss, 'box_loss': nan_loss, 'obj_loss': nan_loss,
                            'kpt_loss': nan_loss, 'num_pos': torch.tensor(0, device=device)
                        }

                    total_box_loss += box_loss.mean()

                    # Keypoint 损失
                    pos_pred_kpts = pred_kpts_flat[pos_mask]  # (num_pos, K, 3)
                    pos_gt_kpts = gt_kpts[pos_gt_idx]          # (num_pos, K, 3)

                    # 预测的关键点需要解码: sigmoid 后作为相对偏移
                    # 模型输出的是 raw logits，需要 sigmoid 变成 [0,1]
                    # 然后映射到 [-1, 1] 范围作为相对偏移 (dx, dy 相对于 bbox)
                    # 使用 sigmoid * 2 - 1 映射到 [-1, 1]
                    pred_kpts_coord = pos_pred_kpts[..., :2].sigmoid() * 2 - 1  # (num_pos, K, 2)

                    # GT 关键点已经是相对偏移 (dx, dy)，可以直接使用
                    target_kpts_coord = pos_gt_kpts[..., :2]   # (num_pos, K, 2)
                    target_vis = pos_gt_kpts[..., 2]           # (num_pos, K)

                    # OKS 用的面积：使用归一化面积 (因为坐标都是归一化的)
                    # 归一化面积 = w_norm * h_norm (而不是像素面积)
                    pos_gt_w_norm = gt_bboxes_norm[pos_gt_idx, 2]  # 归一化宽度
                    pos_gt_h_norm = gt_bboxes_norm[pos_gt_idx, 3]  # 归一化高度
                    # 注意：OKS 中面积用于归一化距离，这里用 1.0 作为参考
                    # 因为 dx, dy 已经是归一化到 bbox 尺寸的，直接用 1.0
                    pos_gt_areas = torch.ones_like(pos_gt_w_norm)

                    # [哨兵 5] 检查 Keypoint Loss 输入
                    if self._check_nan(pred_kpts_coord, "Keypoint pred_coords") or \
                       self._check_nan(target_kpts_coord, "Keypoint target_coords"):
                        nan_loss = torch.tensor(float('nan'), device=device, requires_grad=True)
                        return {
                            'loss': nan_loss, 'box_loss': nan_loss, 'obj_loss': nan_loss,
                            'kpt_loss': nan_loss, 'num_pos': torch.tensor(0, device=device)
                        }

                    kpt_loss_dict = self.kpt_loss(
                        pred_kpts_coord,
                        target_kpts_coord,
                        target_vis,
                        pos_gt_areas
                    )

                    # [哨兵 6] 检查 Keypoint Loss 输出
                    if self._check_nan(kpt_loss_dict['kpt_loss'], "Keypoint Loss output"):
                        nan_loss = torch.tensor(float('nan'), device=device, requires_grad=True)
                        return {
                            'loss': nan_loss, 'box_loss': nan_loss, 'obj_loss': nan_loss,
                            'kpt_loss': nan_loss, 'num_pos': torch.tensor(0, device=device)
                        }

                    total_kpt_loss += kpt_loss_dict['kpt_loss']

        # 归一化
        num_scales = len(outputs)
        num_pos = max(num_pos, 1)

        total_box_loss = self.box_weight * total_box_loss / num_scales
        total_obj_loss = self.obj_weight * total_obj_loss / num_scales
        total_kpt_loss = self.kpt_weight * total_kpt_loss / num_scales

        total_loss = total_box_loss + total_obj_loss + total_kpt_loss

        return {
            'loss': total_loss,
            'box_loss': total_box_loss,
            'obj_loss': total_obj_loss,
            'kpt_loss': total_kpt_loss,
            'num_pos': torch.tensor(num_pos, device=device)
        }


def build_yolo_loss(num_keypoints: int = 17,
                    box_weight: float = 0.5,
                    obj_weight: float = 1.0,
                    kpt_weight: float = 1.0,
                    strides: List[int] = None) -> YOLOPoseLoss:
    """构建 YOLO-Pose 损失函数

    Args:
        num_keypoints: 关键点数量
        box_weight: 边界框损失权重
        obj_weight: 目标置信度损失权重
        kpt_weight: 关键点损失权重
        strides: 各尺度的步长

    Returns:
        YOLOPoseLoss 实例
    """
    return YOLOPoseLoss(
        num_keypoints=num_keypoints,
        box_weight=box_weight,
        obj_weight=obj_weight,
        kpt_weight=kpt_weight,
        strides=strides
    )
