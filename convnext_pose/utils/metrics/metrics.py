"""
姿态检测评估指标

包含:
1. PCK (Percentage of Correct Keypoints)
2. OKS (Object Keypoint Similarity)
3. AP (Average Precision)
"""

import numpy as np
import torch
from typing import Tuple, Optional, List


def compute_pck(pred: np.ndarray, target: np.ndarray,
                threshold: float = 0.5,
                normalize_by: str = 'bbox') -> Tuple[float, np.ndarray]:
    """计算PCK (Percentage of Correct Keypoints)

    Args:
        pred: 预测关键点 (B, K, 2) 或 (K, 2)
        target: 目标关键点 (B, K, 2) 或 (K, 2)
        threshold: 阈值比例
        normalize_by: 归一化方式 ('bbox', 'torso', 'head')

    Returns:
        pck: PCK值
        per_joint_pck: 每个关键点的PCK
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        target = target[np.newaxis, ...]

    batch_size, num_joints, _ = pred.shape

    # 计算归一化因子
    if normalize_by == 'bbox':
        # 使用边界框对角线
        bbox_min = target.min(axis=1)
        bbox_max = target.max(axis=1)
        normalize_factor = np.linalg.norm(bbox_max - bbox_min, axis=1, keepdims=True)
    elif normalize_by == 'torso':
        # 使用躯干长度 (左肩到右髋 或 右肩到左髋)
        # 假设: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
        torso1 = np.linalg.norm(target[:, 5] - target[:, 12], axis=1)
        torso2 = np.linalg.norm(target[:, 6] - target[:, 11], axis=1)
        normalize_factor = np.maximum(torso1, torso2)[:, np.newaxis]
    elif normalize_by == 'head':
        # 使用头部大小 (左耳到右耳)
        # 假设: 3=left_ear, 4=right_ear
        normalize_factor = np.linalg.norm(target[:, 3] - target[:, 4], axis=1, keepdims=True)
    else:
        normalize_factor = np.ones((batch_size, 1))

    normalize_factor = np.maximum(normalize_factor, 1e-6)

    # 计算距离
    dist = np.linalg.norm(pred - target, axis=2)  # (B, K)

    # 归一化距离
    normalized_dist = dist / normalize_factor

    # 计算正确率
    correct = (normalized_dist < threshold).astype(float)

    # 总体PCK
    pck = correct.mean()

    # 每个关键点的PCK
    per_joint_pck = correct.mean(axis=0)

    return pck, per_joint_pck


def compute_oks(pred: np.ndarray, target: np.ndarray,
                area: np.ndarray,
                visibility: Optional[np.ndarray] = None,
                sigmas: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """计算OKS (Object Keypoint Similarity)

    Args:
        pred: 预测关键点 (B, K, 2) 或 (K, 2)
        target: 目标关键点 (B, K, 2) 或 (K, 2)
        area: 目标区域面积 (B,) 或标量
        visibility: 关键点可见性 (B, K) 或 (K,)
        sigmas: 关键点的sigma值

    Returns:
        oks: 平均OKS值
        per_instance_oks: 每个样本的OKS
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(area, torch.Tensor):
        area = area.cpu().numpy()
    if visibility is not None and isinstance(visibility, torch.Tensor):
        visibility = visibility.cpu().numpy()

    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        target = target[np.newaxis, ...]

    if np.isscalar(area):
        area = np.array([area])

    batch_size, num_joints, _ = pred.shape

    # COCO关键点sigma
    if sigmas is None:
        sigmas = np.array([
            0.026, 0.025, 0.025, 0.035, 0.035,
            0.079, 0.079, 0.072, 0.072, 0.062,
            0.062, 0.107, 0.107, 0.087, 0.087,
            0.089, 0.089
        ])[:num_joints]

    if visibility is None:
        visibility = np.ones((batch_size, num_joints))

    # 计算欧氏距离
    d = np.linalg.norm(pred - target, axis=2)  # (B, K)

    # OKS公式
    kappa = 2 * sigmas ** 2
    oks_per_joint = np.exp(-d ** 2 / (2 * area[:, np.newaxis] * kappa))

    # 只考虑可见关键点
    oks_per_joint = oks_per_joint * visibility

    # 每个样本的OKS
    num_visible = visibility.sum(axis=1).clip(min=1)
    per_instance_oks = oks_per_joint.sum(axis=1) / num_visible

    # 平均OKS
    oks = per_instance_oks.mean()

    return oks, per_instance_oks


def compute_ap(oks_list: List[float],
               thresholds: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """计算AP (Average Precision)

    Args:
        oks_list: OKS值列表
        thresholds: OKS阈值列表

    Returns:
        ap: 平均精度
        ap_per_threshold: 每个阈值的精度
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.05)  # COCO标准阈值

    oks_array = np.array(oks_list)
    ap_per_threshold = []

    for thresh in thresholds:
        precision = (oks_array >= thresh).mean()
        ap_per_threshold.append(precision)

    ap = np.mean(ap_per_threshold)

    return ap, np.array(ap_per_threshold)


def decode_heatmap(heatmaps: np.ndarray, stride: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """从热图解码关键点坐标

    Args:
        heatmaps: 热图 (B, K, H, W)
        stride: 热图相对于原图的步长

    Returns:
        keypoints: 关键点坐标 (B, K, 2)
        scores: 关键点置信度 (B, K)
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()

    batch_size, num_joints, height, width = heatmaps.shape

    keypoints = np.zeros((batch_size, num_joints, 2), dtype=np.float32)
    scores = np.zeros((batch_size, num_joints), dtype=np.float32)

    for b in range(batch_size):
        for k in range(num_joints):
            hm = heatmaps[b, k]

            # 找到最大值位置
            idx = np.argmax(hm)
            y, x = np.unravel_index(idx, (height, width))
            score = hm[y, x]

            # 亚像素精度
            px, py = float(x), float(y)
            if score > 0:
                if 0 < x < width - 1:
                    diff_x = hm[y, x + 1] - hm[y, x - 1]
                    px += 0.25 * np.sign(diff_x)
                if 0 < y < height - 1:
                    diff_y = hm[y + 1, x] - hm[y - 1, x]
                    py += 0.25 * np.sign(diff_y)

            keypoints[b, k, 0] = px * stride
            keypoints[b, k, 1] = py * stride
            scores[b, k] = score

    return keypoints, scores


def decode_heatmap_dark(heatmaps: np.ndarray,
                        stride: int = 4,
                        kernel_size: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    """DARK (Distribution-Aware coordinate Representation of Keypoint) 解码

    来自论文: Distribution-Aware Coordinate Representation for Human Pose Estimation

    Args:
        heatmaps: 热图 (B, K, H, W)
        stride: 热图相对于原图的步长
        kernel_size: 高斯核大小

    Returns:
        keypoints: 关键点坐标 (B, K, 2)
        scores: 关键点置信度 (B, K)
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()

    batch_size, num_joints, height, width = heatmaps.shape

    keypoints = np.zeros((batch_size, num_joints, 2), dtype=np.float32)
    scores = np.zeros((batch_size, num_joints), dtype=np.float32)

    for b in range(batch_size):
        for k in range(num_joints):
            hm = heatmaps[b, k].copy()

            # 找到最大值位置
            idx = np.argmax(hm)
            y, x = np.unravel_index(idx, (height, width))
            score = hm[y, x]

            if score < 0.001:
                keypoints[b, k] = [x * stride, y * stride]
                scores[b, k] = score
                continue

            # DARK解码
            # 计算二阶导数
            radius = kernel_size // 2
            x1 = max(0, x - radius)
            x2 = min(width, x + radius + 1)
            y1 = max(0, y - radius)
            y2 = min(height, y + radius + 1)

            patch = hm[y1:y2, x1:x2]

            if patch.size > 1:
                # 对数变换
                patch = np.log(np.maximum(patch, 1e-10))

                # 使用泰勒展开
                if x > 0 and x < width - 1 and y > 0 and y < height - 1:
                    dx = 0.5 * (hm[y, x + 1] - hm[y, x - 1])
                    dy = 0.5 * (hm[y + 1, x] - hm[y - 1, x])
                    dxx = hm[y, x + 1] - 2 * hm[y, x] + hm[y, x - 1]
                    dyy = hm[y + 1, x] - 2 * hm[y, x] + hm[y - 1, x]
                    dxy = 0.25 * (hm[y + 1, x + 1] - hm[y + 1, x - 1] -
                                  hm[y - 1, x + 1] + hm[y - 1, x - 1])

                    # 避免除零
                    det = dxx * dyy - dxy * dxy
                    if abs(det) > 1e-6:
                        offset_x = -(dyy * dx - dxy * dy) / det
                        offset_y = -(dxx * dy - dxy * dx) / det

                        # 限制偏移范围
                        offset_x = np.clip(offset_x, -0.5, 0.5)
                        offset_y = np.clip(offset_y, -0.5, 0.5)

                        x = x + offset_x
                        y = y + offset_y

            keypoints[b, k, 0] = x * stride
            keypoints[b, k, 1] = y * stride
            scores[b, k] = score

    return keypoints, scores


class PoseEvaluator:
    """姿态检测评估器"""

    def __init__(self, num_keypoints: int = 17, threshold: float = 0.5):
        self.num_keypoints = num_keypoints
        self.threshold = threshold
        self.reset()

    def reset(self):
        """重置累积状态"""
        self.predictions = []
        self.targets = []
        self.areas = []
        self.visibilities = []

    def update(self, pred: np.ndarray, target: np.ndarray,
               area: Optional[np.ndarray] = None,
               visibility: Optional[np.ndarray] = None):
        """更新评估状态

        Args:
            pred: 预测关键点 (B, K, 2)
            target: 目标关键点 (B, K, 2)
            area: 区域面积 (B,)
            visibility: 可见性 (B, K)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        self.predictions.append(pred)
        self.targets.append(target)

        if area is not None:
            if isinstance(area, torch.Tensor):
                area = area.cpu().numpy()
            self.areas.append(area)

        if visibility is not None:
            if isinstance(visibility, torch.Tensor):
                visibility = visibility.cpu().numpy()
            self.visibilities.append(visibility)

    def compute(self) -> dict:
        """计算所有评估指标

        Returns:
            metrics: 指标字典
        """
        pred = np.concatenate(self.predictions, axis=0)
        target = np.concatenate(self.targets, axis=0)

        metrics = {}

        # PCK
        pck, per_joint_pck = compute_pck(pred, target, threshold=self.threshold)
        metrics['PCK'] = pck
        metrics['PCK_per_joint'] = per_joint_pck

        # OKS
        if len(self.areas) > 0:
            area = np.concatenate(self.areas, axis=0)
            visibility = np.concatenate(self.visibilities, axis=0) if len(self.visibilities) > 0 else None
            oks, per_instance_oks = compute_oks(pred, target, area, visibility)
            metrics['OKS'] = oks

            # AP
            ap, ap_per_thresh = compute_ap(per_instance_oks)
            metrics['AP'] = ap
            metrics['AP50'] = (per_instance_oks >= 0.5).mean()
            metrics['AP75'] = (per_instance_oks >= 0.75).mean()

        return metrics
