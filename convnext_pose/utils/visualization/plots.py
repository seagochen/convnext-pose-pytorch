"""
姿态检测可视化工具
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional


# COCO骨架连接
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
    (5, 11), (6, 12), (11, 12),  # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
]

# 关键点颜色 (BGR)
KEYPOINT_COLORS = [
    (0, 255, 255),    # nose - 黄色
    (0, 191, 255),    # left_eye - 深黄色
    (0, 191, 255),    # right_eye
    (0, 127, 255),    # left_ear - 橙色
    (0, 127, 255),    # right_ear
    (255, 0, 0),      # left_shoulder - 蓝色
    (255, 0, 0),      # right_shoulder
    (255, 85, 0),     # left_elbow
    (255, 85, 0),     # right_elbow
    (255, 170, 0),    # left_wrist
    (255, 170, 0),    # right_wrist
    (0, 255, 0),      # left_hip - 绿色
    (0, 255, 0),      # right_hip
    (85, 255, 0),     # left_knee
    (85, 255, 0),     # right_knee
    (170, 255, 0),    # left_ankle
    (170, 255, 0),    # right_ankle
]

# 骨架颜色
SKELETON_COLORS = [
    (0, 255, 255), (0, 255, 255), (0, 191, 255), (0, 191, 255),  # 头部
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 85, 0), (255, 170, 0),  # 上肢
    (0, 255, 0), (0, 255, 0), (0, 255, 85),  # 躯干
    (85, 255, 0), (170, 255, 0), (85, 255, 0), (170, 255, 0)  # 下肢
]


def draw_keypoints(image: np.ndarray,
                   keypoints: np.ndarray,
                   visibility: Optional[np.ndarray] = None,
                   radius: int = 4,
                   colors: Optional[List] = None) -> np.ndarray:
    """绘制关键点

    Args:
        image: 输入图像 (H, W, 3) BGR格式
        keypoints: 关键点坐标 (K, 2)
        visibility: 可见性标志 (K,)
        radius: 关键点半径
        colors: 关键点颜色列表

    Returns:
        带关键点的图像
    """
    image = image.copy()

    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()
    if visibility is not None and isinstance(visibility, torch.Tensor):
        visibility = visibility.cpu().numpy()

    if colors is None:
        colors = KEYPOINT_COLORS

    num_keypoints = len(keypoints)

    for i in range(num_keypoints):
        if visibility is not None and visibility[i] <= 0:
            continue

        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])

        if x < 0 or y < 0:
            continue

        color = colors[i % len(colors)]
        cv2.circle(image, (x, y), radius, color, -1)
        cv2.circle(image, (x, y), radius + 1, (255, 255, 255), 1)

    return image


def draw_skeleton(image: np.ndarray,
                  keypoints: np.ndarray,
                  visibility: Optional[np.ndarray] = None,
                  skeleton: Optional[List[Tuple[int, int]]] = None,
                  line_width: int = 2,
                  colors: Optional[List] = None) -> np.ndarray:
    """绘制骨架

    Args:
        image: 输入图像 (H, W, 3) BGR格式
        keypoints: 关键点坐标 (K, 2)
        visibility: 可见性标志 (K,)
        skeleton: 骨架连接列表
        line_width: 线宽
        colors: 骨架颜色列表

    Returns:
        带骨架的图像
    """
    image = image.copy()

    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()
    if visibility is not None and isinstance(visibility, torch.Tensor):
        visibility = visibility.cpu().numpy()

    if skeleton is None:
        skeleton = COCO_SKELETON
    if colors is None:
        colors = SKELETON_COLORS

    for idx, (i, j) in enumerate(skeleton):
        if i >= len(keypoints) or j >= len(keypoints):
            continue

        if visibility is not None:
            if visibility[i] <= 0 or visibility[j] <= 0:
                continue

        x1, y1 = int(keypoints[i, 0]), int(keypoints[i, 1])
        x2, y2 = int(keypoints[j, 0]), int(keypoints[j, 1])

        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            continue

        color = colors[idx % len(colors)]
        cv2.line(image, (x1, y1), (x2, y2), color, line_width)

    return image


def draw_pose(image: np.ndarray,
              keypoints: np.ndarray,
              visibility: Optional[np.ndarray] = None,
              draw_kpts: bool = True,
              draw_skel: bool = True,
              kpt_radius: int = 4,
              line_width: int = 2) -> np.ndarray:
    """绘制完整的姿态

    Args:
        image: 输入图像
        keypoints: 关键点坐标 (K, 2) 或 (K, 3)
        visibility: 可见性标志
        draw_kpts: 是否绘制关键点
        draw_skel: 是否绘制骨架
        kpt_radius: 关键点半径
        line_width: 线宽

    Returns:
        可视化结果
    """
    if keypoints.shape[-1] == 3:
        visibility = keypoints[:, 2]
        keypoints = keypoints[:, :2]

    if draw_skel:
        image = draw_skeleton(image, keypoints, visibility, line_width=line_width)
    if draw_kpts:
        image = draw_keypoints(image, keypoints, visibility, radius=kpt_radius)

    return image


def draw_heatmaps(heatmaps: np.ndarray,
                  image: Optional[np.ndarray] = None,
                  alpha: float = 0.5) -> np.ndarray:
    """可视化热图

    Args:
        heatmaps: 热图 (K, H, W) 或 (B, K, H, W)
        image: 背景图像 (可选)
        alpha: 热图透明度

    Returns:
        热图可视化
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()

    if heatmaps.ndim == 4:
        heatmaps = heatmaps[0]  # 取第一个batch

    num_joints, height, width = heatmaps.shape

    # 合并所有关键点热图
    combined = np.max(heatmaps, axis=0)

    # 归一化
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    combined = (combined * 255).astype(np.uint8)

    # 应用colormap
    heatmap_color = cv2.applyColorMap(combined, cv2.COLORMAP_JET)

    if image is not None:
        # 调整大小
        if image.shape[:2] != (height, width):
            heatmap_color = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))
        # 叠加
        result = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    else:
        result = heatmap_color

    return result


def draw_individual_heatmaps(heatmaps: np.ndarray,
                             keypoint_names: Optional[List[str]] = None,
                             cols: int = 5) -> np.ndarray:
    """绘制每个关键点的独立热图

    Args:
        heatmaps: 热图 (K, H, W)
        keypoint_names: 关键点名称
        cols: 每行显示的热图数量

    Returns:
        热图网格
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()

    if heatmaps.ndim == 4:
        heatmaps = heatmaps[0]

    num_joints, height, width = heatmaps.shape
    rows = (num_joints + cols - 1) // cols

    # 创建画布
    canvas = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)

    for i in range(num_joints):
        row = i // cols
        col = i % cols

        # 归一化
        hm = heatmaps[i]
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        hm = (hm * 255).astype(np.uint8)

        # 应用colormap
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

        # 添加标签
        if keypoint_names is not None:
            cv2.putText(hm_color, keypoint_names[i], (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y1, y2 = row * height, (row + 1) * height
        x1, x2 = col * width, (col + 1) * width
        canvas[y1:y2, x1:x2] = hm_color

    return canvas


def draw_paf(paf: np.ndarray,
             image: Optional[np.ndarray] = None,
             stride: int = 8) -> np.ndarray:
    """可视化Part Affinity Fields

    Args:
        paf: PAF (L*2, H, W) - 每个肢体有x和y两个方向
        image: 背景图像
        stride: 向量场采样步长

    Returns:
        PAF可视化
    """
    if isinstance(paf, torch.Tensor):
        paf = paf.cpu().numpy()

    if paf.ndim == 4:
        paf = paf[0]

    num_limbs = paf.shape[0] // 2
    height, width = paf.shape[1], paf.shape[2]

    if image is not None:
        canvas = image.copy()
        scale_x = image.shape[1] / width
        scale_y = image.shape[0] / height
    else:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        scale_x = scale_y = 1

    # 绘制向量场
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            for limb in range(num_limbs):
                vx = paf[limb * 2, y, x]
                vy = paf[limb * 2 + 1, y, x]

                magnitude = np.sqrt(vx ** 2 + vy ** 2)
                if magnitude < 0.1:
                    continue

                # 归一化方向
                vx, vy = vx / magnitude, vy / magnitude

                # 绘制箭头
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + vx * stride) * scale_x)
                y2 = int((y + vy * stride) * scale_y)

                color = SKELETON_COLORS[limb % len(SKELETON_COLORS)]
                cv2.arrowedLine(canvas, (x1, y1), (x2, y2), color, 1, tipLength=0.3)

    return canvas


def visualize_predictions(images: np.ndarray,
                          predictions: np.ndarray,
                          targets: Optional[np.ndarray] = None,
                          max_samples: int = 8) -> np.ndarray:
    """批量可视化预测结果

    Args:
        images: 图像batch (B, 3, H, W) 或 (B, H, W, 3)
        predictions: 预测关键点 (B, K, 2) 或 (B, K, 3)
        targets: 目标关键点 (可选)
        max_samples: 最大显示样本数

    Returns:
        可视化网格
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if targets is not None and isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # 转换图像格式
    if images.shape[1] == 3:  # (B, 3, H, W)
        images = images.transpose(0, 2, 3, 1)

    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = images * std + mean
    images = (images * 255).clip(0, 255).astype(np.uint8)
    images = images[..., ::-1]  # RGB -> BGR

    batch_size = min(len(images), max_samples)
    vis_list = []

    for i in range(batch_size):
        img = images[i].copy()

        # 绘制预测
        img = draw_pose(img, predictions[i])

        # 绘制目标 (用不同颜色)
        if targets is not None:
            target_colors = [(0, 0, 255)] * 17  # 红色
            img = draw_keypoints(img, targets[i][:, :2],
                                targets[i][:, 2] if targets.shape[-1] == 3 else None,
                                radius=3, colors=target_colors)

        vis_list.append(img)

    # 拼接成网格
    cols = min(4, batch_size)
    rows = (batch_size + cols - 1) // cols

    height, width = vis_list[0].shape[:2]
    canvas = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)

    for i, img in enumerate(vis_list):
        row = i // cols
        col = i % cols
        canvas[row * height:(row + 1) * height, col * width:(col + 1) * width] = img

    return canvas


def draw_bbox(image: np.ndarray,
              bbox: np.ndarray,
              color: Tuple[int, int, int] = (0, 255, 0),
              thickness: int = 2,
              label: Optional[str] = None) -> np.ndarray:
    """绘制边界框

    Args:
        image: 输入图像
        bbox: 边界框 [x1, y1, x2, y2]
        color: 颜色 (BGR)
        thickness: 线宽
        label: 标签文字

    Returns:
        绘制后的图像
    """
    image = image.copy()
    x1, y1, x2, y2 = map(int, bbox[:4])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label:
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                               font_scale, font_thickness)
        cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (255, 255, 255), font_thickness)

    return image


def draw_multi_person_pose(image: np.ndarray,
                           bboxes: np.ndarray,
                           keypoints: np.ndarray,
                           scores: Optional[np.ndarray] = None,
                           is_gt: bool = False,
                           draw_bbox: bool = True) -> np.ndarray:
    """绘制多人姿态检测结果

    Args:
        image: 输入图像 (H, W, 3) BGR
        bboxes: 边界框 (N, 4) [x1, y1, x2, y2]
        keypoints: 关键点 (N, K, 2) 或 (N, K, 3)
        scores: 检测分数 (N,)
        is_gt: 是否是 GT (用于区分颜色)
        draw_bbox: 是否绘制边界框

    Returns:
        可视化结果
    """
    image = image.copy()

    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    num_persons = len(bboxes)

    # GT 用蓝色系，Pred 用绿色系
    if is_gt:
        bbox_color = (255, 100, 0)  # 蓝色
        kpt_alpha = 0.6
    else:
        bbox_color = (0, 255, 100)  # 绿色
        kpt_alpha = 1.0

    for i in range(num_persons):
        # 绘制边界框
        if draw_bbox and bboxes is not None and len(bboxes) > i:
            label = None
            if scores is not None:
                label = f'{scores[i]:.2f}'
            x1, y1, x2, y2 = map(int, bboxes[i][:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 2)
            if label:
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, bbox_color, 1)

        # 绘制关键点和骨架
        if keypoints is not None and len(keypoints) > i:
            kpts = keypoints[i]
            if kpts.shape[-1] == 3:
                vis = kpts[:, 2]
                kpts = kpts[:, :2]
            else:
                vis = None

            # 绘制骨架
            image = draw_skeleton(image, kpts, vis, line_width=2)
            # 绘制关键点
            image = draw_keypoints(image, kpts, vis, radius=4)

    return image


def visualize_val_samples(images: torch.Tensor,
                          pred_bboxes_list: List[np.ndarray],
                          pred_keypoints_list: List[np.ndarray],
                          pred_scores_list: List[np.ndarray],
                          gt_bboxes_list: List[np.ndarray],
                          gt_keypoints_list: List[np.ndarray],
                          save_path: str,
                          max_samples: int = 10) -> None:
    """可视化验证样本的 GT 和预测结果并保存

    Args:
        images: 图像 tensor (B, 3, H, W)
        pred_bboxes_list: 每张图的预测边界框列表
        pred_keypoints_list: 每张图的预测关键点列表
        pred_scores_list: 每张图的预测分数列表
        gt_bboxes_list: 每张图的 GT 边界框列表
        gt_keypoints_list: 每张图的 GT 关键点列表
        save_path: 保存路径
        max_samples: 最大样本数
    """
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    # 转换图像
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    # (B, 3, H, W) -> (B, H, W, 3)
    if images.shape[1] == 3:
        images = images.transpose(0, 2, 3, 1)

    # 反归一化 (假设是 [0, 1] 归一化)
    images = (images * 255).clip(0, 255).astype(np.uint8)
    # RGB -> BGR for cv2
    images = images[..., ::-1].copy()

    batch_size = min(len(images), max_samples)
    vis_list = []

    for i in range(batch_size):
        img = images[i].copy()
        h, w = img.shape[:2]

        # 创建左右对比图: 左边 GT，右边 Pred
        canvas = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
        canvas[:, :w] = img.copy()
        canvas[:, w + 10:] = img.copy()

        # 左边绘制 GT
        gt_img = img.copy()
        if i < len(gt_bboxes_list) and gt_bboxes_list[i] is not None:
            gt_img = draw_multi_person_pose(
                gt_img,
                gt_bboxes_list[i],
                gt_keypoints_list[i],
                is_gt=True,
                draw_bbox=True
            )
        # 添加标签
        cv2.putText(gt_img, 'GT', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 100, 0), 2)
        canvas[:, :w] = gt_img

        # 右边绘制预测
        pred_img = img.copy()
        if i < len(pred_bboxes_list) and pred_bboxes_list[i] is not None and len(pred_bboxes_list[i]) > 0:
            pred_img = draw_multi_person_pose(
                pred_img,
                pred_bboxes_list[i],
                pred_keypoints_list[i],
                scores=pred_scores_list[i] if i < len(pred_scores_list) else None,
                is_gt=False,
                draw_bbox=True
            )
        # 添加标签
        cv2.putText(pred_img, 'Pred', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 255, 100), 2)
        canvas[:, w + 10:] = pred_img

        vis_list.append(canvas)

    # 垂直拼接所有样本
    if vis_list:
        result = np.vstack(vis_list)
        cv2.imwrite(save_path, result)


def save_visualization(image: np.ndarray, path: str):
    """保存可视化结果"""
    cv2.imwrite(path, image)


def show_visualization(image: np.ndarray, window_name: str = 'Visualization'):
    """显示可视化结果"""
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
