"""
YOLO-Pose 后处理工具

包含:
- NMS: 非极大值抑制
- 坐标变换工具
- 可视化工具
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
import cv2


def nms(
    bboxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thresh: float = 0.65
) -> torch.Tensor:
    """非极大值抑制

    Args:
        bboxes: (N, 4) [x1, y1, x2, y2]
        scores: (N,)
        iou_thresh: IoU 阈值

    Returns:
        keep: 保留的索引
    """
    if bboxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=bboxes.device)

    # 使用 torchvision 的 NMS (如果可用)
    try:
        from torchvision.ops import nms as tv_nms
        return tv_nms(bboxes, scores, iou_thresh)
    except ImportError:
        pass

    # 手动实现 NMS
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break

        i = order[0].item()
        keep.append(i)

        # 计算 IoU
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留 IoU 小于阈值的
        mask = iou <= iou_thresh
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=bboxes.device)


def postprocess(
    bboxes: torch.Tensor,
    scores: torch.Tensor,
    keypoints: torch.Tensor,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.65,
    max_det: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """后处理: 置信度过滤 + NMS

    Args:
        bboxes: (N, 4) [x1, y1, x2, y2]
        scores: (N,)
        keypoints: (N, K, 3) [x, y, conf]
        conf_thresh: 置信度阈值
        iou_thresh: NMS IoU 阈值
        max_det: 最大检测数量

    Returns:
        bboxes: (M, 4)
        scores: (M,)
        keypoints: (M, K, 3)
    """
    # 置信度过滤
    mask = scores > conf_thresh
    bboxes = bboxes[mask]
    scores = scores[mask]
    keypoints = keypoints[mask]

    if bboxes.numel() == 0:
        return bboxes, scores, keypoints

    # NMS
    keep = nms(bboxes, scores, iou_thresh)

    # 限制数量
    keep = keep[:max_det]

    return bboxes[keep], scores[keep], keypoints[keep]


def scale_coords(
    coords: torch.Tensor,
    from_size: Tuple[int, int],
    to_size: Tuple[int, int],
    ratio_pad: Optional[Tuple[float, Tuple[float, float]]] = None
) -> torch.Tensor:
    """缩放坐标

    Args:
        coords: (..., 2) 或 (..., 4) 坐标
        from_size: 源尺寸 (H, W)
        to_size: 目标尺寸 (H, W)
        ratio_pad: (ratio, (pad_w, pad_h)) letterbox 参数

    Returns:
        缩放后的坐标
    """
    if ratio_pad is None:
        # 简单缩放
        from_h, from_w = from_size
        to_h, to_w = to_size

        scale_x = to_w / from_w
        scale_y = to_h / from_h

        coords = coords.clone()
        if coords.shape[-1] == 4:
            coords[..., [0, 2]] *= scale_x
            coords[..., [1, 3]] *= scale_y
        else:
            coords[..., 0] *= scale_x
            coords[..., 1] *= scale_y
    else:
        # letterbox 缩放
        ratio, (pad_w, pad_h) = ratio_pad

        coords = coords.clone()
        if coords.shape[-1] == 4:
            coords[..., [0, 2]] -= pad_w
            coords[..., [1, 3]] -= pad_h
            coords[..., :4] /= ratio
        else:
            coords[..., 0] -= pad_w
            coords[..., 1] -= pad_h
            coords[..., :2] /= ratio

    return coords


def clip_coords(
    coords: torch.Tensor,
    img_size: Tuple[int, int]
) -> torch.Tensor:
    """裁剪坐标到图像范围内

    Args:
        coords: (..., 2) 或 (..., 4) 坐标
        img_size: (H, W)

    Returns:
        裁剪后的坐标
    """
    h, w = img_size
    coords = coords.clone()

    if coords.shape[-1] == 4:
        coords[..., 0].clamp_(0, w)
        coords[..., 1].clamp_(0, h)
        coords[..., 2].clamp_(0, w)
        coords[..., 3].clamp_(0, h)
    else:
        coords[..., 0].clamp_(0, w)
        coords[..., 1].clamp_(0, h)

    return coords


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Letterbox 图像缩放

    Args:
        img: 输入图像 (H, W, C)
        new_shape: 目标尺寸 (H, W)
        color: 填充颜色
        auto: 自动计算最小填充
        scale_fill: 拉伸填充 (不保持比例)
        scaleup: 是否放大
        stride: 步长对齐

    Returns:
        img: 缩放后的图像
        ratio: 缩放比例
        pad: (pad_w, pad_h) 填充大小
    """
    shape = img.shape[:2]  # (H, W)

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # 计算新尺寸
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    # 缩放
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 填充
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)


def draw_pose(
    img: np.ndarray,
    keypoints: np.ndarray,
    scores: Optional[np.ndarray] = None,
    bboxes: Optional[np.ndarray] = None,
    skeleton: Optional[List[Tuple[int, int]]] = None,
    kpt_thresh: float = 0.3,
    kpt_radius: int = 5,
    line_thickness: int = 2,
    bbox_color: Tuple[int, int, int] = (0, 255, 0),
    kpt_color: Tuple[int, int, int] = (0, 0, 255),
    limb_color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """绘制姿态估计结果

    Args:
        img: 输入图像 (H, W, C) BGR
        keypoints: (N, K, 3) [x, y, conf]
        scores: (N,) 检测置信度 (可选)
        bboxes: (N, 4) [x1, y1, x2, y2] (可选)
        skeleton: 骨架连接定义 (可选)
        kpt_thresh: 关键点置信度阈值
        kpt_radius: 关键点半径
        line_thickness: 线条粗细
        bbox_color: 边界框颜色
        kpt_color: 关键点颜色
        limb_color: 骨架颜色

    Returns:
        绘制后的图像
    """
    img = img.copy()

    # 默认 COCO 骨架
    if skeleton is None:
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上身
            (5, 11), (6, 12), (11, 12),  # 躯干
            (11, 13), (13, 15), (12, 14), (14, 16)  # 下身
        ]

    # 绘制每个人
    for i in range(len(keypoints)):
        kpts = keypoints[i]

        # 绘制边界框
        if bboxes is not None:
            x1, y1, x2, y2 = bboxes[i].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, line_thickness)

            # 绘制置信度
            if scores is not None:
                label = f'{scores[i]:.2f}'
                cv2.putText(img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1)

        # 绘制骨架
        for j, k in skeleton:
            if j < len(kpts) and k < len(kpts):
                if kpts[j, 2] > kpt_thresh and kpts[k, 2] > kpt_thresh:
                    pt1 = (int(kpts[j, 0]), int(kpts[j, 1]))
                    pt2 = (int(kpts[k, 0]), int(kpts[k, 1]))
                    cv2.line(img, pt1, pt2, limb_color, line_thickness)

        # 绘制关键点
        for j in range(len(kpts)):
            if kpts[j, 2] > kpt_thresh:
                pt = (int(kpts[j, 0]), int(kpts[j, 1]))
                cv2.circle(img, pt, kpt_radius, kpt_color, -1)

    return img


class YOLOPosePredictor:
    """YOLO-Pose 推理器

    封装了预处理、推理、后处理的完整流程
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.65,
        input_size: Tuple[int, int] = (640, 640)
    ):
        """
        Args:
            model: YOLO-Pose 模型
            device: 计算设备
            conf_thresh: 置信度阈值
            iou_thresh: NMS IoU 阈值
            input_size: 输入尺寸 (H, W)
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.input_size = input_size

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        img: Union[np.ndarray, torch.Tensor],
        return_vis: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """预测单张图像

        Args:
            img: 输入图像 (H, W, C) BGR numpy 或 (C, H, W) tensor
            return_vis: 是否返回可视化结果

        Returns:
            bboxes: (N, 4) [x1, y1, x2, y2]
            scores: (N,)
            keypoints: (N, K, 3) [x, y, conf]
            vis_img: 可视化图像 (如果 return_vis=True)
        """
        # 保存原始图像信息
        if isinstance(img, np.ndarray):
            orig_img = img.copy()
            orig_h, orig_w = img.shape[:2]

            # 预处理
            img_resized, ratio, pad = letterbox(img, self.input_size)
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
        else:
            orig_img = None
            orig_h, orig_w = img.shape[-2:]
            img_tensor = img.unsqueeze(0) if img.dim() == 3 else img
            img_tensor = img_tensor.to(self.device)
            ratio, pad = 1.0, (0, 0)

        # 推理
        outputs = self.model(img_tensor)

        # 解码
        bboxes, scores, keypoints = self.model.decode(
            outputs, self.conf_thresh, self.input_size
        )

        # 后处理 (NMS)
        bboxes, scores, keypoints = postprocess(
            bboxes, scores, keypoints,
            conf_thresh=self.conf_thresh,
            iou_thresh=self.iou_thresh
        )

        # 坐标缩放回原图
        if ratio != 1.0 or pad != (0, 0):
            bboxes = scale_coords(bboxes, self.input_size, (orig_h, orig_w), (ratio, pad))
            keypoints[..., :2] = scale_coords(
                keypoints[..., :2], self.input_size, (orig_h, orig_w), (ratio, pad)
            )

        # 裁剪到图像范围
        bboxes = clip_coords(bboxes, (orig_h, orig_w))
        keypoints[..., :2] = clip_coords(keypoints[..., :2], (orig_h, orig_w))

        # 转为 numpy
        bboxes = bboxes.cpu().numpy()
        scores = scores.cpu().numpy()
        keypoints = keypoints.cpu().numpy()

        # 可视化
        vis_img = None
        if return_vis and orig_img is not None:
            vis_img = draw_pose(orig_img, keypoints, scores, bboxes)

        return bboxes, scores, keypoints, vis_img

    def predict_batch(
        self,
        imgs: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """批量预测

        Args:
            imgs: 图像列表

        Returns:
            结果列表 [(bboxes, scores, keypoints), ...]
        """
        results = []
        for img in imgs:
            bboxes, scores, keypoints, _ = self.predict(img)
            results.append((bboxes, scores, keypoints))
        return results


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """计算 IoU

    Args:
        box1: (N, 4) [x1, y1, x2, y2]
        box2: (M, 4) [x1, y1, x2, y2]

    Returns:
        iou: (N, M) IoU 矩阵
    """
    # 扩展维度用于广播
    box1 = box1[:, np.newaxis, :]  # (N, 1, 4)
    box2 = box2[np.newaxis, :, :]  # (1, M, 4)

    # 交集
    inter_x1 = np.maximum(box1[..., 0], box2[..., 0])
    inter_y1 = np.maximum(box1[..., 1], box2[..., 1])
    inter_x2 = np.minimum(box1[..., 2], box2[..., 2])
    inter_y2 = np.minimum(box1[..., 3], box2[..., 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # 并集
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-7)


def match_predictions_to_gt(
    pred_boxes: np.ndarray,
    pred_keypoints: np.ndarray,
    gt_boxes: np.ndarray,
    gt_keypoints: np.ndarray,
    iou_thresh: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将预测与 GT 匹配

    Args:
        pred_boxes: (N, 4) 预测框 [x1, y1, x2, y2]
        pred_keypoints: (N, K, 3) 预测关键点 [x, y, conf]
        gt_boxes: (M, 4) GT 框 [x1, y1, x2, y2]
        gt_keypoints: (M, K, 3) GT 关键点 [x, y, vis]
        iou_thresh: IoU 阈值

    Returns:
        matched_pred_kpts: (P, K, 2) 匹配的预测关键点
        matched_gt_kpts: (P, K, 2) 匹配的 GT 关键点
        matched_gt_vis: (P, K) 匹配的可见性
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.empty((0, pred_keypoints.shape[1], 2)), \
               np.empty((0, gt_keypoints.shape[1], 2)), \
               np.empty((0, gt_keypoints.shape[1]))

    # 计算 IoU 矩阵
    iou_matrix = compute_iou(pred_boxes, gt_boxes)  # (N, M)

    # 贪婪匹配
    matched_pred_idx = []
    matched_gt_idx = []
    used_gt = set()

    # 按最大 IoU 排序
    while True:
        max_iou = iou_matrix.max()
        if max_iou < iou_thresh:
            break

        pred_idx, gt_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)

        if gt_idx not in used_gt:
            matched_pred_idx.append(pred_idx)
            matched_gt_idx.append(gt_idx)
            used_gt.add(gt_idx)

        # 将已匹配的设为 0
        iou_matrix[pred_idx, :] = 0
        iou_matrix[:, gt_idx] = 0

    if len(matched_pred_idx) == 0:
        return np.empty((0, pred_keypoints.shape[1], 2)), \
               np.empty((0, gt_keypoints.shape[1], 2)), \
               np.empty((0, gt_keypoints.shape[1]))

    matched_pred_kpts = pred_keypoints[matched_pred_idx, :, :2]
    matched_gt_kpts = gt_keypoints[matched_gt_idx, :, :2]
    matched_gt_vis = gt_keypoints[matched_gt_idx, :, 2]

    return matched_pred_kpts, matched_gt_kpts, matched_gt_vis


class YOLOPoseEvaluator:
    """YOLO-Pose 多人姿态评估器"""

    # COCO 关键点 sigma 值
    COCO_SIGMAS = np.array([
        0.026, 0.025, 0.025, 0.035, 0.035,
        0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087,
        0.089, 0.089
    ])

    def __init__(
        self,
        num_keypoints: int = 17,
        iou_thresh: float = 0.5,
        pck_thresh: float = 0.2
    ):
        """
        Args:
            num_keypoints: 关键点数量
            iou_thresh: 预测与 GT 匹配的 IoU 阈值
            pck_thresh: PCK 计算的阈值
        """
        self.num_keypoints = num_keypoints
        self.iou_thresh = iou_thresh
        self.pck_thresh = pck_thresh
        self.sigmas = self.COCO_SIGMAS[:num_keypoints]
        self.reset()

    def reset(self):
        """重置累积状态"""
        self.all_pred_kpts = []
        self.all_gt_kpts = []
        self.all_gt_vis = []
        self.all_gt_areas = []
        self.num_gt = 0
        self.num_pred = 0
        self.num_matched = 0

    def update(
        self,
        pred_boxes: np.ndarray,
        pred_keypoints: np.ndarray,
        gt_boxes: np.ndarray,
        gt_keypoints: np.ndarray
    ):
        """更新评估状态

        Args:
            pred_boxes: (N, 4) 预测框 [x1, y1, x2, y2]
            pred_keypoints: (N, K, 3) 预测关键点 [x, y, conf]
            gt_boxes: (M, 4) GT 框 [x1, y1, x2, y2]
            gt_keypoints: (M, K, 3) GT 关键点 [x, y, vis]
        """
        self.num_gt += len(gt_boxes)
        self.num_pred += len(pred_boxes)

        # 匹配预测与 GT
        matched_pred, matched_gt, matched_vis = match_predictions_to_gt(
            pred_boxes, pred_keypoints,
            gt_boxes, gt_keypoints,
            self.iou_thresh
        )

        self.num_matched += len(matched_pred)

        if len(matched_pred) > 0:
            self.all_pred_kpts.append(matched_pred)
            self.all_gt_kpts.append(matched_gt)
            self.all_gt_vis.append(matched_vis)

            # 计算 GT 面积（用于 OKS）
            for i in range(len(matched_gt)):
                vis_mask = matched_vis[i] > 0
                if vis_mask.sum() > 0:
                    kpts = matched_gt[i][vis_mask]
                    area = (kpts[:, 0].max() - kpts[:, 0].min()) * \
                           (kpts[:, 1].max() - kpts[:, 1].min())
                else:
                    area = 1.0
                self.all_gt_areas.append(max(area, 1.0))

    def compute(self) -> Dict:
        """计算所有评估指标

        Returns:
            metrics: 指标字典
        """
        metrics = {
            'num_gt': self.num_gt,
            'num_pred': self.num_pred,
            'num_matched': self.num_matched
        }

        if len(self.all_pred_kpts) == 0:
            metrics['PCK'] = 0.0
            metrics['PCK@0.1'] = 0.0
            metrics['PCK@0.2'] = 0.0
            metrics['OKS'] = 0.0
            metrics['AP'] = 0.0
            metrics['AP50'] = 0.0
            metrics['AP75'] = 0.0
            return metrics

        # 合并所有匹配
        pred_kpts = np.concatenate(self.all_pred_kpts, axis=0)  # (P, K, 2)
        gt_kpts = np.concatenate(self.all_gt_kpts, axis=0)      # (P, K, 2)
        gt_vis = np.concatenate(self.all_gt_vis, axis=0)        # (P, K)
        gt_areas = np.array(self.all_gt_areas)                   # (P,)

        # 计算距离
        dist = np.linalg.norm(pred_kpts - gt_kpts, axis=2)  # (P, K)

        # 计算归一化因子（使用 bbox 对角线）
        bbox_diag = np.sqrt(gt_areas)  # 近似对角线
        bbox_diag = np.maximum(bbox_diag, 1.0)[:, np.newaxis]

        # PCK: 归一化距离 < 阈值
        normalized_dist = dist / bbox_diag

        # 只考虑可见关键点
        visible_mask = gt_vis > 0

        # PCK@0.1
        pck_01 = ((normalized_dist < 0.1) & visible_mask).sum() / max(visible_mask.sum(), 1)
        metrics['PCK@0.1'] = float(pck_01)

        # PCK@0.2
        pck_02 = ((normalized_dist < 0.2) & visible_mask).sum() / max(visible_mask.sum(), 1)
        metrics['PCK@0.2'] = float(pck_02)

        # 默认 PCK（使用 pck_thresh）
        pck = ((normalized_dist < self.pck_thresh) & visible_mask).sum() / max(visible_mask.sum(), 1)
        metrics['PCK'] = float(pck)

        # OKS
        kappa = 2 * self.sigmas ** 2
        oks_per_kpt = np.exp(-dist ** 2 / (2 * gt_areas[:, np.newaxis] * kappa + 1e-7))
        oks_per_kpt = oks_per_kpt * gt_vis  # 只考虑可见点

        num_visible = gt_vis.sum(axis=1).clip(min=1)
        oks_per_instance = oks_per_kpt.sum(axis=1) / num_visible

        metrics['OKS'] = float(oks_per_instance.mean())

        # AP (基于 OKS)
        thresholds = np.arange(0.5, 1.0, 0.05)
        ap_per_thresh = []
        for thresh in thresholds:
            ap_per_thresh.append((oks_per_instance >= thresh).mean())

        metrics['AP'] = float(np.mean(ap_per_thresh))
        metrics['AP50'] = float((oks_per_instance >= 0.5).mean())
        metrics['AP75'] = float((oks_per_instance >= 0.75).mean())

        return metrics
