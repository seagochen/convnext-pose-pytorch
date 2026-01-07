"""
姿态检测数据增强

包含常用的数据增强方法，保证关键点坐标与图像同步变换
"""

import cv2
import numpy as np
import random
import torch
from typing import Tuple, Optional, List


class PoseTransforms:
    """姿态检测数据增强类

    Args:
        input_size: 输入图像大小 (height, width)
        output_size: 输出热图大小 (height, width)
        scale_factor: 缩放因子范围
        rotation_factor: 旋转角度范围
        flip_prob: 水平翻转概率
        half_body_prob: 半身增强概率
        color_jitter: 是否使用颜色抖动
        flip_idx: 关键点翻转索引映射
        train: 是否为训练模式
    """
    def __init__(self,
                 input_size: Tuple[int, int] = (256, 192),
                 output_size: Tuple[int, int] = (64, 48),
                 scale_factor: float = 0.35,
                 rotation_factor: float = 45,
                 flip_prob: float = 0.5,
                 half_body_prob: float = 0.3,
                 color_jitter: bool = True,
                 flip_idx: Optional[List[int]] = None,
                 train: bool = True):
        self.input_size = np.array(input_size)
        self.output_size = np.array(output_size)
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = flip_prob
        self.half_body_prob = half_body_prob
        self.color_jitter = color_jitter
        self.train = train

        # 关键点翻转映射 (COCO格式)
        if flip_idx is not None:
            self.flip_idx = flip_idx
        else:
            # 默认 COCO 17 关键点翻转映射
            self.flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

        # 关键点翻转配对 (COCO格式)
        self.flip_pairs = [
            [1, 2], [3, 4], [5, 6], [7, 8],
            [9, 10], [11, 12], [13, 14], [15, 16]
        ]

        # 上半身和下半身关键点索引
        self.upper_body_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.lower_body_ids = [11, 12, 13, 14, 15, 16]

        # 图像标准化参数 (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, image: np.ndarray, keypoints: np.ndarray,
                 bbox: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        """
        Args:
            image: numpy数组 (H, W, 3) BGR格式
            keypoints: numpy数组 (num_keypoints, 3) [x, y, visibility]
            bbox: 边界框 [x, y, w, h] (可选)

        Returns:
            image: 处理后的图像 tensor (3, H, W)
            keypoints: 处理后的关键点 (num_keypoints, 3)
            heatmap: 生成的热图 (num_keypoints, H', W')
        """
        keypoints = keypoints.copy()
        image = image.copy()

        # 计算中心和缩放
        if bbox is not None:
            center, scale = self._box_to_center_scale(bbox)
        else:
            center, scale = self._keypoints_to_center_scale(keypoints)

        # 训练时的数据增强
        if self.train:
            # 半身增强
            if random.random() < self.half_body_prob:
                c_half, s_half = self._half_body_transform(keypoints)
                if c_half is not None:
                    center, scale = c_half, s_half

            # 缩放增强
            sf = self.scale_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            # 旋转增强
            rf = self.rotation_factor
            rotation = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

            # 水平翻转
            if random.random() < self.flip_prob:
                image = image[:, ::-1, :]
                keypoints = self._flip_keypoints(keypoints, image.shape[1])
                center[0] = image.shape[1] - center[0] - 1
        else:
            rotation = 0

        # 仿射变换
        trans = self._get_affine_transform(center, scale, rotation, self.input_size)
        image = cv2.warpAffine(
            image,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR
        )

        # 变换关键点
        for i in range(len(keypoints)):
            if keypoints[i, 2] > 0:
                keypoints[i, :2] = self._affine_transform(keypoints[i, :2], trans)

        # 颜色增强
        if self.train and self.color_jitter:
            image = self._color_jitter(image)

        # 归一化
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1)  # HWC -> CHW

        # 生成热图
        heatmap = self._generate_heatmap(keypoints)

        return torch.from_numpy(image), keypoints, torch.from_numpy(heatmap)

    def _box_to_center_scale(self, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """从边界框计算中心和缩放"""
        x, y, w, h = bbox[:4]
        center = np.array([x + w / 2, y + h / 2], dtype=np.float32)

        # 保持宽高比
        aspect_ratio = self.input_size[1] / self.input_size[0]
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        scale = scale * 1.25  # 添加一些padding

        return center, scale

    def _keypoints_to_center_scale(self, keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """从关键点计算中心和缩放"""
        visible = keypoints[:, 2] > 0
        if not visible.any():
            return np.array([0, 0]), np.array([1, 1])

        kpts = keypoints[visible, :2]
        min_pt = kpts.min(axis=0)
        max_pt = kpts.max(axis=0)

        center = (min_pt + max_pt) / 2
        scale = (max_pt - min_pt) / 200.0
        scale = np.maximum(scale, [0.5, 0.5])
        scale = scale * 1.5

        return center, scale

    def _half_body_transform(self, keypoints: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """半身数据增强"""
        upper_kpts = []
        lower_kpts = []

        for i in range(len(keypoints)):
            if keypoints[i, 2] > 0:
                if i in self.upper_body_ids:
                    upper_kpts.append(keypoints[i, :2])
                else:
                    lower_kpts.append(keypoints[i, :2])

        # 随机选择上半身或下半身
        if random.random() < 0.5 and len(upper_kpts) > 2:
            selected_kpts = upper_kpts
        elif len(lower_kpts) > 2:
            selected_kpts = lower_kpts
        else:
            return None, None

        selected_kpts = np.array(selected_kpts, dtype=np.float32)
        center = selected_kpts.mean(axis=0)

        min_pt = selected_kpts.min(axis=0)
        max_pt = selected_kpts.max(axis=0)
        scale = (max_pt - min_pt) / 200.0
        scale = np.maximum(scale, [0.5, 0.5])
        scale = scale * 1.5

        return center, scale

    def _flip_keypoints(self, keypoints: np.ndarray, width: int) -> np.ndarray:
        """水平翻转关键点"""
        keypoints[:, 0] = width - keypoints[:, 0] - 1

        # 使用翻转索引重新排列关键点
        keypoints = keypoints[self.flip_idx]

        return keypoints

    def _get_affine_transform(self, center: np.ndarray, scale: np.ndarray,
                              rot: float, output_size: np.ndarray) -> np.ndarray:
        """计算仿射变换矩阵"""
        scale_tmp = scale * 200.0

        src_w = scale_tmp[0]
        dst_w = output_size[1]
        dst_h = output_size[0]

        rot_rad = np.pi * rot / 180
        src_dir = self._get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans

    def _get_dir(self, src_point: list, rot_rad: float) -> list:
        """旋转方向向量"""
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs
        return src_result

    def _get_3rd_point(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """获取第三个点（垂直方向）"""
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def _affine_transform(self, pt: np.ndarray, trans: np.ndarray) -> np.ndarray:
        """对点应用仿射变换"""
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
        new_pt = np.dot(trans, new_pt)
        return new_pt[:2]

    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        """颜色抖动增强"""
        # 亮度
        if random.random() < 0.5:
            delta = random.uniform(-32, 32)
            image = np.clip(image + delta, 0, 255)

        # 对比度
        if random.random() < 0.5:
            alpha = random.uniform(0.5, 1.5)
            image = np.clip(image * alpha, 0, 255)

        # 饱和度
        if random.random() < 0.5:
            image_hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)
            image_hsv[:, :, 1] = np.clip(
                image_hsv[:, :, 1] * random.uniform(0.5, 1.5), 0, 255
            )
            image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

        return image

    def _generate_heatmap(self, keypoints: np.ndarray, sigma: float = 2) -> np.ndarray:
        """生成高斯热图

        Args:
            keypoints: (num_keypoints, 3)
            sigma: 高斯核标准差

        Returns:
            heatmap: (num_keypoints, H, W)
        """
        num_keypoints = len(keypoints)
        heatmap = np.zeros((num_keypoints, self.output_size[0], self.output_size[1]),
                          dtype=np.float32)

        # 计算输入到输出的缩放比例
        stride = self.input_size / self.output_size

        for i, kpt in enumerate(keypoints):
            if kpt[2] <= 0:  # 不可见
                continue

            # 将关键点坐标映射到输出尺度
            mu_x = int(kpt[0] / stride[1] + 0.5)
            mu_y = int(kpt[1] / stride[0] + 0.5)

            # 检查是否在范围内
            if mu_x < 0 or mu_y < 0 or \
               mu_x >= self.output_size[1] or mu_y >= self.output_size[0]:
                continue

            # 生成高斯核
            tmp_size = sigma * 3
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            # 检查边界
            if ul[0] >= self.output_size[1] or ul[1] >= self.output_size[0] \
               or br[0] < 0 or br[1] < 0:
                continue

            # 生成高斯
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # 裁剪边界
            g_x = max(0, -ul[0]), min(br[0], self.output_size[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.output_size[0]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self.output_size[1])
            img_y = max(0, ul[1]), min(br[1], self.output_size[0])

            heatmap[i][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return heatmap


def get_transforms(input_size: Tuple[int, int] = (256, 192),
                   output_size: Tuple[int, int] = (64, 48),
                   scale_factor: float = 0.35,
                   rotation_factor: float = 45,
                   flip_prob: float = 0.5,
                   flip_idx: Optional[List[int]] = None,
                   train: bool = True) -> PoseTransforms:
    """获取数据增强实例

    Args:
        input_size: 输入图像大小
        output_size: 输出热图大小
        scale_factor: 缩放因子
        rotation_factor: 旋转因子
        flip_prob: 翻转概率
        flip_idx: 翻转索引
        train: 是否为训练模式

    Returns:
        PoseTransforms 实例
    """
    if train:
        return PoseTransforms(
            input_size=input_size,
            output_size=output_size,
            scale_factor=scale_factor,
            rotation_factor=rotation_factor,
            flip_prob=flip_prob,
            half_body_prob=0.3,
            color_jitter=True,
            flip_idx=flip_idx,
            train=True
        )
    else:
        return PoseTransforms(
            input_size=input_size,
            output_size=output_size,
            flip_idx=flip_idx,
            train=False
        )
