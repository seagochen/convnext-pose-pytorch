"""
YOLO-Pose 多人姿态检测数据集

支持 YOLO Pose 格式的标签文件:
- 每行格式: class_id x_center y_center width height kpt1_x kpt1_y kpt1_v kpt2_x kpt2_y kpt2_v ...
- 所有坐标归一化到 [0, 1]
- 可见性: 0=不存在, 1=遮挡, 2=可见
- 支持每张图片多人标注
"""

import cv2
import yaml
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict

from .path_utils import resolve_split_path, infer_label_dir, find_image_files, get_label_path


# 默认配置
MAX_PERSONS = 20  # 每张图最多处理的人数


class YOLOPoseDataset(Dataset):
    """YOLO-Pose 多人姿态估计数据集

    Args:
        data_yaml: 数据集配置文件路径
        split: 数据集分割 ('train' 或 'val')
        input_size: 输入图像大小 (height, width)
        num_keypoints: 关键点数量
        max_persons: 每张图最多处理的人数
        augment: 是否启用数据增强 (训练时默认True)
        cache_images: 是否缓存图像到内存
    """

    # 支持的图像格式
    IMG_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')

    def __init__(self,
                 data_yaml: str,
                 split: str = 'train',
                 input_size: Tuple[int, int] = (640, 640),
                 num_keypoints: int = 17,
                 max_persons: int = MAX_PERSONS,
                 augment: bool = True,
                 cache_images: bool = False):
        super().__init__()

        self.data_yaml = Path(data_yaml)
        self.split = split
        self.input_size = input_size  # (height, width)
        self.num_keypoints = num_keypoints
        self.max_persons = max_persons
        self.augment = augment and (split == 'train')
        self.cache_images = cache_images

        # 加载数据集配置
        self.config = self._load_config()

        # 从配置覆盖关键点数量
        if 'kpt_shape' in self.config:
            self.num_keypoints = self.config['kpt_shape'][0]

        # 获取图像和标签路径
        self._setup_paths()

        # 加载样本列表
        self.samples = self._load_samples()

        # 图像缓存
        self.cached_images = {}
        if cache_images:
            self._cache_images()

        print(f"Loaded {len(self.samples)} samples from {split} split")
        print(f"  Max persons per image: {self.max_persons}")
        print(f"  Num keypoints: {self.num_keypoints}")

    def _load_config(self) -> dict:
        """加载数据集配置文件"""
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Config file not found: {self.data_yaml}")

        with open(self.data_yaml, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _setup_paths(self):
        """设置图像和标签目录路径"""
        # 数据集根目录
        data_root = self.config.get('path', self.data_yaml.parent)
        if not Path(data_root).is_absolute():
            data_root = self.data_yaml.parent / data_root
        self.data_root = Path(data_root)

        # 获取分割路径
        split_path = self.config.get(self.split)
        if split_path is None:
            raise ValueError(f"Split '{self.split}' not found in config")

        # 解析图像和标签目录
        self.img_dir = resolve_split_path(self.data_root, split_path)
        self.label_dir = infer_label_dir(self.img_dir)

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

    def _load_samples(self) -> List[Tuple[Path, Path]]:
        """加载所有样本的图像和标签路径"""
        samples = []

        # 查找所有图像文件
        image_files = find_image_files(self.img_dir, self.IMG_FORMATS)

        for img_path in image_files:
            label_path = get_label_path(img_path, self.label_dir)

            # 检查标签文件是否存在
            if label_path.exists():
                samples.append((img_path, label_path))

        if len(samples) == 0:
            print(f"Warning: No valid samples found in {self.img_dir}")

        return samples

    def _cache_images(self):
        """将图像缓存到内存"""
        print(f"Caching {len(self.samples)} images...")
        for idx, (img_path, _) in enumerate(self.samples):
            self.cached_images[idx] = cv2.imread(str(img_path))
            if (idx + 1) % 1000 == 0:
                print(f"  Cached {idx + 1}/{len(self.samples)} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """获取单个样本

        Returns:
            image: 处理后的图像 (3, H, W)
            target: 目标字典，包含:
                - bboxes: 边界框 (max_persons, 4) [cx, cy, w, h] 归一化
                - keypoints: 关键点 (max_persons, num_keypoints, 3) [dx, dy, v] 相对bbox中心偏移
                - num_persons: 实际人数
                - img_path: 图像路径
        """
        img_path, label_path = self.samples[idx]

        # 加载图像
        if self.cache_images and idx in self.cached_images:
            image = self.cached_images[idx].copy()
        else:
            image = cv2.imread(str(img_path))

        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        img_h, img_w = image.shape[:2]

        # 解析多人标签
        bboxes, keypoints, num_persons = self._parse_label_multiperson(label_path)

        # 调整图像大小并转换格式
        image = self._preprocess_image(image)

        # 编码关键点为相对于 bbox 中心的偏移
        keypoints_encoded = self._encode_keypoints(bboxes, keypoints, num_persons)

        target = {
            'bboxes': torch.from_numpy(bboxes),  # (max_persons, 4)
            'keypoints': torch.from_numpy(keypoints_encoded),  # (max_persons, num_keypoints, 3)
            'num_persons': num_persons,
            'img_path': str(img_path)
        }

        return image, target

    def _parse_label_multiperson(self, label_path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
        """解析多人 YOLO Pose 格式标签文件

        格式: class_id x_center y_center width height kpt1_x kpt1_y kpt1_v ...

        Returns:
            bboxes: (max_persons, 4) [cx, cy, w, h] 归一化坐标
            keypoints: (max_persons, num_keypoints, 3) [x, y, v] 归一化坐标
            num_persons: 实际人数
        """
        bboxes = np.zeros((self.max_persons, 4), dtype=np.float32)
        keypoints = np.zeros((self.max_persons, self.num_keypoints, 3), dtype=np.float32)
        num_persons = 0

        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if num_persons >= self.max_persons:
                    break

                values = list(map(float, line.split()))

                # 解析边界框 (保持归一化)
                if len(values) >= 5:
                    class_id = int(values[0])
                    cx = values[1]  # 中心 x (归一化)
                    cy = values[2]  # 中心 y (归一化)
                    w = values[3]   # 宽度 (归一化)
                    h = values[4]   # 高度 (归一化)

                    bboxes[num_persons] = [cx, cy, w, h]

                # 解析关键点 (保持归一化)
                kpt_offset = 5
                for i in range(self.num_keypoints):
                    base_idx = kpt_offset + i * 3
                    if base_idx + 2 < len(values):
                        kx = values[base_idx]      # x (归一化)
                        ky = values[base_idx + 1]  # y (归一化)
                        kv = values[base_idx + 2]  # 可见性
                        keypoints[num_persons, i] = [kx, ky, kv]

                num_persons += 1

        except Exception as e:
            print(f"Warning: Failed to parse label {label_path}: {e}")

        return bboxes, keypoints, num_persons

    def _encode_keypoints(self, bboxes: np.ndarray, keypoints: np.ndarray, num_persons: int) -> np.ndarray:
        """将关键点编码为相对于 bbox 中心的偏移

        Args:
            bboxes: (max_persons, 4) [cx, cy, w, h] 归一化
            keypoints: (max_persons, num_keypoints, 3) [x, y, v] 归一化绝对坐标

        Returns:
            encoded: (max_persons, num_keypoints, 3) [dx, dy, v] 相对偏移
        """
        encoded = np.zeros_like(keypoints)

        for i in range(num_persons):
            cx, cy, w, h = bboxes[i]

            for j in range(self.num_keypoints):
                kx, ky, kv = keypoints[i, j]

                if kv > 0:  # 只处理可见关键点
                    # 计算相对于 bbox 中心的偏移
                    # 偏移量归一化到 bbox 尺寸
                    dx = (kx - cx) / (w + 1e-6)
                    dy = (ky - cy) / (h + 1e-6)
                    encoded[i, j] = [dx, dy, kv]
                else:
                    encoded[i, j] = [0, 0, 0]

        return encoded

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像

        - 调整大小到 input_size
        - BGR -> RGB
        - 归一化到 [0, 1]
        - HWC -> CHW
        """
        # 调整大小 (letterbox 保持比例)
        image = self._letterbox(image, self.input_size)

        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 归一化
        image = image.astype(np.float32) / 255.0

        # HWC -> CHW
        image = image.transpose(2, 0, 1)

        return torch.from_numpy(image)

    def _letterbox(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Letterbox 调整大小，保持宽高比

        Args:
            image: 输入图像
            target_size: 目标大小 (height, width)

        Returns:
            调整后的图像
        """
        target_h, target_w = target_size
        h, w = image.shape[:2]

        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 调整大小
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建画布并居中放置
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        return canvas


def collate_fn_yolo(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, Dict]:
    """自定义 collate 函数，处理多人目标

    Args:
        batch: 批次数据列表

    Returns:
        images: (B, 3, H, W)
        targets: 字典，包含批次目标
    """
    images = []
    bboxes_list = []
    keypoints_list = []
    num_persons_list = []
    img_paths = []

    for image, target in batch:
        images.append(image)
        bboxes_list.append(target['bboxes'])
        keypoints_list.append(target['keypoints'])
        num_persons_list.append(target['num_persons'])
        img_paths.append(target['img_path'])

    # 堆叠图像
    images = torch.stack(images, dim=0)

    # 堆叠目标
    bboxes = torch.stack(bboxes_list, dim=0)  # (B, max_persons, 4)
    keypoints = torch.stack(keypoints_list, dim=0)  # (B, max_persons, num_keypoints, 3)
    num_persons = torch.tensor(num_persons_list, dtype=torch.long)  # (B,)

    targets = {
        'bboxes': bboxes,
        'keypoints': keypoints,
        'num_persons': num_persons,
        'img_paths': img_paths
    }

    return images, targets


def build_dataloader(data_yaml: str,
                     split: str = 'train',
                     input_size: Tuple[int, int] = (640, 640),
                     batch_size: int = 16,
                     num_workers: int = 4,
                     shuffle: bool = None,
                     pin_memory: bool = True,
                     drop_last: bool = None,
                     max_persons: int = MAX_PERSONS) -> DataLoader:
    """构建 YOLO-Pose 数据加载器

    Args:
        data_yaml: 数据集配置文件路径
        split: 数据集分割
        input_size: 输入图像大小 (height, width)
        batch_size: 批大小
        num_workers: 数据加载线程数
        shuffle: 是否打乱 (默认训练集打乱)
        pin_memory: 是否固定内存
        drop_last: 是否丢弃最后不完整批次 (默认训练集丢弃)
        max_persons: 每张图最多处理的人数

    Returns:
        DataLoader 实例
    """
    is_train = (split == 'train')

    if shuffle is None:
        shuffle = is_train
    if drop_last is None:
        drop_last = is_train

    dataset = YOLOPoseDataset(
        data_yaml=data_yaml,
        split=split,
        input_size=input_size,
        max_persons=max_persons,
        augment=is_train
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn_yolo
    )

    return dataloader
