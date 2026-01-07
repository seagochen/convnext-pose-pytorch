"""
路径处理工具

用于解析 YOLO 格式数据集的路径
"""

import os
from pathlib import Path
from typing import Union, Optional


def resolve_split_path(data_root: Union[str, Path], split_path: str) -> Path:
    """解析数据集分割路径

    支持多种路径格式:
    - 绝对路径: /path/to/images
    - 相对路径: images/train
    - Roboflow 格式: ../train/images

    Args:
        data_root: 数据集根目录
        split_path: 分割路径 (可以是相对或绝对路径)

    Returns:
        解析后的完整路径
    """
    data_root = Path(data_root)
    split_path = str(split_path)

    # 处理绝对路径
    if os.path.isabs(split_path):
        return Path(split_path)

    # 处理 Roboflow 格式的 ../ 前缀
    if split_path.startswith('../'):
        # 假设 yaml 文件在数据集根目录
        return (data_root.parent / split_path).resolve()

    # 标准相对路径
    return data_root / split_path


def infer_label_dir(image_dir: Union[str, Path]) -> Path:
    """从图像目录推断标签目录

    支持多种目录结构:
    1. images/train -> labels/train
    2. train/images -> train/labels
    3. 并行目录结构

    Args:
        image_dir: 图像目录路径

    Returns:
        推断的标签目录路径
    """
    image_dir = Path(image_dir)

    # 模式1: images/train -> labels/train
    if 'images' in image_dir.parts:
        idx = image_dir.parts.index('images')
        parts = list(image_dir.parts)
        parts[idx] = 'labels'
        return Path(*parts)

    # 模式2: train/images -> train/labels
    if image_dir.name == 'images':
        return image_dir.parent / 'labels'

    # 默认: 同级 labels 目录
    return image_dir.parent / 'labels' / image_dir.name


def find_image_files(image_dir: Union[str, Path], extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')) -> list:
    """查找目录中的所有图像文件

    Args:
        image_dir: 图像目录
        extensions: 支持的图像扩展名

    Returns:
        图像文件路径列表
    """
    image_dir = Path(image_dir)
    images = []

    for ext in extensions:
        images.extend(image_dir.glob(f'*{ext}'))
        images.extend(image_dir.glob(f'*{ext.upper()}'))

    return sorted(images)


def get_label_path(image_path: Union[str, Path], label_dir: Union[str, Path]) -> Path:
    """获取图像对应的标签文件路径

    Args:
        image_path: 图像文件路径
        label_dir: 标签目录

    Returns:
        标签文件路径
    """
    image_path = Path(image_path)
    label_dir = Path(label_dir)

    # 替换扩展名为 .txt
    label_name = image_path.stem + '.txt'
    return label_dir / label_name


def verify_dataset_structure(data_yaml_path: Union[str, Path]) -> dict:
    """验证数据集结构

    Args:
        data_yaml_path: 数据集配置文件路径

    Returns:
        验证结果字典
    """
    import yaml

    data_yaml_path = Path(data_yaml_path)

    if not data_yaml_path.exists():
        return {'valid': False, 'error': f'Config file not found: {data_yaml_path}'}

    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    result = {
        'valid': True,
        'config': config,
        'train_images': 0,
        'val_images': 0,
        'train_labels': 0,
        'val_labels': 0,
        'warnings': []
    }

    # 获取数据集根目录
    data_root = Path(config.get('path', data_yaml_path.parent))
    if not data_root.is_absolute():
        data_root = data_yaml_path.parent / data_root

    # 检查训练集
    if 'train' in config:
        train_img_dir = resolve_split_path(data_root, config['train'])
        if train_img_dir.exists():
            train_images = find_image_files(train_img_dir)
            result['train_images'] = len(train_images)

            train_label_dir = infer_label_dir(train_img_dir)
            if train_label_dir.exists():
                result['train_labels'] = len(list(train_label_dir.glob('*.txt')))
            else:
                result['warnings'].append(f'Train label dir not found: {train_label_dir}')
        else:
            result['warnings'].append(f'Train image dir not found: {train_img_dir}')

    # 检查验证集
    if 'val' in config:
        val_img_dir = resolve_split_path(data_root, config['val'])
        if val_img_dir.exists():
            val_images = find_image_files(val_img_dir)
            result['val_images'] = len(val_images)

            val_label_dir = infer_label_dir(val_img_dir)
            if val_label_dir.exists():
                result['val_labels'] = len(list(val_label_dir.glob('*.txt')))
            else:
                result['warnings'].append(f'Val label dir not found: {val_label_dir}')
        else:
            result['warnings'].append(f'Val image dir not found: {val_img_dir}')

    # 检查关键点配置
    if 'kpt_shape' not in config:
        result['warnings'].append('kpt_shape not specified in config')

    if result['warnings']:
        result['valid'] = len(result['warnings']) < 3  # 允许一些警告

    return result
