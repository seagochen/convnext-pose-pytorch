#!/usr/bin/env python3
"""
数据集检测脚本

检测 COCO Pose 数据集中的脏数据：
- NaN/Inf 值
- 无效的边界框（宽高为0或负数）
- 超出范围的坐标
- 空标签文件
- 损坏的图像
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def check_label_file(label_path: Path, num_keypoints: int = 17) -> dict:
    """检查单个标签文件

    Returns:
        dict with issues found
    """
    issues = {
        'empty': False,
        'invalid_bbox': [],
        'out_of_range': [],
        'nan_inf': [],
        'zero_wh': [],
        'parse_error': None
    }

    if not label_path.exists():
        issues['empty'] = True
        return issues

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            issues['empty'] = True
            return issues

        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            values = list(map(float, line.split()))

            # 检查 NaN/Inf
            if any(np.isnan(v) or np.isinf(v) for v in values):
                issues['nan_inf'].append(line_idx)
                continue

            if len(values) < 5:
                issues['invalid_bbox'].append((line_idx, 'too few values'))
                continue

            class_id, cx, cy, w, h = values[:5]

            # 检查边界框
            if w <= 0 or h <= 0:
                issues['zero_wh'].append((line_idx, f'w={w:.4f}, h={h:.4f}'))

            # 检查坐标范围
            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                issues['out_of_range'].append((line_idx, f'center: ({cx:.4f}, {cy:.4f})'))

            if w > 1 or h > 1:
                issues['out_of_range'].append((line_idx, f'size: ({w:.4f}, {h:.4f})'))

            # 检查关键点
            kpt_offset = 5
            for i in range(num_keypoints):
                base_idx = kpt_offset + i * 3
                if base_idx + 2 < len(values):
                    kx, ky, kv = values[base_idx:base_idx+3]

                    if kv > 0:  # 只检查可见关键点
                        if not (-0.5 <= kx <= 1.5 and -0.5 <= ky <= 1.5):
                            issues['out_of_range'].append(
                                (line_idx, f'kpt{i}: ({kx:.4f}, {ky:.4f})')
                            )

    except Exception as e:
        issues['parse_error'] = str(e)

    return issues


def check_image_file(img_path: Path) -> dict:
    """检查单个图像文件

    Returns:
        dict with issues found
    """
    issues = {
        'not_found': False,
        'corrupted': False,
        'error': None
    }

    if not img_path.exists():
        issues['not_found'] = True
        return issues

    try:
        img = cv2.imread(str(img_path))
        if img is None:
            issues['corrupted'] = True
    except Exception as e:
        issues['error'] = str(e)

    return issues


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Check COCO Pose dataset for dirty data')
    parser.add_argument('--data', type=str,
                        default='/home/cxt/datasets/coco_pose_yolo/data.yaml',
                        help='Path to data.yaml')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to check')
    parser.add_argument('--check-images', action='store_true',
                        help='Also check image files (slower)')
    parser.add_argument('--fix', action='store_true',
                        help='Try to fix issues (remove problematic samples)')
    args = parser.parse_args()

    import yaml
    with open(args.data, 'r') as f:
        config = yaml.safe_load(f)

    data_root = Path(config.get('path', Path(args.data).parent))

    # 支持不同的目录结构
    # 结构1: data_root/train/images, data_root/train/labels
    # 结构2: data_root/images/train, data_root/labels/train
    split_path_1 = data_root / args.split / 'images'
    split_path_2 = data_root / 'images' / args.split

    if split_path_1.exists():
        split_path = split_path_1
        label_dir = data_root / args.split / 'labels'
    elif split_path_2.exists():
        split_path = split_path_2
        label_dir = data_root / 'labels' / args.split
    else:
        print(f"Error: Cannot find image directory for split '{args.split}'")
        print(f"  Tried: {split_path_1}")
        print(f"  Tried: {split_path_2}")
        return

    # 获取所有图像文件
    img_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = []
    for fmt in img_formats:
        image_files.extend(split_path.glob(f'*{fmt}'))

    print(f"Found {len(image_files)} images in {split_path}")
    print(f"Label dir: {label_dir}")
    print()

    # 统计问题
    total_issues = {
        'empty_labels': [],
        'invalid_bbox': [],
        'out_of_range': [],
        'nan_inf': [],
        'zero_wh': [],
        'parse_error': [],
        'image_not_found': [],
        'image_corrupted': [],
        'image_error': []
    }

    for img_path in tqdm(image_files, desc='Checking'):
        # 获取标签路径
        label_path = label_dir / (img_path.stem + '.txt')

        # 检查标签
        label_issues = check_label_file(label_path)

        if label_issues['empty']:
            total_issues['empty_labels'].append(str(img_path))
        if label_issues['invalid_bbox']:
            total_issues['invalid_bbox'].append((str(img_path), label_issues['invalid_bbox']))
        if label_issues['out_of_range']:
            total_issues['out_of_range'].append((str(img_path), label_issues['out_of_range']))
        if label_issues['nan_inf']:
            total_issues['nan_inf'].append((str(img_path), label_issues['nan_inf']))
        if label_issues['zero_wh']:
            total_issues['zero_wh'].append((str(img_path), label_issues['zero_wh']))
        if label_issues['parse_error']:
            total_issues['parse_error'].append((str(img_path), label_issues['parse_error']))

        # 检查图像（可选）
        if args.check_images:
            img_issues = check_image_file(img_path)
            if img_issues['not_found']:
                total_issues['image_not_found'].append(str(img_path))
            if img_issues['corrupted']:
                total_issues['image_corrupted'].append(str(img_path))
            if img_issues['error']:
                total_issues['image_error'].append((str(img_path), img_issues['error']))

    # 打印结果
    print("\n" + "=" * 70)
    print("Dataset Check Results")
    print("=" * 70)

    print(f"\nTotal images checked: {len(image_files)}")
    print(f"\nIssues found:")
    print(f"  Empty labels:      {len(total_issues['empty_labels'])}")
    print(f"  Invalid bbox:      {len(total_issues['invalid_bbox'])}")
    print(f"  Out of range:      {len(total_issues['out_of_range'])}")
    print(f"  NaN/Inf values:    {len(total_issues['nan_inf'])}")
    print(f"  Zero width/height: {len(total_issues['zero_wh'])}")
    print(f"  Parse errors:      {len(total_issues['parse_error'])}")

    if args.check_images:
        print(f"  Image not found:   {len(total_issues['image_not_found'])}")
        print(f"  Image corrupted:   {len(total_issues['image_corrupted'])}")
        print(f"  Image errors:      {len(total_issues['image_error'])}")

    # 显示详细信息
    if total_issues['nan_inf']:
        print(f"\n--- NaN/Inf samples (first 10) ---")
        for item in total_issues['nan_inf'][:10]:
            print(f"  {item}")

    if total_issues['zero_wh']:
        print(f"\n--- Zero width/height samples (first 10) ---")
        for item in total_issues['zero_wh'][:10]:
            print(f"  {item[0]}: {item[1]}")

    if total_issues['out_of_range']:
        print(f"\n--- Out of range samples (first 10) ---")
        for item in total_issues['out_of_range'][:10]:
            print(f"  {item[0]}: {item[1][:3]}...")

    if total_issues['parse_error']:
        print(f"\n--- Parse error samples (first 10) ---")
        for item in total_issues['parse_error'][:10]:
            print(f"  {item[0]}: {item[1]}")

    # 收集所有有问题的样本
    problematic_samples = set()
    for key in ['empty_labels', 'invalid_bbox', 'nan_inf', 'zero_wh', 'parse_error']:
        if key == 'empty_labels':
            problematic_samples.update(total_issues[key])
        else:
            for item in total_issues[key]:
                if isinstance(item, tuple):
                    problematic_samples.add(item[0])
                else:
                    problematic_samples.add(item)

    print(f"\n--- Summary ---")
    print(f"Total problematic samples: {len(problematic_samples)}")

    # 保存问题样本列表
    if problematic_samples:
        output_file = Path(args.data).parent / f'problematic_samples_{args.split}.txt'
        with open(output_file, 'w') as f:
            for sample in sorted(problematic_samples):
                f.write(sample + '\n')
        print(f"Saved problematic samples list to: {output_file}")

    print("=" * 70)


if __name__ == '__main__':
    main()
