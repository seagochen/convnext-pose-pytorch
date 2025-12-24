#!/usr/bin/env python3
"""
ConvNeXt-Pose 推理脚本

支持:
- 单张图像推理
- 视频推理
- 摄像头实时推理
- 可视化结果保存

用法:
    # 图像推理
    python scripts/detect.py --weights runs/train/best.pt --source image.jpg --show

    # 视频推理
    python scripts/detect.py --weights runs/train/best.pt --source video.mp4 --output result.mp4

    # 摄像头
    python scripts/detect.py --weights runs/train/best.pt --camera 0 --show
"""

import sys
import os
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from convnext_pose.models import ConvNeXtPose
from convnext_pose.utils.metrics import decode_heatmap, decode_heatmap_dark
from convnext_pose.utils.visualization import draw_pose, draw_heatmaps


class PoseEstimator:
    """姿态估计器

    Args:
        weights: 模型权重路径
        backbone: backbone类型
        head_type: 检测头类型
        device: 运行设备
        use_dark: 是否使用DARK解码
    """

    def __init__(self,
                 weights: str,
                 backbone: str = 'tiny',
                 head_type: str = 'heatmap',
                 num_keypoints: int = 17,
                 input_size: tuple = (256, 192),
                 device: str = 'cuda',
                 use_dark: bool = True):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.use_dark = use_dark
        self.num_keypoints = num_keypoints

        # 构建模型
        self.model = ConvNeXtPose(
            backbone=backbone,
            num_keypoints=num_keypoints,
            head_type=head_type
        )

        # 加载权重
        self._load_weights(weights)

        self.model = self.model.to(self.device)
        self.model.eval()

        # 图像预处理参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # 计算步长 (用于热图解码)
        self.stride = input_size[0] // 64  # 假设输出是输入的 1/4

        print(f"Model loaded from {weights}")
        print(f"Device: {self.device}")

    def _load_weights(self, weights: str):
        """加载模型权重"""
        state_dict = torch.load(weights, map_location='cpu')

        # 支持多种检查点格式
        if 'ema' in state_dict and state_dict['ema'] is not None:
            # 优先使用 EMA 权重
            model_dict = state_dict['ema'].get('ema', state_dict['ema'])
        elif 'model' in state_dict:
            model_dict = state_dict['model']
        else:
            model_dict = state_dict

        self.model.load_state_dict(model_dict)

    def preprocess(self, image: np.ndarray, bbox=None):
        """图像预处理

        Args:
            image: 输入图像 (H, W, 3) BGR格式
            bbox: 人物边界框 [x, y, w, h]

        Returns:
            input_tensor: 预处理后的tensor (1, 3, H, W)
            meta: 元信息用于后处理
        """
        orig_h, orig_w = image.shape[:2]

        if bbox is not None:
            x, y, w, h = bbox
            # 扩展边界框
            center_x, center_y = x + w / 2, y + h / 2
            scale = max(w, h) * 1.25

            # 裁剪区域
            x1 = int(center_x - scale / 2)
            y1 = int(center_y - scale / 2)
            x2 = int(center_x + scale / 2)
            y2 = int(center_y + scale / 2)

            # 边界检查和padding
            pad_left = max(0, -x1)
            pad_top = max(0, -y1)
            pad_right = max(0, x2 - orig_w)
            pad_bottom = max(0, y2 - orig_h)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_w, x2)
            y2 = min(orig_h, y2)

            # 裁剪并padding
            crop = image[y1:y2, x1:x2]
            if pad_left or pad_top or pad_right or pad_bottom:
                crop = cv2.copyMakeBorder(
                    crop, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )

            meta = {
                'center': np.array([center_x, center_y]),
                'scale': scale,
                'orig_size': (orig_h, orig_w)
            }
        else:
            crop = image
            meta = {
                'center': np.array([orig_w / 2, orig_h / 2]),
                'scale': max(orig_h, orig_w),
                'orig_size': (orig_h, orig_w)
            }

        # 调整大小
        input_img = cv2.resize(crop, (self.input_size[1], self.input_size[0]))

        # BGR -> RGB, 归一化
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = (input_img - self.mean) / self.std

        # 转换为tensor
        input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1))
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        meta['crop_size'] = crop.shape[:2]

        return input_tensor, meta

    def postprocess(self, heatmaps, meta):
        """后处理: 从热图解码关键点坐标

        Args:
            heatmaps: 热图 (1, K, H, W)
            meta: 预处理元信息

        Returns:
            keypoints: 关键点坐标 (K, 3) [x, y, score]
        """
        # 解码热图
        if self.use_dark:
            coords, scores = decode_heatmap_dark(heatmaps, stride=self.stride)
        else:
            coords, scores = decode_heatmap(heatmaps, stride=self.stride)

        coords = coords[0]  # (K, 2)
        scores = scores[0]  # (K,)

        # 映射回原图坐标
        h_ratio = meta['crop_size'][0] / self.input_size[0]
        w_ratio = meta['crop_size'][1] / self.input_size[1]

        coords[:, 0] = coords[:, 0] * w_ratio
        coords[:, 1] = coords[:, 1] * h_ratio

        # 转换到原图坐标系
        center = meta['center']

        coords[:, 0] = coords[:, 0] - meta['crop_size'][1] / 2 + center[0]
        coords[:, 1] = coords[:, 1] - meta['crop_size'][0] / 2 + center[1]

        # 组合坐标和得分
        keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)
        keypoints[:, :2] = coords
        keypoints[:, 2] = scores

        return keypoints

    @torch.no_grad()
    def predict(self, image: np.ndarray, bbox=None):
        """预测单张图像

        Args:
            image: 输入图像 (H, W, 3) BGR格式
            bbox: 人物边界框 [x, y, w, h] (可选)

        Returns:
            keypoints: 关键点 (K, 3) [x, y, score]
            heatmaps: 热图 (K, H, W)
        """
        # 预处理
        input_tensor, meta = self.preprocess(image, bbox)

        # 推理
        outputs = self.model(input_tensor)

        # 处理不同类型的输出
        if isinstance(outputs, tuple):
            # PAF模式
            _, heatmaps = outputs
            heatmaps = heatmaps[-1]
        else:
            heatmaps = outputs

        # 后处理
        keypoints = self.postprocess(heatmaps, meta)

        return keypoints, heatmaps[0].cpu().numpy()


def process_image(estimator: PoseEstimator,
                  image_path: str,
                  output_path: str = None,
                  show: bool = False,
                  save_heatmap: bool = False,
                  conf_thresh: float = 0.3):
    """处理单张图像"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    # 推理
    start = time.time()
    keypoints, heatmaps = estimator.predict(image)
    elapsed = time.time() - start

    print(f"Image: {image_path}")
    print(f"Inference time: {elapsed*1000:.1f}ms")

    # 过滤低置信度关键点
    visibility = (keypoints[:, 2] > conf_thresh).astype(np.float32)

    # 可视化
    vis = draw_pose(image, keypoints, visibility)

    # 保存结果
    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"Saved to {output_path}")

        # 保存热图
        if save_heatmap:
            heatmap_vis = draw_heatmaps(heatmaps, image, alpha=0.5)
            base, ext = os.path.splitext(output_path)
            heatmap_path = f"{base}_heatmap{ext}"
            cv2.imwrite(heatmap_path, heatmap_vis)
            print(f"Saved heatmap to {heatmap_path}")

    # 显示
    if show:
        cv2.imshow('Pose Estimation', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return keypoints


def process_video(estimator: PoseEstimator,
                  video_path: str,
                  output_path: str = None,
                  show: bool = True,
                  conf_thresh: float = 0.3):
    """处理视频"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")

    # 视频写入器
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 推理
        start = time.time()
        keypoints, _ = estimator.predict(frame)
        elapsed = time.time() - start
        total_time += elapsed

        # 过滤低置信度
        visibility = (keypoints[:, 2] > conf_thresh).astype(np.float32)

        # 可视化
        vis = draw_pose(frame, keypoints, visibility)

        # 显示FPS
        frame_count += 1
        avg_fps = frame_count / total_time
        cv2.putText(vis, f'FPS: {avg_fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示进度
        progress = frame_count / total_frames * 100
        cv2.putText(vis, f'{frame_count}/{total_frames} ({progress:.1f}%)', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 保存
        if writer:
            writer.write(vis)

        # 显示
        if show:
            cv2.imshow('Pose Estimation', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved to {output_path}")
    cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames, Average FPS: {avg_fps:.1f}")


def process_camera(estimator: PoseEstimator,
                   camera_id: int = 0,
                   output_path: str = None,
                   show: bool = True,
                   conf_thresh: float = 0.3):
    """处理摄像头"""
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Failed to open camera: {camera_id}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera: {camera_id}, Resolution: {width}x{height}")
    print("Press 'q' to quit, 's' to save snapshot")

    # 视频写入器
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    frame_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 推理
        start = time.time()
        keypoints, _ = estimator.predict(frame)
        elapsed = time.time() - start
        frame_times.append(elapsed)

        # 过滤低置信度
        visibility = (keypoints[:, 2] > conf_thresh).astype(np.float32)

        # 可视化
        vis = draw_pose(frame, keypoints, visibility)

        # 显示FPS (平滑)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_time = np.mean(frame_times)
        fps = 1.0 / avg_time

        cv2.putText(vis, f'FPS: {fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 保存
        if writer:
            writer.write(vis)

        # 显示
        if show:
            cv2.imshow('Pose Estimation', vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                snapshot_path = f'snapshot_{int(time.time())}.jpg'
                cv2.imwrite(snapshot_path, vis)
                print(f"Saved {snapshot_path}")

    cap.release()
    if writer:
        writer.release()
        print(f"Saved to {output_path}")
    cv2.destroyAllWindows()


def process_directory(estimator: PoseEstimator,
                      source_dir: str,
                      output_dir: str = None,
                      conf_thresh: float = 0.3,
                      save_txt: bool = False):
    """批量处理目录中的图像"""
    source_path = Path(source_dir)

    # 支持的图像格式
    img_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in source_path.iterdir()
                   if f.suffix.lower() in img_formats]

    if not image_files:
        print(f"No images found in {source_dir}")
        return

    print(f"Found {len(image_files)} images")

    # 创建输出目录
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_files):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load: {img_path}")
            continue

        # 推理
        keypoints, _ = estimator.predict(image)

        # 过滤低置信度
        visibility = (keypoints[:, 2] > conf_thresh).astype(np.float32)

        print(f"[{i+1}/{len(image_files)}] {img_path.name}")

        # 保存可视化结果
        if output_dir:
            vis = draw_pose(image, keypoints, visibility)
            out_path = output_path / img_path.name
            cv2.imwrite(str(out_path), vis)

            # 保存关键点为文本文件
            if save_txt:
                txt_path = output_path / f"{img_path.stem}.txt"
                with open(txt_path, 'w') as f:
                    for kpt in keypoints:
                        f.write(f"{kpt[0]:.2f} {kpt[1]:.2f} {kpt[2]:.4f}\n")

    print(f"Done! Results saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='ConvNeXt-Pose Inference')

    # 模型参数
    parser.add_argument('--weights', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--backbone', type=str, default='tiny',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Backbone 类型')
    parser.add_argument('--head-type', type=str, default='heatmap',
                       choices=['heatmap', 'paf', 'yolo'],
                       help='检测头类型')
    parser.add_argument('--img-size', type=int, nargs=2, default=[256, 192],
                       help='输入大小 (H, W)')

    # 输入源
    parser.add_argument('--source', type=str, default=None,
                       help='输入源: 图像路径/视频路径/目录路径')
    parser.add_argument('--camera', type=int, default=None,
                       help='摄像头ID')

    # 输出选项
    parser.add_argument('--output', type=str, default=None,
                       help='输出路径 (图像/视频/目录)')
    parser.add_argument('--show', action='store_true',
                       help='显示结果')
    parser.add_argument('--save-txt', action='store_true',
                       help='保存关键点为文本文件')
    parser.add_argument('--save-heatmap', action='store_true',
                       help='保存热图可视化')

    # 推理选项
    parser.add_argument('--conf-thresh', type=float, default=0.3,
                       help='关键点置信度阈值')
    parser.add_argument('--no-dark', action='store_true',
                       help='不使用 DARK 解码')
    parser.add_argument('--device', type=str, default='cuda',
                       help='运行设备')

    return parser.parse_args()


def main():
    args = parse_args()

    # 创建估计器
    estimator = PoseEstimator(
        weights=args.weights,
        backbone=args.backbone,
        head_type=args.head_type,
        input_size=tuple(args.img_size),
        device=args.device,
        use_dark=not args.no_dark
    )

    # 处理输入
    if args.source:
        source = Path(args.source)

        if source.is_file():
            # 检查是图像还是视频
            img_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            vid_formats = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

            if source.suffix.lower() in img_formats:
                process_image(
                    estimator, str(source), args.output,
                    show=args.show, save_heatmap=args.save_heatmap,
                    conf_thresh=args.conf_thresh
                )
            elif source.suffix.lower() in vid_formats:
                process_video(
                    estimator, str(source), args.output,
                    show=args.show, conf_thresh=args.conf_thresh
                )
            else:
                print(f"Unsupported file format: {source.suffix}")

        elif source.is_dir():
            process_directory(
                estimator, str(source), args.output,
                conf_thresh=args.conf_thresh, save_txt=args.save_txt
            )
        else:
            print(f"Source not found: {source}")

    elif args.camera is not None:
        process_camera(
            estimator, args.camera, args.output,
            show=args.show, conf_thresh=args.conf_thresh
        )
    else:
        print("Please specify --source or --camera")
        print("\nExamples:")
        print("  python scripts/detect.py --weights best.pt --source image.jpg --show")
        print("  python scripts/detect.py --weights best.pt --source video.mp4 --output result.mp4")
        print("  python scripts/detect.py --weights best.pt --source images/ --output results/")
        print("  python scripts/detect.py --weights best.pt --camera 0 --show")


if __name__ == '__main__':
    main()
