#!/usr/bin/env python3
"""
过拟合测试脚本

用于验证模型和数据流水线的正确性：
1. 从数据集随机选择少量样本（默认100张）
2. 反复训练多轮（默认200轮）
3. 预期：Loss 应迅速下降趋近于 0

如果能过拟合，说明：
- 模型结构正确
- 数据预处理正确
- 损失函数正确
- 训练循环正确

用法:
    python scripts/overfit_test.py --data /path/to/dataset.yaml --num-samples 100 --epochs 200
"""

import sys
import os
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from convnext_pose.models import ConvNeXtPose
from convnext_pose.models.loss import HeatmapLoss
from convnext_pose.utils.data import YOLOPoseDataset
from convnext_pose.utils.metrics import decode_heatmap
from convnext_pose.utils.visualization import draw_pose


def parse_args():
    parser = argparse.ArgumentParser(description='过拟合测试')

    # 数据
    parser.add_argument('--data', type=str, required=True,
                       help='YOLO 数据集配置文件')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='测试样本数量')

    # 模型
    parser.add_argument('--backbone', type=str, default='tiny',
                       choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--head-type', type=str, default='heatmap',
                       choices=['heatmap', 'paf', 'yolo'])
    parser.add_argument('--img-size', type=int, nargs=2, default=[256, 192],
                       help='输入大小 (H, W)')

    # 训练
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')

    # 输出
    parser.add_argument('--output-dir', type=str, default='./runs/overfit_test',
                       help='输出目录')
    parser.add_argument('--save-interval', type=int, default=50,
                       help='保存可视化结果的间隔')
    parser.add_argument('--device', type=str, default='cuda',
                       help='运行设备')

    return parser.parse_args()


def create_subset_dataset(data_yaml: str, num_samples: int, input_size, output_size):
    """创建小规模子数据集"""
    # 加载完整数据集（不做数据增强，便于观察）
    full_dataset = YOLOPoseDataset(
        data_yaml=data_yaml,
        split='train',
        input_size=input_size,
        output_size=output_size,
        augment=False  # 关闭数据增强
    )

    # 随机选择样本
    total_samples = len(full_dataset)
    num_samples = min(num_samples, total_samples)

    indices = random.sample(range(total_samples), num_samples)
    subset = Subset(full_dataset, indices)

    print(f"从 {total_samples} 个样本中选择了 {num_samples} 个样本进行过拟合测试")

    return subset, full_dataset


def visualize_predictions(model, dataset, indices, output_dir, epoch, device, input_size):
    """可视化预测结果"""
    model.eval()
    vis_dir = Path(output_dir) / 'visualizations' / f'epoch_{epoch:04d}'
    vis_dir.mkdir(parents=True, exist_ok=True)

    stride = input_size[0] // 64  # 假设输出是输入的 1/4

    # 只可视化前几个样本
    num_vis = min(8, len(indices))

    with torch.no_grad():
        for i in range(num_vis):
            idx = indices[i]
            image, target = dataset.dataset[idx]

            # 准备输入
            input_tensor = image.unsqueeze(0).to(device)

            # 推理
            outputs = model(input_tensor)
            if isinstance(outputs, tuple):
                _, outputs = outputs
                outputs = outputs[-1]

            # 解码预测关键点
            pred_kpts, pred_scores = decode_heatmap(outputs, stride=stride)
            pred_kpts = pred_kpts[0]  # (K, 2)
            pred_scores = pred_scores[0]  # (K,)

            # 获取GT关键点
            gt_kpts = target['keypoints'].numpy()  # (K, 3)

            # 反归一化图像用于可视化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = image.numpy().transpose(1, 2, 0)
            img_np = (img_np * std + mean) * 255
            img_np = img_np.astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # 创建并排对比图
            h, w = img_np.shape[:2]
            canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)

            # 左侧：GT
            img_gt = img_np.copy()
            img_gt = draw_pose(img_gt, gt_kpts, gt_kpts[:, 2])
            cv2.putText(img_gt, 'GT', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 右侧：预测
            img_pred = img_np.copy()
            pred_kpts_with_score = np.zeros((len(pred_kpts), 3))
            pred_kpts_with_score[:, :2] = pred_kpts
            pred_kpts_with_score[:, 2] = pred_scores
            img_pred = draw_pose(img_pred, pred_kpts_with_score, pred_scores)
            cv2.putText(img_pred, 'Pred', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            canvas[:, :w] = img_gt
            canvas[:, w:] = img_pred

            # 保存
            save_path = vis_dir / f'sample_{i:02d}.jpg'
            cv2.imwrite(str(save_path), canvas)

    model.train()
    print(f"  可视化结果已保存到 {vis_dir}")


def compute_pck(pred_kpts, gt_kpts, threshold=0.05, img_size=(256, 192)):
    """计算 PCK (Percentage of Correct Keypoints)

    threshold: 相对于图像对角线的比例
    """
    # 计算图像对角线长度
    diag = np.sqrt(img_size[0]**2 + img_size[1]**2)
    thresh_px = threshold * diag

    # 计算距离
    dist = np.sqrt(np.sum((pred_kpts - gt_kpts[:, :2])**2, axis=1))

    # 只考虑可见的关键点
    visible = gt_kpts[:, 2] > 0

    if visible.sum() == 0:
        return 1.0

    correct = (dist < thresh_px) & visible
    pck = correct.sum() / visible.sum()

    return pck


def main():
    args = parse_args()

    # 设置随机种子
    random.seed(42)
    torch.manual_seed(42)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 输入输出大小
    input_size = tuple(args.img_size)
    output_size = (input_size[0] // 4, input_size[1] // 4)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建子数据集
    subset, full_dataset = create_subset_dataset(
        args.data, args.num_samples, input_size, output_size
    )

    # 保存选中的样本索引
    indices = subset.indices

    # 创建数据加载器
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # 创建模型
    model = ConvNeXtPose(
        backbone=args.backbone,
        num_keypoints=full_dataset.num_keypoints,
        head_type=args.head_type
    )
    model = model.to(device)
    model.train()

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型: ConvNeXt-{args.backbone.upper()}-{args.head_type}")
    print(f"参数量: {num_params:.2f}M")

    # 损失函数
    criterion = HeatmapLoss().to(device)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    print(f"\n开始过拟合测试: {args.epochs} 轮, {len(subset)} 个样本")
    print("=" * 60)

    # 记录
    loss_history = []
    pck_history = []

    stride = input_size[0] // output_size[0]

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0
        epoch_pck = 0
        num_batches = 0

        for images, targets in dataloader:
            images = images.to(device)
            heatmaps = targets['heatmap'].to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)

            if isinstance(outputs, tuple):
                _, outputs = outputs
                outputs = outputs[-1]

            # 计算损失
            loss = criterion(outputs, heatmaps)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 计算 PCK
            with torch.no_grad():
                pred_kpts, pred_scores = decode_heatmap(outputs, stride=stride)
                gt_kpts = targets['keypoints'].numpy()

                batch_pck = 0
                for i in range(len(pred_kpts)):
                    batch_pck += compute_pck(pred_kpts[i], gt_kpts[i],
                                            threshold=0.05, img_size=input_size)
                epoch_pck += batch_pck / len(pred_kpts)

            num_batches += 1

        # 更新学习率
        scheduler.step()

        # 计算平均值
        avg_loss = epoch_loss / num_batches
        avg_pck = epoch_pck / num_batches

        loss_history.append(avg_loss)
        pck_history.append(avg_pck)

        # 打印进度
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:3d}/{args.epochs}] "
              f"Loss: {avg_loss:.6f} | PCK@0.05: {avg_pck:.4f} | LR: {lr:.6f}")

        # 定期保存可视化结果
        if epoch % args.save_interval == 0 or epoch == 1 or epoch == args.epochs:
            visualize_predictions(
                model, subset, indices, output_dir, epoch, device, input_size
            )

    print("=" * 60)
    print("\n过拟合测试完成!")
    print(f"初始 Loss: {loss_history[0]:.6f} -> 最终 Loss: {loss_history[-1]:.6f}")
    print(f"初始 PCK:  {pck_history[0]:.4f} -> 最终 PCK:  {pck_history[-1]:.4f}")

    # 判断是否成功过拟合
    if loss_history[-1] < 0.01 and pck_history[-1] > 0.95:
        print("\n✓ 过拟合测试通过! 模型能够完全拟合训练数据。")
        print("  这表明模型结构、数据预处理和训练流程都是正确的。")
    elif loss_history[-1] < 0.1 and pck_history[-1] > 0.8:
        print("\n△ 过拟合测试基本通过。Loss 和 PCK 都在合理范围内。")
        print("  建议增加训练轮数或检查数据预处理。")
    else:
        print("\n✗ 过拟合测试未通过! 模型无法拟合训练数据。")
        print("  请检查:")
        print("  - 数据预处理是否正确")
        print("  - 热图生成是否正确")
        print("  - 损失函数是否正确")
        print("  - 模型结构是否正确")

    # 保存损失曲线
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss 曲线
        axes[0].plot(loss_history, 'b-')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_yscale('log')
        axes[0].grid(True)

        # PCK 曲线
        axes[1].plot(pck_history, 'g-')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('PCK@0.05')
        axes[1].set_title('PCK Accuracy')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=150)
        print(f"\n训练曲线已保存到 {output_dir / 'training_curves.png'}")
    except ImportError:
        print("\n(matplotlib 未安装，跳过绘图)")

    # 保存最终模型
    torch.save({
        'model': model.state_dict(),
        'loss_history': loss_history,
        'pck_history': pck_history,
        'args': vars(args)
    }, output_dir / 'final_model.pt')
    print(f"模型已保存到 {output_dir / 'final_model.pt'}")


if __name__ == '__main__':
    main()
