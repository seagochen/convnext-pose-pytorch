#!/usr/bin/env python3
"""
过拟合测试 - 验证 YOLO-Pose 多人模型是否能正确收敛

使用少量样本（如 8-16 张图）反复训练，验证：
1. 模型能否过拟合这些样本（loss 降到接近 0）
2. 预测结果是否与 GT 匹配
3. 没有 NaN 或其他数值问题

如果模型正确，应该在几百个 step 内看到 loss 显著下降。
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from convnext_pose.models import ConvNeXtPose
from convnext_pose.models.yolo_loss import build_yolo_loss
from convnext_pose.utils.data import YOLOPoseDataset, collate_fn_yolo


def overfit_test(
    num_samples: int = 8,
    num_steps: int = 500,
    lr: float = 1e-3,
    use_amp: bool = True,
    print_interval: int = 50,
    vis_interval: int = 100,
    save_vis: bool = True
):
    """过拟合测试

    Args:
        num_samples: 用于过拟合的样本数
        num_steps: 训练步数
        lr: 学习率
        use_amp: 是否使用 AMP
        print_interval: 打印间隔
        vis_interval: 可视化保存间隔（每隔多少步保存一次）
        save_vis: 是否保存可视化结果
    """
    device = torch.device('cuda')
    input_size = (256, 192)

    print("=" * 70)
    print(f"YOLO-Pose Overfit Test")
    print(f"  Samples: {num_samples}, Steps: {num_steps}, LR: {lr}, AMP: {use_amp}")
    print("=" * 70)

    # 创建模型
    model = ConvNeXtPose(
        backbone='tiny',
        num_keypoints=17,
        fpn_channels=256,
        use_fpn=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: ConvNeXt-Tiny-YOLOPose ({total_params:.2f}M params)")

    # 损失函数
    criterion = build_yolo_loss(
        num_keypoints=17,
        box_weight=0.5,
        obj_weight=1.0,
        kpt_weight=0.5,
        strides=model.strides
    ).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # GradScaler
    scaler = GradScaler(init_scale=2**10) if use_amp else None

    # 加载数据集，取前 num_samples 个样本
    dataset = YOLOPoseDataset(
        data_yaml='/home/cxt/datasets/coco_pose_yolo/data.yaml',
        split='train',
        input_size=list(input_size),
        max_persons=20,
        augment=False  # 过拟合测试不使用数据增强
    )

    # 选择有多个人的样本来测试多人检测
    print("\nFinding samples with multiple persons...")
    multi_person_indices = []
    for i in range(min(2000, len(dataset))):  # 检查前2000个
        sample = dataset[i]
        if sample is not None:
            _, targets = sample
            # 兼容不同的返回格式
            n_persons = targets['num_persons']
            if isinstance(n_persons, torch.Tensor):
                n_persons = n_persons.item()
            if n_persons >= 2:  # 至少2个人
                multi_person_indices.append(i)
                if len(multi_person_indices) >= num_samples:
                    break

    if len(multi_person_indices) < num_samples:
        print(f"Warning: Only found {len(multi_person_indices)} multi-person samples, adding single-person samples")
        # 如果多人样本不够，补充单人样本
        for i in range(min(2000, len(dataset))):
            if i not in multi_person_indices:
                sample = dataset[i]
                if sample is not None:
                    multi_person_indices.append(i)
                    if len(multi_person_indices) >= num_samples:
                        break

    subset = Subset(dataset, multi_person_indices[:num_samples])

    # 打印选中样本的信息
    print(f"\nSelected {len(subset)} samples:")
    for i, idx in enumerate(multi_person_indices[:min(10, num_samples)]):  # 只打印前10个
        sample = dataset[idx]
        if sample:
            _, targets = sample
            n_persons = targets['num_persons']
            if isinstance(n_persons, torch.Tensor):
                n_persons = n_persons.item()
            print(f"  Sample {i}: index={idx}, persons={n_persons}")
    if num_samples > 10:
        print(f"  ... and {num_samples - 10} more samples")

    loader = DataLoader(
        subset,
        batch_size=min(num_samples, 8),  # batch size 不超过样本数
        shuffle=True,
        num_workers=0,  # 过拟合测试用单线程
        collate_fn=collate_fn_yolo,
        drop_last=False
    )

    print(f"\nDataLoader: {len(loader)} batches, batch_size={loader.batch_size}")

    # 固定一个 batch 用于过拟合
    fixed_batch = None
    for batch in loader:
        fixed_batch = batch
        break

    images, targets = fixed_batch
    images = images.to(device)
    targets_device = {
        'bboxes': targets['bboxes'].to(device),
        'keypoints': targets['keypoints'].to(device),
        'num_persons': targets['num_persons'].to(device)
    }

    print(f"\nFixed batch info:")
    print(f"  Images shape: {images.shape}")
    print(f"  Bboxes shape: {targets['bboxes'].shape}")
    print(f"  Keypoints shape: {targets['keypoints'].shape}")
    print(f"  Num persons: {targets['num_persons'].tolist()}")

    # 训练循环
    print(f"\n{'='*70}")
    print("Starting overfit training...")
    print(f"{'='*70}")

    # 创建可视化输出目录
    vis_dir = project_root / 'runs' / 'overfit_test'
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 辅助函数：保存当前预测结果的可视化
    def save_visualization(step_num):
        from convnext_pose.utils.visualization.plots import visualize_val_samples

        model.eval()
        with torch.no_grad():
            outputs = model(images)

            pred_bboxes_list = []
            pred_keypoints_list = []
            pred_scores_list = []
            gt_bboxes_list = []
            gt_keypoints_list = []

            img_h, img_w = input_size

            for b in range(images.shape[0]):
                n_persons = targets['num_persons'][b].item()

                # 预测
                single_outputs = [out[b:b+1] for out in outputs]
                pred_bboxes, pred_scores, pred_keypoints = model.decode(
                    single_outputs, conf_thresh=0.1, input_size=input_size
                )

                pred_bboxes_list.append(pred_bboxes.cpu().numpy() if len(pred_bboxes) > 0 else None)
                pred_keypoints_list.append(pred_keypoints.cpu().numpy() if len(pred_keypoints) > 0 else None)
                pred_scores_list.append(pred_scores.cpu().numpy() if len(pred_scores) > 0 else None)

                # GT
                if n_persons > 0:
                    gt_bboxes_norm = targets['bboxes'][b, :n_persons]
                    gt_kpts = targets['keypoints'][b, :n_persons]

                    # 转换 bbox 到像素坐标
                    cx = gt_bboxes_norm[:, 0] * img_w
                    cy = gt_bboxes_norm[:, 1] * img_h
                    w = gt_bboxes_norm[:, 2] * img_w
                    h = gt_bboxes_norm[:, 3] * img_h
                    gt_boxes_xyxy = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

                    # 转换关键点到像素坐标
                    gt_kpts_abs = gt_kpts.clone()
                    bbox_cx = gt_bboxes_norm[:, 0:1]
                    bbox_cy = gt_bboxes_norm[:, 1:2]
                    bbox_w = gt_bboxes_norm[:, 2:3]
                    bbox_h = gt_bboxes_norm[:, 3:4]
                    gt_kpts_abs[:, :, 0] = (bbox_cx + gt_kpts[:, :, 0] * bbox_w) * img_w
                    gt_kpts_abs[:, :, 1] = (bbox_cy + gt_kpts[:, :, 1] * bbox_h) * img_h

                    gt_bboxes_list.append(gt_boxes_xyxy.cpu().numpy())
                    gt_keypoints_list.append(gt_kpts_abs.cpu().numpy())
                else:
                    gt_bboxes_list.append(None)
                    gt_keypoints_list.append(None)

            # 保存可视化
            vis_path = vis_dir / f'step_{step_num:04d}.jpg'
            visualize_val_samples(
                images=images.cpu(),
                pred_bboxes_list=pred_bboxes_list,
                pred_keypoints_list=pred_keypoints_list,
                pred_scores_list=pred_scores_list,
                gt_bboxes_list=gt_bboxes_list,
                gt_keypoints_list=gt_keypoints_list,
                save_path=str(vis_path),
                max_samples=num_samples
            )
            print(f"  [Saved visualization to {vis_path.name}]")

        model.train()

    model.train()
    loss_history = []

    # 保存初始状态（step 0）
    if save_vis:
        print("\nSaving initial visualization (before training)...")
        save_visualization(0)

    for step in range(num_steps):
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(images)
                loss_dict = criterion(outputs, targets_device, input_size)
                loss = loss_dict['loss']

            # 检查 NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[Step {step}] ❌ NaN/Inf detected!")
                print(f"  box_loss: {loss_dict['box_loss'].item()}")
                print(f"  obj_loss: {loss_dict['obj_loss'].item()}")
                print(f"  kpt_loss: {loss_dict['kpt_loss'].item()}")
                return False

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss_dict = criterion(outputs, targets_device, input_size)
            loss = loss_dict['loss']

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[Step {step}] ❌ NaN/Inf detected!")
                return False

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        loss_history.append(loss.item())

        # 打印进度
        if step % print_interval == 0 or step == num_steps - 1:
            box_loss = loss_dict['box_loss'].item()
            obj_loss = loss_dict['obj_loss'].item()
            kpt_loss = loss_dict['kpt_loss'].item()
            num_pos = loss_dict['num_pos'].item()

            print(f"[Step {step:4d}] loss={loss.item():.4f} "
                  f"(box={box_loss:.4f}, obj={obj_loss:.4f}, kpt={kpt_loss:.4f}) "
                  f"pos={num_pos:.0f} grad={grad_norm:.2f}")

        # 定期保存可视化
        if save_vis and (step + 1) % vis_interval == 0:
            save_visualization(step + 1)

    # 分析收敛情况
    print(f"\n{'='*70}")
    print("Analysis")
    print(f"{'='*70}")

    initial_loss = np.mean(loss_history[:10])
    final_loss = np.mean(loss_history[-10:])
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100

    print(f"Initial loss (avg first 10): {initial_loss:.4f}")
    print(f"Final loss (avg last 10):    {final_loss:.4f}")
    print(f"Loss reduction:              {loss_reduction:.1f}%")

    # 评估最终预测
    print(f"\n{'='*70}")
    print("Final Predictions vs GT")
    print(f"{'='*70}")

    model.eval()
    with torch.no_grad():
        outputs = model(images)

        # 解码预测
        for b in range(images.shape[0]):
            n_persons_gt = targets['num_persons'][b].item()
            print(f"\n[Image {b}] GT persons: {n_persons_gt}")

            # 解码单张图像的预测
            single_outputs = [out[b:b+1] for out in outputs]
            pred_bboxes, pred_scores, pred_keypoints = model.decode(
                single_outputs, conf_thresh=0.1, input_size=input_size
            )

            print(f"  Predictions: {len(pred_scores)} detections")

            if len(pred_scores) > 0:
                # 显示 top-k 预测
                top_k = min(5, len(pred_scores))
                sorted_idx = pred_scores.argsort(descending=True)[:top_k]

                for i, idx in enumerate(sorted_idx):
                    score = pred_scores[idx].item()
                    bbox = pred_bboxes[idx].cpu().numpy()
                    print(f"    [{i}] score={score:.3f}, bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")

            # 显示 GT
            if n_persons_gt > 0:
                gt_bboxes = targets['bboxes'][b, :n_persons_gt]
                img_h, img_w = input_size
                for p in range(n_persons_gt):
                    cx, cy, w, h = gt_bboxes[p].cpu().numpy()
                    x1 = (cx - w/2) * img_w
                    y1 = (cy - h/2) * img_h
                    x2 = (cx + w/2) * img_w
                    y2 = (cy + h/2) * img_h
                    print(f"    GT[{p}] bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    # 保存可视化
    if save_vis:
        print(f"\n{'='*70}")
        print("Saving visualization...")
        print(f"{'='*70}")

        from convnext_pose.utils.visualization.plots import visualize_val_samples

        # 收集预测和 GT
        pred_bboxes_list = []
        pred_keypoints_list = []
        pred_scores_list = []
        gt_bboxes_list = []
        gt_keypoints_list = []

        img_h, img_w = input_size

        with torch.no_grad():
            for b in range(images.shape[0]):
                n_persons = targets['num_persons'][b].item()

                # 预测
                single_outputs = [out[b:b+1] for out in outputs]
                pred_bboxes, pred_scores, pred_keypoints = model.decode(
                    single_outputs, conf_thresh=0.1, input_size=input_size
                )

                pred_bboxes_list.append(pred_bboxes.cpu().numpy() if len(pred_bboxes) > 0 else None)
                pred_keypoints_list.append(pred_keypoints.cpu().numpy() if len(pred_keypoints) > 0 else None)
                pred_scores_list.append(pred_scores.cpu().numpy() if len(pred_scores) > 0 else None)

                # GT
                if n_persons > 0:
                    gt_bboxes_norm = targets['bboxes'][b, :n_persons]
                    gt_kpts = targets['keypoints'][b, :n_persons]

                    # 转换 bbox 到像素坐标
                    cx = gt_bboxes_norm[:, 0] * img_w
                    cy = gt_bboxes_norm[:, 1] * img_h
                    w = gt_bboxes_norm[:, 2] * img_w
                    h = gt_bboxes_norm[:, 3] * img_h
                    gt_boxes_xyxy = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

                    # 转换关键点到像素坐标
                    gt_kpts_abs = gt_kpts.clone()
                    bbox_cx = gt_bboxes_norm[:, 0:1]
                    bbox_cy = gt_bboxes_norm[:, 1:2]
                    bbox_w = gt_bboxes_norm[:, 2:3]
                    bbox_h = gt_bboxes_norm[:, 3:4]
                    gt_kpts_abs[:, :, 0] = (bbox_cx + gt_kpts[:, :, 0] * bbox_w) * img_w
                    gt_kpts_abs[:, :, 1] = (bbox_cy + gt_kpts[:, :, 1] * bbox_h) * img_h

                    gt_bboxes_list.append(gt_boxes_xyxy.cpu().numpy())
                    gt_keypoints_list.append(gt_kpts_abs.cpu().numpy())
                else:
                    gt_bboxes_list.append(None)
                    gt_keypoints_list.append(None)

        # 保存可视化
        vis_path = project_root / 'runs' / 'overfit_test.jpg'
        vis_path.parent.mkdir(parents=True, exist_ok=True)

        visualize_val_samples(
            images=images.cpu(),
            pred_bboxes_list=pred_bboxes_list,
            pred_keypoints_list=pred_keypoints_list,
            pred_scores_list=pred_scores_list,
            gt_bboxes_list=gt_bboxes_list,
            gt_keypoints_list=gt_keypoints_list,
            save_path=str(vis_path),
            max_samples=num_samples
        )
        print(f"Saved visualization to: {vis_path}")

    # 判断是否成功
    success = loss_reduction > 50  # 损失下降超过 50% 认为成功

    print(f"\n{'='*70}")
    if success:
        print("✓ OVERFIT TEST PASSED - Model can learn from the data")
    else:
        print("❌ OVERFIT TEST FAILED - Model may have issues")
    print(f"{'='*70}")

    return success


def main():
    import argparse
    parser = argparse.ArgumentParser(description='YOLO-Pose Overfit Test')
    parser.add_argument('--samples', type=int, default=8, help='Number of samples to overfit')
    parser.add_argument('--steps', type=int, default=500, help='Training steps')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--no-amp', action='store_true', help='Disable AMP')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    parser.add_argument('--vis-interval', type=int, default=100, help='Visualization save interval')
    args = parser.parse_args()

    overfit_test(
        num_samples=args.samples,
        num_steps=args.steps,
        lr=args.lr,
        use_amp=not args.no_amp,
        vis_interval=args.vis_interval,
        save_vis=not args.no_vis
    )


if __name__ == '__main__':
    main()
