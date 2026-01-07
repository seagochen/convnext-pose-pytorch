#!/usr/bin/env python3
"""
快速 NaN 调试实验

通过减少数据量和 epoch 数，快速验证不同学习率下 NaN 的发生情况
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from convnext_pose.models import ConvNeXtPose
from convnext_pose.models.yolo_loss import build_yolo_loss
from convnext_pose.utils.data import YOLOPoseDataset, collate_fn_yolo


def quick_test(lr: float, num_samples: int = 500, num_steps: int = 200, use_amp: bool = True):
    """快速测试特定学习率是否会导致 NaN

    Args:
        lr: 学习率
        num_samples: 使用的样本数
        num_steps: 训练步数
        use_amp: 是否使用 AMP
    """
    device = torch.device('cuda')
    input_size = (256, 192)

    print(f"\n{'='*60}")
    print(f"Testing LR={lr}, samples={num_samples}, steps={num_steps}, AMP={use_amp}")
    print(f"{'='*60}")

    # 创建模型
    model = ConvNeXtPose(
        backbone='tiny',
        num_keypoints=17,
        fpn_channels=256,
        use_fpn=True,
    ).to(device)

    # 损失函数
    criterion = build_yolo_loss(
        num_keypoints=17,
        box_weight=0.5,
        obj_weight=1.0,
        kpt_weight=0.5,
        strides=model.strides
    ).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    # GradScaler
    scaler = GradScaler(init_scale=2**10) if use_amp else None

    # 数据集 (取子集)
    dataset = YOLOPoseDataset(
        data_yaml='/home/cxt/datasets/coco_pose_yolo/data.yaml',
        split='train',
        input_size=list(input_size),
        max_persons=20,
        augment=True
    )
    subset = Subset(dataset, range(min(num_samples, len(dataset))))
    loader = DataLoader(
        subset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn_yolo,
        drop_last=True
    )

    # 模拟 warmup: 线性增加学习率
    warmup_steps = 50

    model.train()
    step = 0
    nan_step = None

    # 训练循环
    for epoch in range(10):  # 最多 10 个 epoch
        for images, targets in loader:
            if step >= num_steps:
                break

            images = images.to(device)
            targets_device = {
                'bboxes': targets['bboxes'].to(device),
                'keypoints': targets['keypoints'].to(device),
                'num_persons': targets['num_persons'].to(device)
            }

            # Warmup 学习率
            if step < warmup_steps:
                warmup_lr = lr * (step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr
                current_lr = warmup_lr
            else:
                current_lr = lr

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss_dict = criterion(outputs, targets_device, input_size)
                    loss = loss_dict['loss']

                # 检查 NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[Step {step}] NaN detected! LR={current_lr:.6f}")
                    print(f"  box_loss: {loss_dict['box_loss'].item()}")
                    print(f"  obj_loss: {loss_dict['obj_loss'].item()}")
                    print(f"  kpt_loss: {loss_dict['kpt_loss'].item()}")
                    nan_step = step
                    break

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"\n[Step {step}] Gradient NaN! LR={current_lr:.6f}, grad_norm={grad_norm}")
                    nan_step = step
                    break

                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss_dict = criterion(outputs, targets_device, input_size)
                loss = loss_dict['loss']

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[Step {step}] NaN detected! LR={current_lr:.6f}")
                    nan_step = step
                    break

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"\n[Step {step}] Gradient NaN! LR={current_lr:.6f}")
                    nan_step = step
                    break

                optimizer.step()

            # 打印进度
            if step % 20 == 0:
                print(f"[Step {step:3d}] loss={loss.item():.4f}, lr={current_lr:.6f}, "
                      f"grad_norm={grad_norm:.2f}, scale={scaler.get_scale() if scaler else 'N/A'}")

            step += 1

        if nan_step is not None or step >= num_steps:
            break

    # 结果
    if nan_step is not None:
        print(f"\n❌ FAILED: NaN at step {nan_step}")
        return False, nan_step
    else:
        print(f"\n✓ PASSED: No NaN in {step} steps")
        return True, step


def long_test(lr: float, num_steps: int = 5000, use_amp: bool = True):
    """长时间测试，模拟真实训练情况

    Args:
        lr: 学习率
        num_steps: 训练步数 (5000 步约等于 1.5 个 epoch)
        use_amp: 是否使用 AMP
    """
    device = torch.device('cuda')
    input_size = (256, 192)

    print(f"\n{'='*60}")
    print(f"Long Test: LR={lr}, steps={num_steps}, AMP={use_amp}")
    print(f"{'='*60}")

    # 创建模型
    model = ConvNeXtPose(
        backbone='tiny',
        num_keypoints=17,
        fpn_channels=256,
        use_fpn=True,
    ).to(device)

    # 损失函数
    criterion = build_yolo_loss(
        num_keypoints=17,
        box_weight=0.5,
        obj_weight=1.0,
        kpt_weight=0.5,
        strides=model.strides
    ).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    # GradScaler - 使用和 trainer 相同的配置
    scaler = GradScaler(init_scale=2**10, growth_interval=2000) if use_amp else None

    # 完整数据集
    dataset = YOLOPoseDataset(
        data_yaml='/home/cxt/datasets/coco_pose_yolo/data.yaml',
        split='train',
        input_size=list(input_size),
        max_persons=20,
        augment=True
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_yolo,
        drop_last=True
    )

    # Warmup 配置 (5 epochs = 5 * 3266 = 16330 steps)
    warmup_steps = 5 * len(loader)
    print(f"Warmup steps: {warmup_steps}, Total loader size: {len(loader)}")

    model.train()
    step = 0
    nan_step = None

    # 记录参数范围
    def check_param_stats():
        max_val = 0
        min_val = 0
        for p in model.parameters():
            if p is not None:
                max_val = max(max_val, p.data.abs().max().item())
                min_val = min(min_val, p.data.min().item())
        return max_val, min_val

    # 训练循环
    for epoch in range(20):  # 最多 20 个 epoch
        for images, targets in loader:
            if step >= num_steps:
                break

            images = images.to(device)
            targets_device = {
                'bboxes': targets['bboxes'].to(device),
                'keypoints': targets['keypoints'].to(device),
                'num_persons': targets['num_persons'].to(device)
            }

            # Warmup 学习率 (线性增加)
            if step < warmup_steps:
                warmup_lr = lr * (step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr
                current_lr = warmup_lr
            else:
                current_lr = lr

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss_dict = criterion(outputs, targets_device, input_size)
                    loss = loss_dict['loss']

                # 检查 NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[Step {step}] NaN detected! LR={current_lr:.6f}")
                    print(f"  box_loss: {loss_dict['box_loss'].item()}")
                    print(f"  obj_loss: {loss_dict['obj_loss'].item()}")
                    print(f"  kpt_loss: {loss_dict['kpt_loss'].item()}")
                    print(f"  scale: {scaler.get_scale()}")
                    max_p, min_p = check_param_stats()
                    print(f"  param max_abs: {max_p:.4f}, min: {min_p:.4f}")
                    nan_step = step
                    break

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # 梯度裁剪
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"\n[Step {step}] Gradient NaN! LR={current_lr:.6f}, grad_norm={grad_norm}")
                    print(f"  scale: {scaler.get_scale()}")
                    nan_step = step
                    break

                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss_dict = criterion(outputs, targets_device, input_size)
                loss = loss_dict['loss']

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[Step {step}] NaN detected! LR={current_lr:.6f}")
                    nan_step = step
                    break

                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"\n[Step {step}] Gradient NaN! LR={current_lr:.6f}")
                    nan_step = step
                    break

                optimizer.step()

            # 打印进度 (每 500 步)
            if step % 500 == 0:
                max_p, min_p = check_param_stats()
                scale_str = f"{scaler.get_scale():.0f}" if scaler else "N/A"
                print(f"[Step {step:5d}] loss={loss.item():.4f}, lr={current_lr:.6f}, "
                      f"grad={grad_norm:.2f}, scale={scale_str}, param_max={max_p:.2f}")

            step += 1

        if nan_step is not None or step >= num_steps:
            break

    # 结果
    if nan_step is not None:
        print(f"\n❌ FAILED: NaN at step {nan_step}")
        return False, nan_step
    else:
        print(f"\n✓ PASSED: No NaN in {step} steps")
        return True, step


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--long', action='store_true', help='Run long test (5000 steps)')
    parser.add_argument('--steps', type=int, default=5000, help='Steps for long test')
    parser.add_argument('--debug-nan', action='store_true', help='Enable anomaly detection (like trainer)')
    args = parser.parse_args()

    # 测试 debug-nan 是否是问题根源
    if args.debug_nan:
        print("=" * 60)
        print("[DEBUG] Enabling torch.autograd.set_detect_anomaly(True)")
        print("=" * 60)
        torch.autograd.set_detect_anomaly(True)

    if args.long:
        # 长时间测试
        long_test(lr=1e-3, num_steps=args.steps, use_amp=True)
    else:
        # 快速测试
        print("=" * 60)
        print("Quick NaN Test - Testing different learning rates")
        print("=" * 60)

        # 测试不同的学习率
        learning_rates = [1e-3, 5e-4, 1e-4]
        results = {}

        for lr in learning_rates:
            passed, step = quick_test(lr=lr, num_samples=500, num_steps=200, use_amp=True)
            results[lr] = (passed, step)

        # 汇总结果
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        for lr, (passed, step) in results.items():
            status = "✓ PASSED" if passed else f"❌ FAILED at step {step}"
            print(f"  LR={lr}: {status}")

        # 也测试不使用 AMP 的情况
        print("\n" + "-" * 60)
        print("Testing without AMP (lr=1e-3)")
        print("-" * 60)
        quick_test(lr=1e-3, num_samples=500, num_steps=200, use_amp=False)


if __name__ == '__main__':
    main()
