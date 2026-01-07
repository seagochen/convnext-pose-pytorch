"""
YOLO-Pose 多人姿态估计训练器

支持:
- 端到端多人姿态估计训练
- 指数移动平均 (EMA)
- 学习率预热和调度
- 检查点保存和恢复
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models import ConvNeXtPose, create_model
from ..models.yolo_loss import YOLOPoseLoss, build_yolo_loss
from ..utils.data import YOLOPoseDataset, build_dataloader, collate_fn_yolo
from ..utils.callbacks import build_scheduler, ModelEMA
from ..utils.yolo_utils import YOLOPoseEvaluator, postprocess


class YOLOPoseTrainer:
    """YOLO-Pose 多人姿态估计训练器

    Args:
        config: 配置字典
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))

        # 设置输出目录
        self.output_dir = self._setup_output_dir()

        # 设置日志
        self.logger = self._setup_logging()

        # 构建组件
        self.model = self._build_model()
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.train_loader, self.val_loader = self._build_dataloaders()
        self.scheduler = self._build_scheduler()

        # 可选组件
        self.ema = self._build_ema() if config['training']['ema'] else None

        # 训练状态
        self.start_epoch = 0
        self.best_metric = float('inf')  # 使用损失作为指标，越小越好
        self.global_step = 0

        # 输入尺寸
        self.input_size = tuple(config['data']['input_size'])

        # 恢复训练
        if config.get('resume'):
            self._resume_training(config['resume'])
        elif config.get('weights'):
            self._load_weights(config['weights'])

    def _setup_output_dir(self) -> Path:
        """设置输出目录"""
        output_cfg = self.config['output']
        base_dir = Path(output_cfg['dir']) / output_cfg['project']

        # 自动命名
        name = output_cfg['name']
        output_dir = base_dir / name

        # 如果恢复训练，使用原目录
        if self.config.get('resume'):
            resume_path = Path(self.config['resume'])
            if resume_path.is_file():
                # 从检查点路径推断输出目录
                output_dir = resume_path.parent.parent
                return output_dir

        # 如果存在则添加序号
        idx = 1
        while output_dir.exists():
            output_dir = base_dir / f"{name}{idx}"
            idx += 1

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'weights').mkdir(exist_ok=True)

        return output_dir

    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('yolo_trainer')
        logger.setLevel(logging.INFO)

        # 清除已有的处理器
        logger.handlers = []

        # 文件处理器
        fh = logging.FileHandler(self.output_dir / 'train.log')
        fh.setLevel(logging.INFO)

        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def _build_model(self) -> nn.Module:
        """构建模型"""
        model_cfg = self.config['model']

        model = ConvNeXtPose(
            backbone=model_cfg['backbone'],
            num_keypoints=model_cfg['num_keypoints'],
            fpn_channels=model_cfg.get('fpn_channels', 256),
            use_fpn=model_cfg.get('use_fpn', True),
        )

        # 加载预训练 backbone
        if model_cfg.get('pretrained'):
            model.load_pretrained_backbone(model_cfg['pretrained'])

        # 冻结 backbone
        if self.config['training']['freeze_backbone']:
            model.freeze_backbone()

        model = model.to(self.device)

        # 打印模型信息
        params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        self.logger.info(f"Model: ConvNeXt-{model_cfg['backbone'].upper()}-YOLOPose")
        self.logger.info(f"Parameters: {params:.2f}M (trainable: {trainable:.2f}M)")

        return model

    def _build_criterion(self) -> nn.Module:
        """构建损失函数"""
        num_keypoints = self.config['model']['num_keypoints']

        criterion = build_yolo_loss(
            num_keypoints=num_keypoints,
            box_weight=0.5,
            obj_weight=1.0,
            kpt_weight=0.5,
            strides=self.model.strides
        )

        return criterion.to(self.device)

    def _build_optimizer(self) -> optim.Optimizer:
        """构建优化器"""
        train_cfg = self.config['training']

        # 参数分组: backbone 使用较小学习率
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': train_cfg['lr'] * 0.1},
            {'params': head_params, 'lr': train_cfg['lr']}
        ]

        optimizer = optim.AdamW(
            param_groups,
            lr=train_cfg['lr'],
            weight_decay=train_cfg['weight_decay']
        )

        return optimizer

    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """构建数据加载器"""
        data_cfg = self.config['data']
        train_cfg = self.config['training']

        # 训练数据加载器
        train_dataset = YOLOPoseDataset(
            data_yaml=data_cfg['yaml_path'],
            split='train',
            input_size=data_cfg['input_size'],
            max_persons=20,
            augment=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=True,
            num_workers=data_cfg['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn_yolo,
            drop_last=True
        )

        # 验证数据加载器
        val_dataset = YOLOPoseDataset(
            data_yaml=data_cfg['yaml_path'],
            split='val',
            input_size=data_cfg['input_size'],
            max_persons=20,
            augment=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=False,
            num_workers=data_cfg['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn_yolo
        )

        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Val samples: {len(val_dataset)}")

        return train_loader, val_loader

    def _build_scheduler(self):
        """构建学习率调度器"""
        train_cfg = self.config['training']

        scheduler = build_scheduler(
            optimizer=self.optimizer,
            scheduler_type=train_cfg['lr_scheduler'],
            epochs=train_cfg['epochs'],
            steps_per_epoch=len(self.train_loader),
            warmup_epochs=train_cfg['warmup_epochs']
        )

        return scheduler

    def _build_ema(self) -> Optional[ModelEMA]:
        """构建 EMA"""
        if not self.config['training']['ema']:
            return None

        return ModelEMA(
            self.model,
            decay=self.config['training']['ema_decay']
        )

    def _resume_training(self, checkpoint_path: str):
        """恢复训练"""
        self.logger.info(f"Resuming from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)

        if self.ema and 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'])

        self.logger.info(f"Resumed from epoch {self.start_epoch}")

    def _load_weights(self, weights_path: str):
        """加载权重"""
        self.logger.info(f"Loading weights from {weights_path}")

        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch

        Returns:
            losses: 各项损失的字典
        """
        self.model.train()
        train_cfg = self.config['training']

        total_loss = 0
        total_box_loss = 0
        total_obj_loss = 0
        total_kpt_loss = 0
        total_num_pos = 0
        num_batches = len(self.train_loader)

        # 使用 tqdm 显示进度
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{train_cfg['epochs']}",
            unit='batch',
            ncols=140,
            leave=True
        )

        for step, (images, targets) in enumerate(pbar):
            images = images.to(self.device)

            # 将 targets 移到设备
            targets_device = {
                'bboxes': targets['bboxes'].to(self.device),
                'keypoints': targets['keypoints'].to(self.device),
                'num_persons': targets['num_persons'].to(self.device)
            }

            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(images)
            loss_dict = self.criterion(outputs, targets_device, self.input_size)
            loss = loss_dict['loss']

            # 检查 loss 是否为 NaN
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"[Step {step}] Loss is NaN/Inf, skipping this batch")
                self.logger.warning(f"  box_loss: {loss_dict['box_loss'].item()}, "
                                   f"obj_loss: {loss_dict['obj_loss'].item()}, "
                                   f"kpt_loss: {loss_dict['kpt_loss'].item()}")
                self.optimizer.zero_grad()
                continue

            loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                self.logger.warning(f"[Step {step}] Gradient NaN/Inf detected, skipping update")
                self.optimizer.zero_grad()
                continue

            self.optimizer.step()

            # 更新学习率
            self.scheduler.step()

            # 更新 EMA
            if self.ema:
                self.ema.update(self.model)

            # 累计损失
            total_loss += loss.item()
            total_box_loss += loss_dict['box_loss'].item()
            total_obj_loss += loss_dict['obj_loss'].item()
            total_kpt_loss += loss_dict['kpt_loss'].item()
            total_num_pos += loss_dict['num_pos'].item()

            self.global_step += 1

            # 更新进度条
            avg_loss = total_loss / (step + 1)
            current_lr = self.optimizer.param_groups[1]['lr']  # head lr
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'box': f'{loss_dict["box_loss"].item():.3f}',
                'obj': f'{loss_dict["obj_loss"].item():.3f}',
                'kpt': f'{loss_dict["kpt_loss"].item():.3f}',
                'lr': f'{current_lr:.2e}'
            })

        # 计算平均损失
        losses = {
            'loss': total_loss / num_batches,
            'box_loss': total_box_loss / num_batches,
            'obj_loss': total_obj_loss / num_batches,
            'kpt_loss': total_kpt_loss / num_batches,
            'num_pos': total_num_pos / num_batches
        }

        self.logger.info(
            f'Epoch [{epoch+1}/{train_cfg["epochs"]}] '
            f'Loss: {losses["loss"]:.4f} '
            f'(box: {losses["box_loss"]:.4f}, obj: {losses["obj_loss"]:.4f}, kpt: {losses["kpt_loss"]:.4f}) '
            f'LR: {self.optimizer.param_groups[1]["lr"]:.6f}'
        )

        return losses

    @torch.no_grad()
    def validate(self, epoch: int, num_vis_samples: int = 10) -> Dict[str, float]:
        """验证

        Args:
            epoch: 当前 epoch
            num_vis_samples: 可视化样本数量
        """
        model = self.ema.ema if self.ema else self.model
        model.eval()

        total_loss = 0
        total_box_loss = 0
        total_obj_loss = 0
        total_kpt_loss = 0
        num_batches = len(self.val_loader)

        # 创建评估器
        evaluator = YOLOPoseEvaluator(
            num_keypoints=self.config['model']['num_keypoints'],
            iou_thresh=0.5,
            pck_thresh=0.2
        )

        # 可视化相关：随机选择样本索引
        import random
        total_samples = len(self.val_loader.dataset)
        vis_indices = set(random.sample(range(total_samples), min(num_vis_samples, total_samples)))
        vis_data = {
            'images': [],
            'pred_bboxes': [],
            'pred_keypoints': [],
            'pred_scores': [],
            'gt_bboxes': [],
            'gt_keypoints': []
        }
        sample_idx = 0  # 跟踪当前样本索引

        # 使用 tqdm 显示验证进度
        pbar = tqdm(
            self.val_loader,
            desc="Validating",
            unit='batch',
            ncols=120,
            leave=False
        )

        for images, targets in pbar:
            images = images.to(self.device)
            batch_size = images.shape[0]

            targets_device = {
                'bboxes': targets['bboxes'].to(self.device),
                'keypoints': targets['keypoints'].to(self.device),
                'num_persons': targets['num_persons'].to(self.device)
            }

            outputs = model(images)
            loss_dict = self.criterion(outputs, targets_device, self.input_size)

            total_loss += loss_dict['loss'].item()
            total_box_loss += loss_dict['box_loss'].item()
            total_obj_loss += loss_dict['obj_loss'].item()
            total_kpt_loss += loss_dict['kpt_loss'].item()

            # 转换 GT 格式并进行评估 - 逐图像处理
            for b in range(batch_size):
                n_persons = targets['num_persons'][b].item()

                # 对单张图像解码预测
                single_outputs = [out[b:b+1] for out in outputs]
                pred_bboxes, pred_scores, pred_keypoints = model.decode(
                    single_outputs, conf_thresh=0.25, input_size=self.input_size
                )

                # 后处理 (NMS)
                if pred_bboxes.numel() > 0:
                    pred_bboxes, pred_scores, pred_keypoints = postprocess(
                        pred_bboxes, pred_scores, pred_keypoints,
                        conf_thresh=0.25, iou_thresh=0.65
                    )

                # GT: 从归一化坐标转换为像素坐标
                img_h, img_w = self.input_size

                if n_persons > 0:
                    gt_bboxes_norm = targets['bboxes'][b, :n_persons]  # (N, 4) [cx, cy, w, h]
                    gt_kpts_encoded = targets['keypoints'][b, :n_persons]  # (N, K, 3) [dx, dy, v]

                    # 转换 bbox 到像素坐标 [x1, y1, x2, y2]
                    bbox_cx_pixel = gt_bboxes_norm[:, 0] * img_w
                    bbox_cy_pixel = gt_bboxes_norm[:, 1] * img_h
                    bbox_w_pixel = gt_bboxes_norm[:, 2] * img_w
                    bbox_h_pixel = gt_bboxes_norm[:, 3] * img_h

                    gt_boxes_xyxy = torch.stack([
                        bbox_cx_pixel - bbox_w_pixel / 2,
                        bbox_cy_pixel - bbox_h_pixel / 2,
                        bbox_cx_pixel + bbox_w_pixel / 2,
                        bbox_cy_pixel + bbox_h_pixel / 2
                    ], dim=-1).cpu().numpy()

                    # 转换关键点到绝对坐标
                    # gt_kpts_encoded 的 xy 是相对于 bbox 尺寸的偏移 (dx, dy)
                    # 在 dataset 中: dx = (kx - cx) / w, dy = (ky - cy) / h
                    # 因此: kx = cx + dx * w, ky = cy + dy * h
                    gt_kpts_abs = gt_kpts_encoded.clone()
                    # 使用归一化的 bbox 参数
                    bbox_cx_norm = gt_bboxes_norm[:, 0:1]  # (N, 1)
                    bbox_cy_norm = gt_bboxes_norm[:, 1:2]  # (N, 1)
                    bbox_w_norm = gt_bboxes_norm[:, 2:3]   # (N, 1)
                    bbox_h_norm = gt_bboxes_norm[:, 3:4]   # (N, 1)
                    # 先转换为归一化的绝对坐标，再转换为像素坐标
                    gt_kpts_abs[:, :, 0] = (bbox_cx_norm + gt_kpts_encoded[:, :, 0] * bbox_w_norm) * img_w
                    gt_kpts_abs[:, :, 1] = (bbox_cy_norm + gt_kpts_encoded[:, :, 1] * bbox_h_norm) * img_h
                    gt_kpts_abs = gt_kpts_abs.cpu().numpy()

                    # 更新评估器
                    evaluator.update(
                        pred_boxes=pred_bboxes.cpu().numpy(),
                        pred_keypoints=pred_keypoints.cpu().numpy(),
                        gt_boxes=gt_boxes_xyxy,
                        gt_keypoints=gt_kpts_abs
                    )
                else:
                    gt_boxes_xyxy = None
                    gt_kpts_abs = None

                # 收集可视化数据
                if sample_idx in vis_indices:
                    vis_data['images'].append(images[b:b+1].cpu())
                    vis_data['pred_bboxes'].append(pred_bboxes.cpu().numpy() if pred_bboxes.numel() > 0 else None)
                    vis_data['pred_keypoints'].append(pred_keypoints.cpu().numpy() if pred_keypoints.numel() > 0 else None)
                    vis_data['pred_scores'].append(pred_scores.cpu().numpy() if pred_scores.numel() > 0 else None)
                    vis_data['gt_bboxes'].append(gt_boxes_xyxy)
                    vis_data['gt_keypoints'].append(gt_kpts_abs)

                sample_idx += 1

        # 计算评估指标
        eval_metrics = evaluator.compute()

        # 计算平均损失
        metrics = {
            'loss': total_loss / num_batches,
            'box_loss': total_box_loss / num_batches,
            'obj_loss': total_obj_loss / num_batches,
            'kpt_loss': total_kpt_loss / num_batches,
            'PCK@0.2': eval_metrics['PCK@0.2'],
            'AP50': eval_metrics['AP50'],
            'OKS': eval_metrics['OKS']
        }

        self.logger.info(
            f'Validation - Loss: {metrics["loss"]:.4f} '
            f'(box: {metrics["box_loss"]:.4f}, obj: {metrics["obj_loss"]:.4f}, kpt: {metrics["kpt_loss"]:.4f}) '
            f'| PCK@0.2: {metrics["PCK@0.2"]:.4f}, AP50: {metrics["AP50"]:.4f}, OKS: {metrics["OKS"]:.4f}'
        )

        # 保存可视化结果
        if vis_data['images']:
            from ..utils.visualization.plots import visualize_val_samples
            import torch as th

            # 拼接所有图像
            all_images = th.cat(vis_data['images'], dim=0)

            vis_path = self.output_dir / 'vis' / f'epoch_{epoch+1}.jpg'
            visualize_val_samples(
                images=all_images,
                pred_bboxes_list=vis_data['pred_bboxes'],
                pred_keypoints_list=vis_data['pred_keypoints'],
                pred_scores_list=vis_data['pred_scores'],
                gt_bboxes_list=vis_data['gt_bboxes'],
                gt_keypoints_list=vis_data['gt_keypoints'],
                save_path=str(vis_path),
                max_samples=num_vis_samples
            )
            self.logger.info(f'Saved visualization to {vis_path}')

        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'global_step': self.global_step,
            'config': self.config,
        }

        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()

        # 保存最新
        torch.save(checkpoint, self.output_dir / 'weights' / 'last.pt')

        # 保存最佳
        if is_best:
            torch.save(checkpoint, self.output_dir / 'weights' / 'best.pt')
            self.logger.info(f'Saved best model with loss: {self.best_metric:.4f}')

    def _log_config(self):
        """记录训练配置到日志"""
        self.logger.info("=" * 60)
        self.logger.info("YOLO-Pose Training Configuration")
        self.logger.info("=" * 60)

        # 数据配置
        data_cfg = self.config['data']
        self.logger.info(f"Data: {data_cfg['yaml_path']}")
        self.logger.info(f"Input size: {data_cfg['input_size']}")

        # 模型配置
        model_cfg = self.config['model']
        self.logger.info(f"Backbone: {model_cfg['backbone']}")
        self.logger.info(f"Num keypoints: {model_cfg['num_keypoints']}")
        self.logger.info(f"Pretrained: {model_cfg.get('pretrained', None)}")

        # 训练配置
        train_cfg = self.config['training']
        self.logger.info(f"Epochs: {train_cfg['epochs']}")
        self.logger.info(f"Batch size: {train_cfg['batch_size']}")
        self.logger.info(f"Learning rate: {train_cfg['lr']}")
        self.logger.info(f"Weight decay: {train_cfg['weight_decay']}")
        self.logger.info(f"LR scheduler: {train_cfg['lr_scheduler']}")
        self.logger.info(f"Warmup epochs: {train_cfg['warmup_epochs']}")
        self.logger.info(f"EMA: {train_cfg['ema']}")
        if train_cfg['ema']:
            self.logger.info(f"EMA decay: {train_cfg['ema_decay']}")
        self.logger.info(f"Freeze backbone: {train_cfg['freeze_backbone']}")
        if train_cfg['freeze_backbone']:
            self.logger.info(f"Freeze epochs: {train_cfg['freeze_epochs']}")

        # 恢复信息
        if self.config.get('resume'):
            self.logger.info(f"Resume from: {self.config['resume']}")
        if self.config.get('weights'):
            self.logger.info(f"Load weights: {self.config['weights']}")

        self.logger.info("=" * 60)

    def train(self):
        """完整训练流程"""
        train_cfg = self.config['training']
        output_cfg = self.config['output']

        # 记录训练配置
        self._log_config()

        self.logger.info(f"Starting training for {train_cfg['epochs']} epochs")
        self.logger.info(f"Output dir: {self.output_dir}")

        for epoch in range(self.start_epoch, train_cfg['epochs']):
            # 解冻 backbone
            if train_cfg['freeze_backbone'] and epoch == train_cfg['freeze_epochs']:
                self.model.unfreeze_backbone()
                self.logger.info(f'Unfreezing backbone at epoch {epoch}')

            # 训练
            train_losses = self.train_one_epoch(epoch)

            # 验证
            if (epoch + 1) % output_cfg['val_interval'] == 0:
                metrics = self.validate(epoch)

                # 保存最佳 (使用总损失作为指标)
                is_best = metrics['loss'] < self.best_metric
                if is_best:
                    self.best_metric = metrics['loss']

                self.save_checkpoint(epoch, is_best)

            # 定期保存
            elif (epoch + 1) % output_cfg['save_interval'] == 0:
                self.save_checkpoint(epoch)

        # 保存最终模型
        self.save_checkpoint(train_cfg['epochs'] - 1)
        self.logger.info(f'Training completed. Best loss: {self.best_metric:.4f}')


# 为了向后兼容，保留 Trainer 别名
Trainer = YOLOPoseTrainer
