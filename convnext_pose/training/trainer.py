"""
训练器类

封装训练循环逻辑
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ..models import ConvNeXtPose, HeatmapLoss, PAFLoss, build_loss
from ..utils.data import YOLOPoseDataset, build_dataloader
from ..utils.metrics import PoseEvaluator, decode_heatmap
from ..utils.callbacks import build_scheduler, ModelEMA


class Trainer:
    """姿态估计训练器

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
        self.scaler = GradScaler() if config['training']['amp'] else None
        self.ema = self._build_ema() if config['training']['ema'] else None

        # 训练状态
        self.start_epoch = 0
        self.best_metric = 0
        self.global_step = 0

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
        logger = logging.getLogger('trainer')
        logger.setLevel(logging.INFO)

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
            head_type=model_cfg['head_type'],
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
        self.logger.info(f"Model: ConvNeXt-{model_cfg['backbone'].upper()}-{model_cfg['head_type']}")
        self.logger.info(f"Parameters: {params:.2f}M")

        return model

    def _build_criterion(self) -> nn.Module:
        """构建损失函数"""
        loss_type = self.config['training']['loss_type']
        head_type = self.config['model']['head_type']

        if head_type == 'paf':
            criterion = PAFLoss(num_stages=2)
        else:
            criterion = build_loss(loss_type)

        return criterion.to(self.device)

    def _build_optimizer(self) -> optim.Optimizer:
        """构建优化器"""
        train_cfg = self.config['training']

        # 参数分组: backbone 使用较小学习率
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
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

    def _build_dataloaders(self):
        """构建数据加载器"""
        data_cfg = self.config['data']
        train_cfg = self.config['training']

        train_loader = build_dataloader(
            data_yaml=data_cfg['yaml_path'],
            split='train',
            input_size=data_cfg['input_size'],
            output_size=data_cfg['output_size'],
            batch_size=train_cfg['batch_size'],
            num_workers=data_cfg['num_workers'],
        )

        val_loader = build_dataloader(
            data_yaml=data_cfg['yaml_path'],
            split='val',
            input_size=data_cfg['input_size'],
            output_size=data_cfg['output_size'],
            batch_size=train_cfg['batch_size'],
            num_workers=data_cfg['num_workers'],
        )

        self.logger.info(f"Train samples: {len(train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(val_loader.dataset)}")

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
        self.best_metric = checkpoint.get('best_metric', 0)

        if self.ema and 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'])

        if self.scaler and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])

        self.logger.info(f"Resumed from epoch {self.start_epoch}")

    def _load_weights(self, weights_path: str):
        """加载权重"""
        self.logger.info(f"Loading weights from {weights_path}")

        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

    def train_one_epoch(self, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        train_cfg = self.config['training']

        total_loss = 0
        num_batches = len(self.train_loader)

        # 使用 tqdm 显示进度
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{train_cfg['epochs']}",
            unit='batch',
            ncols=120,
            leave=True
        )

        for step, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            heatmaps = targets['heatmap'].to(self.device)
            target_weight = targets.get('target_weight')
            if target_weight is not None:
                target_weight = target_weight.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            if train_cfg['amp']:
                with autocast():
                    outputs = self.model(images)
                    if self.config['model']['head_type'] == 'paf':
                        _, hm_outputs = outputs
                        loss = self.criterion(hm_outputs[-1], heatmaps, target_weight)
                    else:
                        loss = self.criterion(outputs, heatmaps, target_weight)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if self.config['model']['head_type'] == 'paf':
                    _, hm_outputs = outputs
                    loss = self.criterion(hm_outputs[-1], heatmaps, target_weight)
                else:
                    loss = self.criterion(outputs, heatmaps, target_weight)

                loss.backward()
                self.optimizer.step()

            # 更新学习率
            self.scheduler.step()

            # 更新 EMA
            if self.ema:
                self.ema.update(self.model)

            total_loss += loss.item()
            self.global_step += 1

            # 更新进度条
            avg_loss = total_loss / (step + 1)
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.6f}'
            })

        # Epoch 结束后记录日志
        avg_loss = total_loss / num_batches
        self.logger.info(
            f'Epoch [{epoch+1}/{train_cfg["epochs"]}] '
            f'Train Loss: {avg_loss:.4f} '
            f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}'
        )

        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证"""
        model = self.ema.ema if self.ema else self.model
        model.eval()

        total_loss = 0
        evaluator = PoseEvaluator(num_keypoints=self.config['model']['num_keypoints'])

        input_size = self.config['data']['input_size']
        output_size = self.config['data']['output_size']
        stride = input_size[0] // output_size[0]

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
            heatmaps = targets['heatmap'].to(self.device)

            outputs = model(images)

            if self.config['model']['head_type'] == 'paf':
                _, hm_outputs = outputs
                outputs = hm_outputs[-1]

            loss = self.criterion(outputs, heatmaps)
            total_loss += loss.item()

            # 解码预测
            pred_kpts, scores = decode_heatmap(outputs, stride=stride)

            # 更新评估器
            target_kpts = targets['keypoints'].numpy()
            evaluator.update(
                pred_kpts,
                target_kpts[:, :, :2],
                visibility=target_kpts[:, :, 2]
            )

        metrics = evaluator.compute()
        metrics['loss'] = total_loss / len(self.val_loader)

        self.logger.info(
            f'Validation - Loss: {metrics["loss"]:.4f} '
            f'PCK@0.5: {metrics["PCK"]:.4f}'
        )

        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
        }

        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()

        if self.scaler:
            checkpoint['scaler'] = self.scaler.state_dict()

        # 保存最新
        torch.save(checkpoint, self.output_dir / 'weights' / 'last.pt')

        # 保存最佳
        if is_best:
            torch.save(checkpoint, self.output_dir / 'weights' / 'best.pt')

    def train(self):
        """完整训练流程"""
        train_cfg = self.config['training']
        output_cfg = self.config['output']

        self.logger.info(f"Starting training for {train_cfg['epochs']} epochs")
        self.logger.info(f"Output dir: {self.output_dir}")

        for epoch in range(self.start_epoch, train_cfg['epochs']):
            # 解冻 backbone
            if train_cfg['freeze_backbone'] and epoch == train_cfg['freeze_epochs']:
                self.model.unfreeze_backbone()
                self.logger.info(f'Unfreezing backbone at epoch {epoch}')

            # 训练
            train_loss = self.train_one_epoch(epoch)

            # 验证
            if (epoch + 1) % output_cfg['val_interval'] == 0:
                metrics = self.validate(epoch)

                # 保存最佳
                is_best = metrics['PCK'] > self.best_metric
                if is_best:
                    self.best_metric = metrics['PCK']
                    self.logger.info(f'New best PCK: {self.best_metric:.4f}')

                self.save_checkpoint(epoch, is_best)

            # 定期保存
            elif (epoch + 1) % output_cfg['save_interval'] == 0:
                self.save_checkpoint(epoch)

        # 保存最终模型
        self.save_checkpoint(train_cfg['epochs'] - 1)
        self.logger.info(f'Training completed. Best PCK: {self.best_metric:.4f}')
