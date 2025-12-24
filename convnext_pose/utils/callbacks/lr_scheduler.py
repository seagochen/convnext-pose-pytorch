"""
学习率调度器
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class WarmupScheduler(_LRScheduler):
    """带 Warmup 的学习率调度器包装器

    Args:
        optimizer: 优化器
        warmup_epochs: Warmup 轮数
        steps_per_epoch: 每轮步数
        after_scheduler: Warmup 后的调度器
        warmup_method: Warmup 方法 ('linear' 或 'constant')
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int,
                 steps_per_epoch: int,
                 after_scheduler: Optional[_LRScheduler] = None,
                 warmup_method: str = 'linear',
                 last_epoch: int = -1):
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.after_scheduler = after_scheduler
        self.warmup_method = warmup_method
        self.finished_warmup = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup 阶段
            if self.warmup_method == 'linear':
                alpha = (self.last_epoch + 1) / self.warmup_steps
            else:  # constant
                alpha = 1.0
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Warmup 后
            if not self.finished_warmup:
                self.finished_warmup = True
                if self.after_scheduler is not None:
                    self.after_scheduler.base_lrs = self.base_lrs

            if self.after_scheduler is not None:
                return self.after_scheduler.get_last_lr()
            return self.base_lrs

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_steps and self.after_scheduler is not None:
            self.after_scheduler.step()
        super().step(epoch)


class CosineAnnealingWarmupScheduler(_LRScheduler):
    """Cosine Annealing with Warmup

    Args:
        optimizer: 优化器
        warmup_epochs: Warmup 轮数
        total_epochs: 总训练轮数
        steps_per_epoch: 每轮步数
        min_lr: 最小学习率
        warmup_method: Warmup 方法
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int,
                 total_epochs: int,
                 steps_per_epoch: int,
                 min_lr: float = 1e-6,
                 warmup_method: str = 'linear',
                 last_epoch: int = -1):
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.min_lr = min_lr
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup 阶段
            if self.warmup_method == 'linear':
                alpha = (self.last_epoch + 1) / self.warmup_steps
            else:
                alpha = 1.0
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine 退火阶段
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


def build_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_type: str,
                    epochs: int,
                    steps_per_epoch: int,
                    warmup_epochs: int = 5,
                    min_lr: float = 1e-6) -> _LRScheduler:
    """构建学习率调度器

    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型 ('cosine', 'step', 'multi_step')
        epochs: 总训练轮数
        steps_per_epoch: 每轮步数
        warmup_epochs: Warmup 轮数
        min_lr: 最小学习率

    Returns:
        学习率调度器
    """
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            min_lr=min_lr
        )
    elif scheduler_type == 'step':
        # 阶梯衰减: 每 30 轮衰减 0.1
        base_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30 * steps_per_epoch,
            gamma=0.1
        )
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            steps_per_epoch=steps_per_epoch,
            after_scheduler=base_scheduler
        )
    elif scheduler_type == 'multi_step':
        # 多阶段衰减: 170, 200 轮衰减
        milestones = [170 * steps_per_epoch, 200 * steps_per_epoch]
        base_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1
        )
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            steps_per_epoch=steps_per_epoch,
            after_scheduler=base_scheduler
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler
