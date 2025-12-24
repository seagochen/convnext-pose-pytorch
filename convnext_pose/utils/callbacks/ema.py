"""
指数移动平均 (Exponential Moving Average)
"""

import copy
import torch
import torch.nn as nn
from typing import Optional


class ModelEMA:
    """模型参数的指数移动平均

    用于提高模型的泛化性能和稳定性

    Args:
        model: 原始模型
        decay: 衰减率 (越接近 1 越平滑)
        updates: 初始更新次数 (用于预热)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, updates: int = 0):
        self.ema = copy.deepcopy(model).eval()  # EMA 模型
        self.updates = updates
        self.decay = decay

        # 禁用 EMA 模型的梯度
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model: nn.Module):
        """更新 EMA 参数

        Args:
            model: 当前模型
        """
        with torch.no_grad():
            self.updates += 1

            # 使用预热衰减
            d = self.decay * (1 - pow(1 - self.decay, self.updates))

            # 更新参数
            model_params = dict(model.named_parameters())
            ema_params = dict(self.ema.named_parameters())

            for name, param in model_params.items():
                if param.requires_grad:
                    ema_params[name].data.mul_(d).add_(param.data, alpha=1 - d)

            # 更新 buffer (如 BatchNorm 的 running_mean/var)
            model_buffers = dict(model.named_buffers())
            ema_buffers = dict(self.ema.named_buffers())

            for name in model_buffers:
                if name in ema_buffers:
                    ema_buffers[name].data.copy_(model_buffers[name].data)

    def update_attr(self, model: nn.Module, include: tuple = (), exclude: tuple = ('process_group', 'reducer')):
        """更新 EMA 模型的属性

        Args:
            model: 当前模型
            include: 要包含的属性
            exclude: 要排除的属性
        """
        for k, v in model.__dict__.items():
            if (include and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(self.ema, k, v)

    def state_dict(self) -> dict:
        """获取状态字典"""
        return {
            'ema': self.ema.state_dict(),
            'updates': self.updates,
            'decay': self.decay
        }

    def load_state_dict(self, state_dict: dict):
        """加载状态字典"""
        self.ema.load_state_dict(state_dict['ema'])
        self.updates = state_dict.get('updates', 0)
        self.decay = state_dict.get('decay', self.decay)

    def __call__(self, *args, **kwargs):
        """前向传播"""
        return self.ema(*args, **kwargs)

    def eval(self):
        """设置为评估模式"""
        self.ema.eval()
        return self

    def train(self, mode: bool = True):
        """设置模式 (EMA 模型始终为 eval)"""
        # EMA 模型始终保持 eval 模式
        self.ema.eval()
        return self

    def cuda(self, device: Optional[int] = None):
        """移动到 GPU"""
        self.ema.cuda(device)
        return self

    def to(self, *args, **kwargs):
        """移动模型"""
        self.ema.to(*args, **kwargs)
        return self


def de_parallel(model: nn.Module) -> nn.Module:
    """移除模型的并行包装

    Args:
        model: 可能被包装的模型

    Returns:
        原始模型
    """
    return model.module if hasattr(model, 'module') else model
