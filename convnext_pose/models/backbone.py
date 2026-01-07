"""
ConvNeXt Backbone for Pose Estimation

基于论文: A ConvNet for the 2020s (https://arxiv.org/abs/2201.03545)
实现了ConvNeXt的各个变体，作为姿态检测的backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class LayerNorm(nn.Module):
    """支持两种数据格式的LayerNorm: channels_last (默认) 或 channels_first

    channels_last对应输入形状 (batch_size, height, width, channels)
    channels_first对应输入形状 (batch_size, channels, height, width)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block

    结构: DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv -> DropPath
    使用Inverted Bottleneck设计 (扩展比例为4)
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # 深度可分离卷积 7x7
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        # Pointwise/1x1 卷积，扩展通道
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        # 压缩通道回原始维度
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # Layer Scale
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt Backbone

    Args:
        in_chans: 输入图像通道数
        num_classes: 分类数（如果为0则不使用分类头）
        depths: 每个stage的block数量
        dims: 每个stage的通道数
        drop_path_rate: DropPath概率
        layer_scale_init_value: Layer Scale初始值
        out_indices: 输出特征图的stage索引（用于FPN等）
    """
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3]):
        super().__init__()
        self.out_indices = out_indices
        self.num_stages = len(depths)

        # Stem: 4x4 conv with stride 4 (patchify)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        # 下采样层 (stage transitions)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # ConvNeXt stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                  layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # 为每个输出stage添加norm层
        self.out_norms = nn.ModuleList()
        for i in out_indices:
            norm = LayerNorm(dims[i], eps=1e-6, data_format="channels_first")
            self.out_norms.append(norm)

        # 分类头（可选）
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.LayerNorm(dims[-1]),
                nn.Linear(dims[-1], num_classes)
            )

        self.apply(self._init_weights)

        # 记录每个stage的输出通道数
        self.out_channels = [dims[i] for i in out_indices]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """提取多尺度特征图"""
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                idx = self.out_indices.index(i)
                outs.append(self.out_norms[idx](x))
        return outs

    def forward(self, x):
        if self.num_classes > 0:
            # 分类模式
            x = self.forward_features(x)[-1]
            x = self.head(x)
            return x
        else:
            # 特征提取模式
            return self.forward_features(x)


# 预定义的ConvNeXt变体
def convnext_tiny(pretrained=False, **kwargs):
    """ConvNeXt-Tiny: 28M params"""
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnext_small(pretrained=False, **kwargs):
    """ConvNeXt-Small: 50M params"""
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnext_base(pretrained=False, **kwargs):
    """ConvNeXt-Base: 89M params"""
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def convnext_large(pretrained=False, **kwargs):
    """ConvNeXt-Large: 198M params"""
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model
