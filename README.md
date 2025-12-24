# ConvNeXt-Pose

基于 ConvNeXt 的人体姿态估计模型，支持 YOLO 数据格式。

## 特性

- **ConvNeXt Backbone**: 支持 tiny/small/base/large 四种规模
- **多种检测头**:
  - `heatmap`: 热图回归（类似 SimpleBaseline/HRNet）
  - `paf`: Part Affinity Fields（类似 OpenPose）
  - `yolo`: YOLO 风格检测+关键点（类似 YOLOv8-Pose）
- **YOLO 数据格式**: 支持标准 YOLO Pose 格式数据集
- **纯命令行配置**: 无需 YAML 配置文件
- **完整训练流程**: 支持混合精度、EMA、学习率调度
- **丰富的损失函数**: MSE、OKS、Wing Loss、AWing Loss 等
- **DARK 解码**: 支持亚像素精度的关键点解码

## 项目结构

```
convnext-pose-pytorch/
├── setup.py                      # 包安装脚本
├── requirements.txt
├── README.md
│
├── convnext_pose/                # 主核心包
│   ├── models/                   # 模型模块
│   │   ├── backbone.py           # ConvNeXt backbone
│   │   ├── pose_head.py          # 检测头
│   │   ├── convnext_pose.py      # 完整模型
│   │   └── loss.py               # 损失函数
│   │
│   ├── training/                 # 训练模块
│   │   ├── config.py             # 命令行配置
│   │   ├── trainer.py            # 训练器类
│   │   └── evaluator.py          # 评估器类
│   │
│   └── utils/                    # 工具模块
│       ├── data/                 # 数据相关
│       │   ├── dataset.py        # YOLO 格式数据集
│       │   ├── transforms.py     # 数据增强
│       │   └── path_utils.py     # 路径工具
│       ├── callbacks/            # 回调
│       │   ├── lr_scheduler.py   # 学习率调度
│       │   └── ema.py            # EMA
│       ├── metrics/              # 指标
│       │   └── metrics.py        # PCK, OKS, AP
│       └── visualization/        # 可视化
│           └── plots.py          # 绘图工具
│
└── scripts/                      # 脚本目录
    ├── train.py                  # 训练入口
    └── detect.py                 # 推理入口
```

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 安装包（开发模式）
pip install -e .
```

## YOLO Pose 数据格式

### 标签格式

每行一个人，格式如下：
```
class_id x_center y_center width height kpt1_x kpt1_y kpt1_v kpt2_x kpt2_y kpt2_v ...
```

- 所有坐标归一化到 [0, 1]
- kpt_v: 0=不存在, 1=遮挡, 2=可见
- 17 个 COCO 关键点 = 5 + 17*3 = 56 个值

### 数据集目录结构

```
dataset/
├── images/
│   ├── train/*.jpg
│   └── val/*.jpg
├── labels/
│   ├── train/*.txt
│   └── val/*.txt
└── dataset.yaml
```

### dataset.yaml 配置示例

```yaml
path: /path/to/dataset
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
nc: 1
names: ['person']
```

## 快速开始

### 训练

```bash
python scripts/train.py \
    --data /path/to/dataset.yaml \
    --backbone tiny \
    --head-type heatmap \
    --img-size 256 192 \
    --batch-size 32 \
    --epochs 210 \
    --lr 0.001 \
    --amp \
    --ema \
    --output-dir ./runs/train
```

### 推理

```bash
# 单张图像
python scripts/detect.py \
    --weights runs/train/best.pt \
    --source image.jpg \
    --show

# 视频
python scripts/detect.py \
    --weights runs/train/best.pt \
    --source video.mp4 \
    --output result.mp4

# 目录批量处理
python scripts/detect.py \
    --weights runs/train/best.pt \
    --source images/ \
    --output results/

# 摄像头
python scripts/detect.py \
    --weights runs/train/best.pt \
    --camera 0 \
    --show
```

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | (必需) | YOLO 数据集配置文件 |
| `--backbone` | tiny | tiny/small/base/large |
| `--head-type` | heatmap | heatmap/paf/yolo |
| `--img-size` | 256 192 | 输入尺寸 (H W) |
| `--batch-size` | 32 | 批大小 |
| `--epochs` | 210 | 训练轮数 |
| `--lr` | 0.001 | 学习率 |
| `--amp` | False | 混合精度训练 |
| `--ema` | False | 指数移动平均 |
| `--freeze-backbone` | False | 冻结 backbone |
| `--output-dir` | ./runs/train | 输出目录 |

## 模型配置

### Backbone 选项

| Backbone | 参数量 | 说明 |
|----------|--------|------|
| tiny     | ~28M   | 轻量级，适合实时应用 |
| small    | ~50M   | 平衡性能和速度 |
| base     | ~89M   | 较高精度 |
| large    | ~198M  | 最高精度 |

### 检测头选项

| Head Type | 说明 |
|-----------|------|
| heatmap   | 直接回归热图，简单高效 |
| paf       | Part Affinity Fields，支持多人检测 |
| yolo      | YOLO 风格，同时输出检测框和关键点 |

## 代码示例

```python
import torch
from convnext_pose import ConvNeXtPose

# 创建模型
model = ConvNeXtPose(
    backbone='tiny',
    head_type='heatmap',
    num_keypoints=17
)

# 推理
x = torch.randn(1, 3, 256, 192)
heatmaps = model(x)  # (1, 17, 64, 48)

# 解码关键点
from convnext_pose.utils.metrics import decode_heatmap
keypoints, scores = decode_heatmap(heatmaps, stride=4)
```

## COCO 关键点

```
0: nose
1: left_eye      2: right_eye
3: left_ear      4: right_ear
5: left_shoulder 6: right_shoulder
7: left_elbow    8: right_elbow
9: left_wrist    10: right_wrist
11: left_hip     12: right_hip
13: left_knee    14: right_knee
15: left_ankle   16: right_ankle
```

## 参考

- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) - A ConvNet for the 2020s
- [SimpleBaseline](https://github.com/microsoft/human-pose-estimation.pytorch) - Simple Baselines for Human Pose Estimation
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) - Real-time multi-person keypoint detection
- [YOLOv8-Pose](https://github.com/ultralytics/ultralytics) - YOLO with pose estimation
