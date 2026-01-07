"""Model components for ConvNeXt-Pose."""

from .backbone import (
    ConvNeXt,
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
)
from .pose_head import YOLOPoseHead, FPN, build_pose_head
from .convnext_pose import ConvNeXtPose, create_model
from .loss import HeatmapLoss, AWingLoss

__all__ = [
    # Backbone
    "ConvNeXt",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    # Pose heads
    "YOLOPoseHead",
    "FPN",
    "build_pose_head",
    # Full model
    "ConvNeXtPose",
    "create_model",
    # Losses (for heatmap-based single-person estimation)
    "HeatmapLoss",
    "AWingLoss",
]
