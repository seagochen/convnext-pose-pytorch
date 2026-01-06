"""
ConvNeXt-Pose: End-to-End Multi-Person Pose Estimation with ConvNeXt Backbone

A lightweight and efficient multi-person pose estimation framework using
ConvNeXt as the backbone network with YOLO-Pose style detection head.
"""

__version__ = "0.2.0"

from .models import (
    ConvNeXt,
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
    YOLOPoseHead,
    FPN,
    build_pose_head,
    ConvNeXtPose,
    create_model,
)

__all__ = [
    "ConvNeXt",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "YOLOPoseHead",
    "FPN",
    "build_pose_head",
    "ConvNeXtPose",
    "create_model",
]
