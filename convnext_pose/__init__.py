"""
ConvNeXt-Pose: Human Pose Estimation with ConvNeXt Backbone

A lightweight and efficient human pose estimation framework using
ConvNeXt as the backbone network with support for YOLO data format.
"""

__version__ = "0.1.0"

from .models import (
    ConvNeXt,
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
    PoseHead,
    PAFHead,
    ConvNeXtPose,
)

__all__ = [
    "ConvNeXt",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "PoseHead",
    "PAFHead",
    "ConvNeXtPose",
]
