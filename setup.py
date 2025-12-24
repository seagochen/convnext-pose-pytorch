#!/usr/bin/env python3
"""
ConvNeXt-Pose 安装脚本

安装方式:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取 README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# 读取依赖
requirements = [
    "torch>=1.10.0",
    "torchvision>=0.11.0",
    "numpy>=1.19.0",
    "opencv-python>=4.5.0",
    "pyyaml>=5.4.0",
    "tensorboard>=2.5.0",
    "tqdm>=4.60.0",
]

setup(
    name="convnext-pose",
    version="1.0.0",
    author="ConvNeXt-Pose Authors",
    description="Human Pose Estimation with ConvNeXt Backbone",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xxx/convnext-pose-pytorch",
    packages=find_packages(exclude=["tests", "scripts", "configs", "datasets"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "flake8>=3.9.0",
            "black>=21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "convnext-pose-train=scripts.train:main",
            "convnext-pose-detect=scripts.detect:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="pose estimation, human pose, keypoint detection, convnext, deep learning",
)
