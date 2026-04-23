"""训练设备：自动选择 CUDA（若可用），否则 CPU。"""

from __future__ import annotations

import os

import torch


def get_train_device(prefer_cuda: bool = True) -> torch.device:
    """
    优先使用 GPU；可通过环境变量 CUDA_VISIBLE_DEVICES 控制可见卡。
    """
    if not prefer_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync_device(tensor_or_module, device: torch.device):
    """将张量或模块移到 device。"""
    return tensor_or_module.to(device)
