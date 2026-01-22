"""Shared Modal configuration for wall-e-world."""

from functools import wraps
from pathlib import Path
from typing import Callable

import modal

CUDA_VERSION = "12.8.1"
PYTHON_VERSION = "3.10"
MOUNT_PATH = Path("/mnt")
SRC_LOCAL_DIR = Path(__file__).parent.parent / "src"
MODAL_LOCAL_DIR = Path(__file__).parent

VOLUME = modal.Volume.from_name("world-model-eval", create_if_missing=True, version=2)

BASE_PACKAGES = (
    "torch>=2.5.0 torchvision>=0.20.0 accelerate>=1.10.0 "
    "diffusers==0.35.1 einops==0.8.1 numpy==1.26.4 tqdm==4.67.1 "
    "pillow==11.3.0 opencv-python==4.11.0.86 pytorchvideo==0.1.5 "
    "fire==0.7.1 imageio==2.37.0 matplotlib==3.10.6 mediapy==1.2.4 "
    "tensorboardX backports.strenum==1.3.1 openai moviepy"
)

DATA_PACKAGES = "tensorflow==2.15.0 tensorflow-datasets==4.9.3"

_base_image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu22.04",
        add_python=PYTHON_VERSION,
    )
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "git")
    .run_commands("python -m pip install --no-cache-dir -U pip uv")
    .run_commands(f"uv pip install --system {BASE_PACKAGES}")
    .env(
        {
            "HF_HOME": "/mnt/hf_cache",
            "CHECKPOINT_DIR": "/mnt/checkpoints",
            "DATA_DIR": "/mnt/data",
            "PYTHONPATH": "/root",
            # NCCL settings for robustness in cloud environments
            "NCCL_TIMEOUT": "1800",  # 30 min timeout (default is 10 min)
            "NCCL_ASYNC_ERROR_HANDLING": "1",  # Enable async error handling
            "NCCL_IB_DISABLE": "1",  # Disable InfiniBand (not available on Modal)
            "NCCL_DEBUG": "WARN",  # Show warnings only
            "TORCH_NCCL_BLOCKING_WAIT": "0",  # Non-blocking for better error recovery
        }
    )
)

image = _base_image.add_local_dir(
    str(SRC_LOCAL_DIR / "wall_e_world"), "/root/wall_e_world"
).add_local_file(str(MODAL_LOCAL_DIR / "modal_config.py"), "/root/modal_config.py")

data_image = (
    _base_image.run_commands(f"uv pip install --system {DATA_PACKAGES}")
    .add_local_dir(str(SRC_LOCAL_DIR / "wall_e_world"), "/root/wall_e_world")
    .add_local_file(str(MODAL_LOCAL_DIR / "modal_config.py"), "/root/modal_config.py")
)


def setup_modal_env():
    """Setup environment before running Modal functions."""
    import os

    for path in ["/mnt/checkpoints", "/mnt/data", "/mnt/hf_cache"]:
        os.makedirs(path, exist_ok=True)


def with_modal(
    app_name: str,
    timeout: int = 1,
    cpu: int | None = None,
    gpu: str | None = None,
    img: modal.Image = image,
) -> Callable:
    """
    Usage:
        @with_modal("world-model-train", timeout=24, gpu="A100-80GB")
        def foo():
            pass
    Args:
        app_name: Name of the Modal app
        timeout: Timeout in hours (default: 1 hour)
        cpu: CPU count
        gpu: GPU type (e.g., "A100-80GB", "A100-40GB")
    """

    kwargs = {
        "image": img,
        "volumes": {MOUNT_PATH: VOLUME},
        "secrets": [modal.Secret.from_name("huggingface-secret")],
        "timeout": timeout * 60 * 60,
    }

    if gpu:
        kwargs["gpu"] = gpu
    if cpu:
        kwargs["cpu"] = cpu

    app = modal.App(app_name)

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kw):
            setup_modal_env()
            return fn(*args, **kw)

        return app.function(**kwargs)(wrapper)

    return decorator
