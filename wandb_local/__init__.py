"""
WandB Local - A drop-in replacement for wandb that logs to local filesystem
Compatible with most wandb SDK APIs but without login requirements
"""

from .core import init, log, finish, save, watch, alert, config
from .classes import Table, Image, Audio, Video, Artifact
from .sweep import sweep
from .utils import set_dir, get_dir, reset, get_history, get_summary

__version__ = "1.0.0"
__all__ = [
    # Core APIs
    "init", "log", "finish", "save", "watch", "alert", "config",
    # Classes
    "Table", "Image", "Audio", "Video", "Artifact",
    # Advanced features
    "sweep",
    # Utils
    "set_dir", "get_dir", "reset", "get_history", "get_summary"
]