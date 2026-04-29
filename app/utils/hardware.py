import importlib.util
import platform

import torch


class HardwareDetector:
    """
    Utility to detect the available hardware backend.
    Adheres to Single Responsibility Principle.
    """

    @staticmethod
    def get_device_type() -> str:
        """Returns 'mlx' (Apple Silicon), 'cuda' (NVIDIA), or 'cpu'."""
        # 1. Check for Apple Silicon / MLX Compatibility
        if platform.system() == "Darwin" and platform.processor() == "arm":
            # Extra check: mlx only runs on Apple Silicon
            if importlib.util.find_spec("mlx.core") is not None:
                return "mlx"

        # 2. Check for NVIDIA CUDA
        if torch.cuda.is_available():
            return "cuda"

        # 3. Check for Apple MPS (via PyTorch)
        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    @staticmethod
    def get_total_ram_gb() -> float:
        """Returns total system RAM in gigabytes."""
        import psutil

        return psutil.virtual_memory().total / (1024**3)

    @staticmethod
    def is_apple_silicon() -> bool:
        return HardwareDetector.get_device_type() == "mlx"

    @staticmethod
    def is_nvidia() -> bool:
        return HardwareDetector.get_device_type() == "cuda"
