import os

from app.utils.hardware import HardwareDetector


class TrainerFactory:
    """
    Factory to instantiate the appropriate trainer service based on hardware.
    Adheres to Open/Closed Principle.
    """

    @staticmethod
    def get_trainer(model_id: str):
        # Allow explicit override via env var (useful in CI/Colab)
        device = (
            os.environ.get("FORCE_BACKEND")
            or HardwareDetector.get_device_type()
        )

        # Apple Silicon / MLX
        if device == "mlx":
            print("[Hardware] Apple Silicon detected. Using MLX backend.")
            # Lazy import to avoid importing MLX on non-Apple environments
            from app.models.trainer_mlx import MLXTrainerService

            return MLXTrainerService(model_id)

        # NVIDIA CUDA (or general torch-based trainer)
        if device == "cuda":
            print("[Hardware] NVIDIA GPU detected. Using CUDA/PEFT backend.")
            from app.models.trainer_cuda import CUDATrainerService

            return CUDATrainerService(model_id)

        # For other environments (mps, cpu)
        # prefer the CUDA-style trainer if available
        if device in ("mps", "cpu"):
            print(
                f"[Hardware] Backend '{device}' detected. "
                f"Using generic CUDA-style trainer (CPU/MPS compatible)."
            )
            try:
                from app.models.trainer_cuda import CUDATrainerService

                return CUDATrainerService(model_id)
            except Exception:
                # Fallback to MLX trainer only if CUDA trainer not available
                try:
                    from app.models.trainer_mlx import MLXTrainerService

                    return MLXTrainerService(model_id)
                except Exception as e:
                    raise RuntimeError(f"No trainer backend available: {e}")

        # If none matched, attempt graceful import sequence
        try:
            from app.models.trainer_cuda import CUDATrainerService

            return CUDATrainerService(model_id)
        except Exception:
            from app.models.trainer_mlx import MLXTrainerService

            return MLXTrainerService(model_id)
