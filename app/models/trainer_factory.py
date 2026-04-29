from app.models.trainer_mlx import MLXTrainerService
from app.utils.hardware import HardwareDetector


class TrainerFactory:
    """
    Factory to instantiate the appropriate trainer service based on hardware.
    Adheres to Open/Closed Principle.
    """

    @staticmethod
    def get_trainer(model_id: str):
        device = HardwareDetector.get_device_type()

        if device == "mlx":
            print("[Hardware] Apple Silicon detected. Using MLX backend.")
            return MLXTrainerService(model_id)

        if device == "cuda":
            print("[Hardware] NVIDIA GPU detected. Using CUDA/PEFT backend.")
            # Lazy import to avoid MLX errors
            # in CUDA environments and viceversa
            from app.models.trainer_cuda import CUDATrainerService

            return CUDATrainerService(model_id)

        print(
            f"[Hardware] Backend '{device}' not optimized for training. \
            Falling back to MLX structure."
        )
        return MLXTrainerService(model_id)
