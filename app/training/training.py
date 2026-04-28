from app.constants import TEST_SPLIT_STRUCTURED_DIR, TRAIN_SPLIT_STRUCTURED_DIR
from app.converters.mlx_data import MLXDataConverter
from app.models.trainer_mlx import MLXTrainerService
from app.training.config import training_settings


def run_full_optimization_phase():
    """
    Main entry point for the fine-tuning phase.
    Orchestrates data conversion and sequential training
    using consolidated configuration.
    """
    print("=== NLP Transformers: Fine-Tuning Phase (MLX) ===")

    # 1. Prepare Data for MLX
    converter = MLXDataConverter()
    print("[+] Preparing training and testing files for MLX...")
    converter.convert_split(TRAIN_SPLIT_STRUCTURED_DIR)
    converter.convert_split(TEST_SPLIT_STRUCTURED_DIR)

    # 2. Fine-tune QWEN
    qwen_cfg = training_settings.qwen_config
    qwen_trainer = MLXTrainerService(qwen_cfg["model"])
    qwen_trainer.train(qwen_cfg)

    # 3. Fine-tune GEMMA
    gemma_cfg = training_settings.gemma_config
    gemma_trainer = MLXTrainerService(gemma_cfg["model"])
    gemma_trainer.train(gemma_cfg)


if __name__ == "__main__":
    run_full_optimization_phase()
