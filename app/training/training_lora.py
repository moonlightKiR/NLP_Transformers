from app.converters.mlx_data import MLXDataConverter
from app.models.trainer_mlx import MLXTrainerService
from app.training.config import training_settings


def run_lora_challenge_phase():
    """
    Main entry point for Section 7: LoRA Efficient Fine-Tuning.
    Executes final training with LoRA optimizations.
    """
    print(
        "=== NLP Transformers: Section 7 - \
        LoRA Challenge Phase (Efficient FT) ==="
    )

    # 1. Ensure Data is ready
    converter = MLXDataConverter()
    converter.ensure_train_valid()

    # 2. Final Training with Optimized Parameters
    qwen_trainer = MLXTrainerService(training_settings.qwen_config["model"])
    phi_trainer = MLXTrainerService(training_settings.phi_config["model"])

    print("\n[SECTION 7] Starting final LoRA training for Qwen...")
    qwen_trainer.train(
        training_settings.qwen_config, experiment_label="final_lora"
    )

    print("\n[SECTION 7] Starting final LoRA training for Phi...")
    phi_trainer.train(
        training_settings.phi_config, experiment_label="final_lora"
    )

    print("\n[✓] LoRA Challenge Phase Completed.")


if __name__ == "__main__":
    run_lora_challenge_phase()
