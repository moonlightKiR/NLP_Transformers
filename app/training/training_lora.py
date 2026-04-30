import os

import optuna

from app.converters.mlx_data import MLXDataConverter
from app.models.trainer_mlx import MLXTrainerService
from app.training.config import training_settings


def _load_best_optuna_params(model_label: str, base_config: dict) -> dict:
    """
    Load the best completed Optuna parameters for a model and merge them
    over the base LoRA config. Falls back cleanly when the study is missing
    or empty.
    """
    resolved_config = dict(base_config)
    db_path = os.path.join("results", "optuna", f"optuna_{model_label}.db")

    if not os.path.exists(db_path):
        print(
            f"[Optuna] No database found for {model_label.upper()} at "
            f"{db_path}. Using base LoRA config."
        )
        return resolved_config

    try:
        study = optuna.load_study(
            study_name=f"optimization_{model_label}",
            storage=f"sqlite:///{db_path}",
        )
    except Exception as e:
        print(
            f"[Optuna] Failed to load study for {model_label.upper()}: {e}. "
            "Using base LoRA config."
        )
        return resolved_config

    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        print(
            f"[Optuna] Study for {model_label.upper()} has no completed "
            "trials. Using base LoRA config."
        )
        return resolved_config

    best_params = study.best_params
    resolved_config.update(
        {
            "learning_rate": best_params["learning_rate"],
            "rank": best_params["rank"],
        }
    )
    print(
        f"[Optuna] Using best params for {model_label.upper()}: "
        f"learning_rate={best_params['learning_rate']:.2e}, "
        f"rank={best_params['rank']}"
    )
    return resolved_config


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
    llama_trainer = MLXTrainerService(training_settings.llama_config["model"])
    qwen_final_config = _load_best_optuna_params(
        "qwen", training_settings.qwen_config
    )
    llama_final_config = _load_best_optuna_params(
        "llama", training_settings.llama_config
    )

    print("\n[SECTION 7] Starting final LoRA training for Qwen...")
    qwen_trainer.train(qwen_final_config, experiment_label="final_lora")

    print("\n[SECTION 7] Starting final LoRA training for Llama...")
    llama_trainer.train(llama_final_config, experiment_label="final_lora")

    print("\n[✓] LoRA Challenge Phase Completed.")


if __name__ == "__main__":
    run_lora_challenge_phase()
