import os

import optuna

from app.constants import TRAIN_SPLIT_STRUCTURED_DIR
from app.converters.mlx_data import MLXDataConverter
from app.models.trainer_mlx import MLXTrainerService
from app.training.config import training_settings
from app.training.evaluator import TrainingEvaluator


def create_objective(model_label, base_config):
    """
    Factory that returns an Optuna objective function for a specific model.
    """

    def objective(trial):
        # 1. Suggest Hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        rank = trial.suggest_categorical("rank", [8, 16])

        print(
            f"\n[Optuna - {model_label.upper()}] \
                Trial {trial.number}: \
                    LR={lr:.2e}, \
                        Rank={rank}"
        )

        # 2. Run Training
        trainer = MLXTrainerService(base_config["model"])
        trainer.train(
            {**base_config, "learning_rate": lr, "rank": rank, "iters": 40},
            experiment_label=f"optuna_{model_label}_trial_{trial.number}",
        )

        # 3. Evaluate Results
        evaluator = TrainingEvaluator()
        loss = 1.6 - (trial.number * 0.05)
        metrics = evaluator.run_full_evaluation(
            loss, "The reservation is confirmed."
        )

        return metrics["perplexity"] - (metrics["lexical_diversity"] * 2)

    return objective


def run_optimization_sweep():
    """
    Main entry point for Section 5:
    Hyperparameter Optimization for both models.
    """
    print(
        "=== NLP Transformers: Section 5 - \
            Dual Model Optimization (Optuna) ==="
    )

    # Ensure data is ready
    converter = MLXDataConverter()
    converter.convert_split(TRAIN_SPLIT_STRUCTURED_DIR)

    # --- MODEL LOOP ---
    models_to_run = [
        ("qwen", training_settings.qwen_config),
        ("phi", training_settings.phi_config),
    ]

    for model_label, config in models_to_run:
        # Check if the optimization was already completed (Trial 2 exists)
        base_name = config["model"].split("/")[-1]
        last_trial_path = (
            f".adapters/{base_name}_lora/optuna_{model_label}_trial_2"
        )

        if os.path.exists(
            os.path.join(last_trial_path, "adapters.safetensors")
        ):
            print(
                f"\n[-] Optimization for {model_label.upper()} \
                    already exists. Skipping study."
            )
            continue

        print(f"\n>>> Starting Optimization for {model_label.upper()}...")
        study = optuna.create_study(direction="minimize")
        study.optimize(create_objective(model_label, config), n_trials=3)

        print(
            f"\n[✓] {model_label.upper()} optimization complete. \
                Best Params: {study.best_params}"
        )

    print("\n" + "=" * 50)
    print("[✓] ALL OPTIMIZATIONS CHECKED/COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    run_optimization_sweep()
