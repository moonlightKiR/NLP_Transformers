import os
from typing import Any, Optional

from mlx_lm import lora

from app.config import settings
from app.training.constants import adapters_lora_root


class MLXTrainerService:
    """
    Service to handle model fine-tuning using MLX LoRA.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id

    def _prepare_args(self, custom_args: Optional[dict[str, Any]] = None):
        """
        Creates a complete argument object by simulating CLI arguments.
        """
        parser = lora.build_parser()

        # Determine the base adapter path
        default_adapter_path = str(adapters_lora_root(self.model_id))

        cli_args = [
            "--model",
            self.model_id,
            "--train",
            "--data",
            str(settings.structured_path),
            "--adapter-path",
            default_adapter_path,
        ]

        if custom_args:
            for key, value in custom_args.items():
                # Special handling for lora_parameters keys if passed flat
                if key in ["rank", "scale", "dropout"]:
                    continue  # Will be handled in the defaults
                    # getattr logic below

                flag = "--" + key.replace("_", "-")
                if isinstance(value, bool):
                    if value and flag not in cli_args:
                        cli_args.append(flag)
                else:
                    # Update if already exists, else append
                    if flag in cli_args:
                        idx = cli_args.index(flag)
                        cli_args[idx + 1] = str(value)
                    else:
                        cli_args.extend([flag, str(value)])

        args = parser.parse_args(cli_args)

        # Standard defaults
        defaults = {
            "num_layers": 16,
            "batch_size": 1,
            "iters": 100,
            "learning_rate": 1e-5,
            "steps_per_report": 10,
            "steps_per_eval": 200,
            "save_every": 50,
            "max_seq_length": 2048,
            "grad_accumulation_steps": 1,
            "seed": 0,
            "fine_tune_type": "lora",
            "optimizer": "adam",
            "lr_schedule": None,
            "optimizer_config": {},
            "mask_prompt": False,
            "grad_checkpoint": True,
        }

        for key, value in defaults.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)

        # Handle LoRA Rank/Scale overrides from custom_args
        rank = 8
        scale = 20.0
        if custom_args:
            rank = custom_args.get("rank", 8)
            scale = custom_args.get("scale", 20.0)

        if getattr(args, "lora_parameters", None) is None:
            args.lora_parameters = {
                "rank": rank,
                "scale": scale,
                "dropout": 0.0,
            }

        return args

    def train(
        self,
        custom_args: Optional[dict[str, Any]] = None,
        experiment_label: str = "",
    ):
        """
        Orchestrates the training process with experiment tracking.
        """
        args = self._prepare_args(custom_args)

        # If an experiment label is provided, append it to the adapter path
        if experiment_label:
            args.adapter_path = os.path.join(
                args.adapter_path, experiment_label
            )

        adapter_file = os.path.join(args.adapter_path, "adapters.safetensors")
        if os.path.exists(adapter_file):
            print(
                f"[-] Experiment '{experiment_label}'\
                already exists at {args.adapter_path}. Skipping."
            )
            return

        print(
            f"\n--- Starting MLX Fine-Tuning:\
            {self.model_id} [{experiment_label}] ---"
        )
        print(
            f"[+] LR: {args.learning_rate}\
            Batch: {args.batch_size}\
            Rank: {args.lora_parameters['rank']}"
        )

        os.makedirs(args.adapter_path, exist_ok=True)

        try:
            lora.run(args)
            print(f"[✓] Completed: {self.model_id} ({experiment_label})")
        except Exception as e:
            print(f"[!] Error in experiment '{experiment_label}': {e}")
            # We don't raise here to allow
            # other experiments in the loop to continue
