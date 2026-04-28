import os
from typing import Any, Optional

from mlx_lm import lora

from app.config import settings


class MLXTrainerService:
    """
    Service to handle model fine-tuning using MLX LoRA.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id

    def _prepare_args(self, custom_args: Optional[dict[str, Any]] = None):
        """
        Creates a complete argument object by simulating CLI arguments
        and adding missing defaults for programmatic usage.
        """
        parser = lora.build_parser()

        # Mandatory CLI flags
        cli_args = [
            "--model",
            self.model_id,
            "--train",
            "--data",
            str(settings.structured_path),
            "--adapter-path",
            f".adapters/{self.model_id.split('/')[-1]}_lora",
        ]

        if custom_args:
            for key, value in custom_args.items():
                flag = "--" + key.replace("_", "-")
                cli_args.extend([flag, str(value)])

        args = parser.parse_args(cli_args)

        # Manually fill attributes that don't have defaults in the parser
        # but are required
        defaults = {
            "num_layers": 16,
            "batch_size": 4,
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
            "grad_checkpoint": False,
        }

        for key, value in defaults.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)

        # Ensure lora_parameters is properly initialized if missing
        if getattr(args, "lora_parameters", None) is None:
            args.lora_parameters = {
                "rank": getattr(args, "rank", 8),
                "scale": getattr(args, "scale", 20.0),
                "dropout": getattr(args, "dropout", 0.0),
            }

        return args

    def train(self, custom_args: Optional[dict[str, Any]] = None):
        """
        Orchestrates the training process.
        """
        args = self._prepare_args(custom_args)

        print(f"\n--- Starting MLX Fine-Tuning: {self.model_id} ---")
        print(f"[+] Learning Rate: {args.learning_rate}")
        print(f"[+] Batch Size: {args.batch_size}")
        print(f"[+] Adapter Path: {args.adapter_path}")

        os.makedirs(".adapters", exist_ok=True)

        try:
            lora.run(args)
            print(f"[✓] Training completed successfully for {self.model_id}")
        except Exception as e:
            print(f"[!] Training error: {e}")
            raise e
