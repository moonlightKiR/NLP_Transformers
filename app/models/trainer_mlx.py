import os
import traceback
from typing import Any, Optional

import mlx.core as mx
from mlx_lm import lora

from app.config import settings
from app.training.constants import adapters_lora_root
from app.utils.hardware import HardwareDetector


class MLXTrainerService:
    """
    Service to handle model fine-tuning using MLX LoRA.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._setup_memory_safeguards()

    def _setup_memory_safeguards(self):
        """Configures Metal cache limits to prevent system instability."""
        if HardwareDetector.get_device_type() == "mlx":
            total_ram = HardwareDetector.get_total_ram_gb()
            # Reserve 6GB for the OS, use 70% of the rest for MLX cache
            safe_limit = max(4, int((total_ram - 6) * 0.7))
            print(
                f"[Memory] Setting Metal cache limit to {safe_limit}GB "
                f"(Total: {int(total_ram)}GB, Reserved for OS: 6GB)"
            )
            mx.metal.set_cache_limit(safe_limit * 1024 * 1024 * 1024)

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

        # Standard defaults optimized for 18GB RAM
        defaults = {
            "num_layers": 16,
            "batch_size": 1,
            "iters": 100,
            "learning_rate": 1e-5,
            "steps_per_report": 10,
            "steps_per_eval": 200,
            "val_batches": 5,  # Reduced to save memory during eval
            "save_every": 50,
            "max_seq_length": 768,  # Reduced from 2048 to prevent OOM
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
            Rank: {args.lora_parameters['rank']}\
            SeqLen: {args.max_seq_length}"
        )

        os.makedirs(args.adapter_path, exist_ok=True)

        try:
            lora.run(args)
            print(f"[✓] Completed: {self.model_id} ({experiment_label})")
        except FileNotFoundError as e:
            # MLX raises FileNotFoundError when no safetensors are found
            print(f"[!] FileNotFoundError during training: {e}")

            # Generic fallback:
            # download full HF snapshot into .models/hf_models/
            from app.models.config import model_settings
            from app.models.downloader import ModelDownloader, ModelService

            # Map "org/name" to "name" for the directory
            model_folder_name = self.model_id.split("/")[-1]
            local_dir = model_settings.hf_models_dir / model_folder_name

            print(
                f"[!] Attempting to download full snapshot for \
                {self.model_id}..."
            )
            model_service = ModelService(ModelDownloader())

            snapshot_path = model_service.ensure_hf_snapshot(
                repo_id=self.model_id,
                target_dir=local_dir,
                allow_patterns=[
                    "*.safetensors",
                    "model.safetensors.index.json",
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ],
            )

            if snapshot_path:
                args.model = snapshot_path
                print(
                    f"[+] Retrying training with local snapshot at \
                    {snapshot_path}..."
                )
                lora.run(args)
                print(
                    f"[✓] Completed after fallback: \
                    {self.model_id} ({experiment_label})"
                )
                return

            print(traceback.format_exc())
        except Exception as e:
            print(f"[!] Error in experiment '{experiment_label}': {e}")
            print(traceback.format_exc())
            # We don't raise here to allow
            # other experiments in the loop to continue
        finally:
            # Crucial: clear Metal cache after each run
            # to free memory for Optuna/next trial
            print("[Memory] Clearing Metal cache...")
            mx.metal.clear_cache()
