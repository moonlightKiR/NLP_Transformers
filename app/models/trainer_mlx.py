import gc
import os
import subprocess
import sys
from typing import Any, Optional

import mlx.core as mx
import yaml

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
            # For 18GB RAM, be very specific.
            # We want to leave enough room for the model + activations
            # but limit the "slack" cache that MLX keeps.
            reserved = 8.0  # Increase system reservation to 8GB
            safe_limit = max(2, int((total_ram - reserved) * 0.5))
            print(
                f"[Memory] Setting Metal cache limit to {safe_limit}GB "
                f"(Total: {int(total_ram)}GB, Reserved: {reserved}GB)"
            )
            # Use non-deprecated API
            mx.set_cache_limit(safe_limit * 1024 * 1024 * 1024)

    def _prepare_config(
        self,
        custom_args: Optional[dict[str, Any]] = None,
        experiment_label: str = "",
    ):
        """
        Creates a dictionary for the YAML configuration file.
        """
        # Determine the base adapter path
        default_adapter_path = str(adapters_lora_root(self.model_id))
        if experiment_label:
            default_adapter_path = os.path.join(
                default_adapter_path, experiment_label
            )

        # Standard defaults optimized for 18GB RAM
        config = {
            "model": self.model_id,
            "train": True,
            "data": str(settings.structured_path),
            "adapter_path": default_adapter_path,
            "batch_size": 1,
            "iters": 100,
            "learning_rate": 1e-5,
            "steps_per_report": 10,
            "steps_per_eval": 200,
            "val_batches": 1,
            "save_every": 100,
            "max_seq_length": 384,
            "grad_checkpoint": True,
            "num_layers": 16,
            "lora_parameters": {
                "rank": 8,
                "scale": 16.0,
                "dropout": 0.0,
            },
        }

        # Handle custom overrides
        if custom_args:
            for key, value in custom_args.items():
                if key == "rank":
                    config["lora_parameters"]["rank"] = value
                    # Keep scale = 2.0 * rank for stability
                    # (equivalent to alpha in other frameworks)
                    config["lora_parameters"]["scale"] = float(2 * value)
                elif key == "scale":
                    config["lora_parameters"]["scale"] = float(value)
                elif key == "lora_layers":
                    config["num_layers"] = value
                elif key in config:
                    config[key] = value
                elif key == "adapter_path":
                    # If adapter_path is provided
                    # we still want to append the experiment_label
                    if experiment_label:
                        config["adapter_path"] = os.path.join(
                            value, experiment_label
                        )
                    else:
                        config["adapter_path"] = value

        return config

    def train(
        self,
        custom_args: Optional[dict[str, Any]] = None,
        experiment_label: str = "",
    ):
        """
        Orchestrates the training process using a SUBPROCESS and a YAML config.
        This ensures memory reclamation and correct parameter passing.
        """
        gc.collect()
        # Use non-deprecated API
        mx.clear_cache()

        # Build Config
        train_config = self._prepare_config(custom_args, experiment_label)

        adapter_path = train_config["adapter_path"]
        adapter_file = os.path.join(adapter_path, "adapters.safetensors")

        if os.path.exists(adapter_file):
            print(
                f"[-] Experiment '{experiment_label}' "
                f"already exists. Skipping."
            )
            return

        # Write temporary YAML config
        config_path = f"config_{experiment_label}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(train_config, f)

        # Build CLI command using the recommended entry point
        cli_command = [
            sys.executable,
            "-m",
            "mlx_lm",
            "lora",
            "--config",
            config_path,
        ]

        print(
            f"\n>>> [SUBPROCESS START] Fine-Tuning: "
            f"{self.model_id} [{experiment_label}]"
        )
        print(f"[Config saved to: {config_path}]")

        try:
            # Run the training in a separate process
            subprocess.run(cli_command, check=True)
            print(f"[✓] [SUBPROCESS END] Completed: {experiment_label}")

            # Save a copy of the config in the results directory
            # for Git tracking
            import shutil

            results_lora_dir = "results/lora"
            os.makedirs(results_lora_dir, exist_ok=True)
            dest_config = os.path.join(
                results_lora_dir, f"{experiment_label}_config.yaml"
            )
            shutil.copy2(config_path, dest_config)
            print(f"[✓] Configuration trace saved to: {dest_config}")

        except subprocess.CalledProcessError as e:
            print(
                f"[!] Subprocess training FAILED for "
                f"'{experiment_label}' (Exit code: {e.returncode})"
            )
        except Exception as e:
            print(f"[!] Error launching training subprocess: {e}")
        finally:
            # Cleanup temp config
            if os.path.exists(config_path):
                os.remove(config_path)
            mx.clear_cache()
            gc.collect()
