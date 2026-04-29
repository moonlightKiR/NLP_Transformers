from dataclasses import dataclass, field
from typing import Any, Dict

from app.training.constants import (
    DEFAULT_ITERS,
    DEFAULT_LEARNING_RATE,
    LLAMA_32_1B,
    QWEN_35_2B,
    QWEN_LEARNING_RATE,
    adapters_lora_root,
)


@dataclass
class TrainingPhaseConfig:
    """Configuration for the full fine-tuning phase (2B/4B Models)."""

    qwen_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "model": QWEN_35_2B,
            "batch_size": 1,
            "learning_rate": QWEN_LEARNING_RATE,
            "iters": DEFAULT_ITERS,
            "adapter_path": str(adapters_lora_root(QWEN_35_2B)),
            "grad_checkpoint": True,
        }
    )

    llama_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "model": LLAMA_32_1B,
            "batch_size": 1,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "iters": DEFAULT_ITERS,
            "adapter_path": str(adapters_lora_root(LLAMA_32_1B)),
            "grad_checkpoint": True,
        }
    )


training_settings = TrainingPhaseConfig()
