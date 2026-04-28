from dataclasses import dataclass, field
from typing import Any, Dict

from app.training.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ITERS,
    DEFAULT_LEARNING_RATE,
    GEMMA_4_2B,
    GEMMA_ADAPTER_DIR,
    QWEN_35_2B,
    QWEN_ADAPTER_DIR,
    QWEN_LEARNING_RATE,
)


@dataclass
class TrainingPhaseConfig:
    """Configuration for the full fine-tuning phase."""

    qwen_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "model": QWEN_35_2B,
            "batch_size": DEFAULT_BATCH_SIZE,
            "learning_rate": QWEN_LEARNING_RATE,
            "iters": DEFAULT_ITERS,
            "adapter_path": str(QWEN_ADAPTER_DIR),
        }
    )

    gemma_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "model": GEMMA_4_2B,
            "batch_size": DEFAULT_BATCH_SIZE,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "iters": DEFAULT_ITERS,
            "adapter_path": str(GEMMA_ADAPTER_DIR),
        }
    )


training_settings = TrainingPhaseConfig()
