from dataclasses import dataclass, field
from typing import Any, Dict

from app.training.constants import (
    DEFAULT_ITERS,
    DEFAULT_LEARNING_RATE,
    PHI_4_MINI,
    PHI_ADAPTER_DIR,
    QWEN_35_2B,
    QWEN_ADAPTER_DIR,
    QWEN_LEARNING_RATE,
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
            "adapter_path": str(QWEN_ADAPTER_DIR),
            "grad_checkpoint": True,
        }
    )

    phi_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "model": PHI_4_MINI,
            "batch_size": 1,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "iters": DEFAULT_ITERS,
            "adapter_path": str(PHI_ADAPTER_DIR),
            "grad_checkpoint": True,
        }
    )


training_settings = TrainingPhaseConfig()
