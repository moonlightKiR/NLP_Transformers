from pathlib import Path

from app.models.constants import PHI_TRAIN_ID, QWEN_TRAIN_ID

# Model IDs from central constants
QWEN_35_2B = QWEN_TRAIN_ID
PHI_4_MINI = PHI_TRAIN_ID

# Directories
ADAPTERS_ROOT = Path(".adapters")
QWEN_ADAPTER_DIR = ADAPTERS_ROOT / "qwen3_5_2b_lora"
PHI_ADAPTER_DIR = ADAPTERS_ROOT / "phi4_mini_lora"

# Default Hyperparameters
DEFAULT_ITERS = 100
DEFAULT_BATCH_SIZE = 1
DEFAULT_LEARNING_RATE = 1e-5
QWEN_LEARNING_RATE = 2e-5
