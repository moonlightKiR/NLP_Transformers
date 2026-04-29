from pathlib import Path

from app.models.constants import PHI_TRAIN_ID, QWEN_TRAIN_ID

# Model IDs from central constants
QWEN_35_2B = QWEN_TRAIN_ID
PHI_4_MINI = PHI_TRAIN_ID

# Directories
ADAPTERS_ROOT = Path(".adapters")
LORA_DIR_NAME = "lora"
OPTUNA_DIR_NAME = "optuna"


def model_dir_name(model_id: str) -> str:
    """Maps a HF model id (org/name) to a stable folder name."""
    return model_id.split("/")[-1]


def adapters_model_root(model_id: str) -> Path:
    return ADAPTERS_ROOT / model_dir_name(model_id)


def adapters_lora_root(model_id: str) -> Path:
    return adapters_model_root(model_id) / LORA_DIR_NAME


def adapters_optuna_root(model_id: str) -> Path:
    return adapters_model_root(model_id) / OPTUNA_DIR_NAME


# Default Hyperparameters
DEFAULT_ITERS = 100
DEFAULT_BATCH_SIZE = 1
DEFAULT_LEARNING_RATE = 1e-5
QWEN_LEARNING_RATE = 2e-5
