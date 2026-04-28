from pathlib import Path

# Model IDs - Corrected based on user links
# Note: Using the exact IDs from the provided Hugging Face URLs
QWEN_35_2B = "Qwen/Qwen3.5-2B"
GEMMA_4_2B = "google/gemma-4-E2B"

# Directories
ADAPTERS_ROOT = Path(".adapters")
QWEN_ADAPTER_DIR = ADAPTERS_ROOT / "qwen3_5_2b_lora"
GEMMA_ADAPTER_DIR = ADAPTERS_ROOT / "gemma4_2b_lora"

# Default Hyperparameters
DEFAULT_ITERS = 100
DEFAULT_BATCH_SIZE = 2  # 2B models can handle slightly larger batches than 9B
DEFAULT_LEARNING_RATE = 1e-5
QWEN_LEARNING_RATE = 2e-5
