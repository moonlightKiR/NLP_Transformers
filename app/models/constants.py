# Hugging Face Model IDs for Training (Safetensors)
QWEN_TRAIN_ID = "Qwen/Qwen3.5-2B"
PHI_TRAIN_ID = "microsoft/phi-4-mini-instruct"

# GGUF Model URLs for Inference (Unsloth versions)
QWEN_MODEL_URL = "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf"
PHI_MODEL_URL = "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf"

# Local GGUF Filenames
QWEN_GGUF_NAME = "Qwen3.5-2B-Q4_K_M.gguf"
PHI_GGUF_NAME = "Phi-4-mini-instruct-Q4_K_M.gguf"

# Tokenizer File URLs
QWEN_TOKENIZER_FILES = [
    f"https://huggingface.co/{QWEN_TRAIN_ID}/resolve/main/tokenizer.json",
    f"https://huggingface.co/{QWEN_TRAIN_ID}/resolve/main/tokenizer_config.json",
]

PHI_TOKENIZER_FILES = [
    f"https://huggingface.co/{PHI_TRAIN_ID}/resolve/main/tokenizer.json",
    f"https://huggingface.co/{PHI_TRAIN_ID}/resolve/main/tokenizer_config.json",
]

# Tokenizer directory names (for local storage)
QWEN_TOKENIZER_DIR_NAME = "qwen"
PHI_TOKENIZER_DIR_NAME = "phi"
