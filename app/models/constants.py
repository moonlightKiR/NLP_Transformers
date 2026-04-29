# Hugging Face Model IDs for Training (Safetensors)
QWEN_TRAIN_ID = "Qwen/Qwen3.5-2B"
LLAMA_TRAIN_ID = "meta-llama/Llama-3.2-1B-Instruct"

# GGUF Model URLs for Inference (Unsloth versions)
QWEN_MODEL_URL = "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf"
LLAMA_MODEL_URL = "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# Local GGUF Filenames
QWEN_GGUF_NAME = "Qwen3.5-2B-Q4_K_M.gguf"
LLAMA_GGUF_NAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# Tokenizer File URLs
QWEN_TOKENIZER_FILES = [
    f"https://huggingface.co/{QWEN_TRAIN_ID}/resolve/main/tokenizer.json",
    f"https://huggingface.co/{QWEN_TRAIN_ID}/resolve/main/tokenizer_config.json",
]

LLAMA_TOKENIZER_FILES = [
    f"https://huggingface.co/{LLAMA_TRAIN_ID}/resolve/main/tokenizer.json",
    f"https://huggingface.co/{LLAMA_TRAIN_ID}/resolve/main/tokenizer_config.json",
]

# Tokenizer directory names (for local storage)
QWEN_TOKENIZER_DIR_NAME = "qwen"
LLAMA_TOKENIZER_DIR_NAME = "llama"
