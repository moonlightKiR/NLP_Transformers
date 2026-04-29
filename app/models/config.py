from pathlib import Path

from app.models.constants import (
    LLAMA_TOKENIZER_DIR_NAME,
    QWEN_TOKENIZER_DIR_NAME,
)


class ModelConfig:
    """Manages model-specific paths and configurations."""

    @property
    def base_dir(self):
        """Returns the base hidden models directory at the project root."""
        return Path(__file__).resolve().parent.parent.parent / ".models"

    @property
    def gguf_dir(self):
        """Returns the directory for GGUF model files."""
        return self.base_dir / "gguf"

    @property
    def tokenizers_dir(self):
        """Returns the directory for local tokenizer files."""
        return self.base_dir / "tokenizers"

    @property
    def hf_models_dir(self):
        """Returns the directory for full HF model snapshots (Safetensors)."""
        return self.base_dir / "hf_models"

    @property
    def qwen_tok_path(self):
        """Returns the local path for the Qwen tokenizer."""
        return self.tokenizers_dir / QWEN_TOKENIZER_DIR_NAME

    @property
    def llama_tok_path(self):
        """Returns the local path for the Llama tokenizer."""
        return self.tokenizers_dir / LLAMA_TOKENIZER_DIR_NAME


# Global configuration instance for models
model_settings = ModelConfig()
