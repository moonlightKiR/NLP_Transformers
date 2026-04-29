from pathlib import Path

from app.models.constants import (
    PHI_TOKENIZER_DIR_NAME,
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
    def qwen_tok_path(self):
        """Returns the local path for the Qwen tokenizer."""
        return self.tokenizers_dir / QWEN_TOKENIZER_DIR_NAME

    @property
    def phi_tok_path(self):
        """Returns the local path for the Phi tokenizer."""
        return self.tokenizers_dir / PHI_TOKENIZER_DIR_NAME

    @property
    def gemma_tok_path(self):
        """DEPRECATED: Returns the local path for the Phi tokenizer
        for backward compatibility."""
        return self.phi_tok_path


# Global configuration instance for models
model_settings = ModelConfig()
