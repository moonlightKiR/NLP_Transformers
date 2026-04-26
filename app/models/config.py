from pathlib import Path


class ModelConfig:
    """Manages model-specific paths and configurations."""

    @property
    def models_dir(self):
        """Returns the path for the models directory
        (.models/ at the project root)."""
        # .parent.parent.parent goes from app/models/config.py to the root
        return Path(__file__).resolve().parent.parent.parent / ".models"


# Global configuration instance for models
model_settings = ModelConfig()
