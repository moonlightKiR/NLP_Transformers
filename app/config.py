import os
from pathlib import Path

from dotenv import load_dotenv

from app.constants import (
    DATASET_DIR_NAME,
    TEST_SPLIT_STRUCTURED_DIR,
    TRAIN_SPLIT_STRUCTURED_DIR,
)

# Load environment variables from .env file
load_dotenv()


class AppConfig:
    """Manages dynamic project configuration and paths."""

    @property
    def project_root(self) -> Path:
        """Returns the project root directory."""
        return Path(__file__).resolve().parent.parent

    @property
    def dataset_path(self) -> Path:
        """Returns the path inside a '.data' folder in the root directory."""
        return self.project_root / ".data" / DATASET_DIR_NAME

    @property
    def manifest_path(self) -> Path:
        """Returns the path for the integrity manifest
        file in the .data/ directory."""
        return self.project_root / ".data" / "manifest.json"

    @property
    def structured_path(self) -> Path:
        """Returns the directory
        where structured JSON dialogues will be stored."""
        return self.project_root / ".data" / "structured"

    @property
    def preprocessed_path(self) -> Path:
        """Returns the directory where tokenized tensors will be stored."""
        return self.project_root / ".data" / "preprocessed"

    @property
    def train_raw_path(self) -> Path:
        """Returns the raw training data path."""
        return self.dataset_path / TRAIN_SPLIT_STRUCTURED_DIR

    @property
    def test_raw_path(self) -> Path:
        """Returns the raw test data path."""
        return self.dataset_path / TEST_SPLIT_STRUCTURED_DIR

    @property
    def hf_token(self) -> str:
        """Returns the Hugging Face token from environment variables."""
        return os.getenv("HF_TOKEN", "")


# Global configuration instance
settings = AppConfig()
