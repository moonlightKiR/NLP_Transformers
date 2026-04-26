from pathlib import Path

from app.constants import DATASET_DIR_NAME


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


# Global configuration instance
settings = AppConfig()
