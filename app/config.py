from pathlib import Path

from app.constants import DATASET_DIR_NAME


class AppConfig:
    """Gestiona la configuración dinámica y rutas del proyecto."""

    @property
    def project_root(self) -> Path:
        """Retorna la ruta raíz del proyecto."""
        # .parent.parent.parent sube de app/ a la raíz
        return Path(__file__).resolve().parent.parent.parent

    @property
    def dataset_path(self) -> Path:
        """Retorna la ruta completa donde se alojará el dataset."""
        return self.project_root / DATASET_DIR_NAME


# Instancia global de configuración (Singleton pattern sugerido)
settings = AppConfig()
