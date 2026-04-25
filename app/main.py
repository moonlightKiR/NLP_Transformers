from app.config import settings
from app.constants import SGD_REPO_URL
from app.dataset.downloader import DatasetService, GitDownloader


def main() -> None:
    """Punto de entrada principal del sistema."""

    # Inyección de dependencias y configuración (SOLID)
    downloader = GitDownloader()
    service = DatasetService(downloader=downloader)

    # Ejecución utilizando la configuración centralizada (Clean Code)
    print("=== NLP Transformers: Dataset Setup ===")
    service.setup_dataset(url=SGD_REPO_URL, target_path=settings.dataset_path)


if __name__ == "__main__":
    main()
