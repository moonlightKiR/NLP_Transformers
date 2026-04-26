from app.config import settings
from app.constants import SGD_REPO_URL
from app.dataset.downloader import (
    DatasetService,
    GitDownloader,
    IntegrityVerifier,
)


def main() -> None:
    """Main entry point for the application."""

    # Dependency injection and configuration (SOLID)
    downloader = GitDownloader()
    verifier = IntegrityVerifier()

    # The service now receives both the downloader and the integrity verifier
    service = DatasetService(downloader=downloader, verifier=verifier)

    # Execution using centralized configuration (Clean Code)
    print("=== NLP Transformers: Dataset Setup ===")
    service.setup_dataset(
        url=SGD_REPO_URL,
        target_path=settings.dataset_path,
        manifest_path=settings.manifest_path,
    )


if __name__ == "__main__":
    main()
