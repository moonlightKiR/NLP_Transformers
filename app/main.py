from app.config import settings
from app.constants import SGD_REPO_URL
from app.dataset.downloader import (
    DatasetService,
    GitDownloader,
    IntegrityVerifier,
)
from app.models.config import model_settings
from app.models.constants import GEMMA_MODEL_URL, QWEN_MODEL_URL
from app.models.downloader import ModelDownloader, ModelService


def main():
    """Main entry point for the application."""

    # 1. Dataset Setup
    git_downloader = GitDownloader()
    verifier = IntegrityVerifier()
    dataset_service = DatasetService(git_downloader, verifier=verifier)

    print("=== NLP Transformers: Dataset Setup ===")
    dataset_service.setup_dataset(
        url=SGD_REPO_URL,
        target_path=settings.dataset_path,
        manifest_path=settings.manifest_path,
    )

    # 2. Models Setup
    model_downloader = ModelDownloader()
    model_service = ModelService(model_downloader)

    print("\n=== NLP Transformers: Models Setup ===")

    # Setup Gemma
    model_service.setup_model(
        url=GEMMA_MODEL_URL, target_dir=model_settings.models_dir
    )

    # Setup Qwen
    model_service.setup_model(
        url=QWEN_MODEL_URL, target_dir=model_settings.models_dir
    )


if __name__ == "__main__":
    main()
