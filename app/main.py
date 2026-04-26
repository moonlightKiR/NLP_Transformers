from app.config import settings
from app.constants import SGD_REPO_URL
from app.dataset.downloader import (
    DatasetService,
    GitDownloader,
    IntegrityVerifier,
)
from app.dataset.processor import DialogueProcessor, PreprocessingService
from app.models.config import model_settings
from app.models.constants import (
    GEMMA_MODEL_URL,
    GEMMA_TOKENIZER_FILES,
    QWEN_MODEL_URL,
    QWEN_TOKENIZER_FILES,
)
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
    model_service.setup_model(
        url=GEMMA_MODEL_URL, target_dir=model_settings.gguf_dir
    )
    model_service.setup_model(
        url=QWEN_MODEL_URL, target_dir=model_settings.gguf_dir
    )

    model_service.setup_tokenizer_from_urls(
        QWEN_TOKENIZER_FILES, model_settings.qwen_tok_path
    )
    model_service.setup_tokenizer_from_urls(
        GEMMA_TOKENIZER_FILES, model_settings.gemma_tok_path
    )

    # 3. Preprocessing (Tokenization, Truncation, Padding)
    train_dir = settings.dataset_path / "train"
    dialogue_processor = DialogueProcessor()

    # --- Qwen Preprocessing ---
    print("\n=== NLP Transformers: Qwen Bulk Preprocessing ===")
    qwen_service = PreprocessingService(
        dialogue_processor,
        model_label="qwen",
        tokenizer_name=str(model_settings.qwen_tok_path),
    )
    qwen_service.process_all(train_dir)

    # --- Gemma Preprocessing ---
    print("\n=== NLP Transformers: Gemma Bulk Preprocessing ===")
    gemma_service = PreprocessingService(
        dialogue_processor,
        model_label="gemma",
        tokenizer_name=str(model_settings.gemma_tok_path),
    )
    gemma_service.process_all(train_dir)


if __name__ == "__main__":
    main()
