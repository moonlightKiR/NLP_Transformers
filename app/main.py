from app.config import settings
from app.constants import (
    SGD_REPO_URL,
    TEST_SPLIT_STRUCTURED_DIR,
    TRAIN_SPLIT_STRUCTURED_DIR,
)
from app.dataset.downloader import (
    DatasetService,
    GitDownloader,
    IntegrityVerifier,
)
from app.dataset.processor import (
    DialogueProcessor,
    PreprocessingService,
    StructuringService,
)
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

    # 1. Dataset & Models Ingestion
    git_downloader = GitDownloader()
    verifier = IntegrityVerifier()
    dataset_service = DatasetService(git_downloader, verifier=verifier)

    print("=== NLP Transformers: Resource Ingestion ===")
    dataset_service.setup_dataset(
        url=SGD_REPO_URL,
        target_path=settings.dataset_path,
        manifest_path=settings.manifest_path,
    )

    model_downloader = ModelDownloader()
    model_service = ModelService(model_downloader)
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

    # 2. Structuring (Raw SGD -> JSON Chat Format)
    print("\n=== NLP Transformers: Data Structuring ===")
    dialogue_processor = DialogueProcessor()
    structuring_service = StructuringService(dialogue_processor)

    structuring_service.structure_directory(settings.train_raw_path)
    structuring_service.structure_directory(settings.test_raw_path)

    # 3. Preprocessing (JSON Chat -> Model Tensors)
    # --- Qwen ---
    print("\n=== NLP Transformers: Qwen Tokenization ===")
    qwen_service = PreprocessingService(
        dialogue_processor,
        model_label="qwen",
        tokenizer_name=str(model_settings.qwen_tok_path),
    )
    qwen_service.process_structured_directory(TRAIN_SPLIT_STRUCTURED_DIR)
    qwen_service.process_structured_directory(TEST_SPLIT_STRUCTURED_DIR)

    # --- Gemma ---
    print("\n=== NLP Transformers: Gemma Tokenization ===")
    gemma_service = PreprocessingService(
        dialogue_processor,
        model_label="gemma",
        tokenizer_name=str(model_settings.gemma_tok_path),
    )
    gemma_service.process_structured_directory(TRAIN_SPLIT_STRUCTURED_DIR)
    gemma_service.process_structured_directory(TEST_SPLIT_STRUCTURED_DIR)


if __name__ == "__main__":
    main()
