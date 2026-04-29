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
    PHI_MODEL_URL,
    PHI_TOKENIZER_FILES,
    QWEN_MODEL_URL,
    QWEN_TOKENIZER_FILES,
)
from app.models.downloader import ModelDownloader, ModelService
from app.models.orchestrator import InitialInferenceService


def main():
    """Main entry point for the application orchestrating the full pipeline."""

    # 1. Resource Ingestion
    print("=== NLP Transformers: Resource Ingestion ===")
    dataset_service = DatasetService(
        GitDownloader(), verifier=IntegrityVerifier()
    )
    dataset_service.setup_dataset(
        url=SGD_REPO_URL,
        target_path=settings.dataset_path,
        manifest_path=settings.manifest_path,
    )

    model_service = ModelService(ModelDownloader())
    model_service.setup_model(
        url=PHI_MODEL_URL, target_dir=model_settings.gguf_dir
    )
    model_service.setup_model(
        url=QWEN_MODEL_URL, target_dir=model_settings.gguf_dir
    )
    model_service.setup_tokenizer_from_urls(
        QWEN_TOKENIZER_FILES, model_settings.qwen_tok_path
    )
    model_service.setup_tokenizer_from_urls(
        PHI_TOKENIZER_FILES, model_settings.phi_tok_path
    )

    # 2. Data Structuring
    print("\n=== NLP Transformers: Data Structuring ===")
    dialogue_processor = DialogueProcessor()
    structuring_service = StructuringService(dialogue_processor)
    structuring_service.structure_directory(settings.train_raw_path)
    structuring_service.structure_directory(settings.test_raw_path)

    # 3. Preprocessing
    print("\n=== NLP Transformers: Preprocessing & Tokenization ===")
    qwen_preprocessor = PreprocessingService(
        dialogue_processor,
        model_label="qwen",
        tokenizer_name=str(model_settings.qwen_tok_path),
    )
    qwen_preprocessor.process_structured_directory(TRAIN_SPLIT_STRUCTURED_DIR)
    qwen_preprocessor.process_structured_directory(TEST_SPLIT_STRUCTURED_DIR)

    phi_preprocessor = PreprocessingService(
        dialogue_processor,
        model_label="phi",
        tokenizer_name=str(model_settings.phi_tok_path),
    )
    phi_preprocessor.process_structured_directory(TRAIN_SPLIT_STRUCTURED_DIR)
    phi_preprocessor.process_structured_directory(TEST_SPLIT_STRUCTURED_DIR)

    # 4. Initial Inference (Comparative Analysis)
    inference_orchestrator = InitialInferenceService(
        dialogue_processor, TEST_SPLIT_STRUCTURED_DIR
    )
    inference_orchestrator.run_comparative_inference()


if __name__ == "__main__":
    main()
