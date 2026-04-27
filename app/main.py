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
from app.models.inference import InferenceService
from app.models.inference_cpp import InferenceCPPService


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

    gemma_preprocessor = PreprocessingService(
        dialogue_processor,
        model_label="gemma",
        tokenizer_name=str(model_settings.gemma_tok_path),
    )
    gemma_preprocessor.process_structured_directory(TRAIN_SPLIT_STRUCTURED_DIR)
    gemma_preprocessor.process_structured_directory(TEST_SPLIT_STRUCTURED_DIR)

    # 4. Initial Inference (Demonstrating failures and successes)
    print("\n=== NLP Transformers: Initial Inference ===")

    qwen_gguf = model_settings.gguf_dir / "Qwen3.5-9B-Q3_K_M.gguf"
    gemma_gguf = model_settings.gguf_dir / "gemma-4-E2B-it-Q3_K_M.gguf"

    # ATTEMPT 1: Standard Transformers Library
    # (Known to fail for these new models)
    print(
        "\n[STEP A] Attempting inference \
            with Standard 'transformers' library..."
    )
    qwen_trans = InferenceService(
        qwen_gguf, model_settings.qwen_tok_path, "qwen"
    )
    qwen_trans.run_initial_test(dialogue_processor, TEST_SPLIT_STRUCTURED_DIR)

    gemma_trans = InferenceService(
        gemma_gguf, model_settings.gemma_tok_path, "gemma"
    )
    gemma_trans.run_initial_test(dialogue_processor, TEST_SPLIT_STRUCTURED_DIR)

    # ATTEMPT 2: Llama-cpp-python
    # (Fallback optimized for GGUF and Mac)
    print(
        "\n[STEP B] Attempting inference with 'llama-cpp-python' \
            (Optimized Backend)..."
    )
    qwen_cpp = InferenceCPPService(qwen_gguf, "qwen")
    qwen_cpp.run_initial_test(dialogue_processor, TEST_SPLIT_STRUCTURED_DIR)

    gemma_cpp = InferenceCPPService(gemma_gguf, "gemma")
    gemma_cpp.run_initial_test(dialogue_processor, TEST_SPLIT_STRUCTURED_DIR)


if __name__ == "__main__":
    main()
