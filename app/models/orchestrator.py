from app.models.config import model_settings
from app.models.constants import LLAMA_GGUF_NAME, QWEN_GGUF_NAME
from app.models.inference import InferenceService
from app.models.inference_cpp import InferenceCPPService


class InitialInferenceService:
    """
    Service responsible for orchestrating initial inference tests.
    Adheres to SOLID by encapsulating the comparative inference logic.
    """

    def __init__(self, dialogue_processor, test_split):
        self._processor = dialogue_processor
        self._split = test_split
        self._qwen_gguf = model_settings.gguf_dir / QWEN_GGUF_NAME
        self._llama_gguf = model_settings.gguf_dir / LLAMA_GGUF_NAME

    def run_comparative_inference(self):
        """Runs inference tests across different backends for documentation."""
        print("\n=== NLP Transformers: Initial Inference ===")

        # --- STEP A: Standard Transformers Library (Documenting Failure) ---
        print(
            "\n[STEP A] Attempting inference with\
            Standard 'transformers' library..."
        )

        # Qwen Attempt
        try:
            qwen_trans = InferenceService(
                self._qwen_gguf, model_settings.qwen_tok_path, "qwen"
            )
            qwen_trans.run_initial_test(self._processor, self._split)
        except Exception as e:
            print(f"[!] QWEN Transformers Backend failed as expected: {e}")

        # Llama Attempt
        try:
            llama_trans = InferenceService(
                self._llama_gguf, model_settings.llama_tok_path, "llama"
            )
            llama_trans.run_initial_test(self._processor, self._split)
        except Exception as e:
            print(f"[!] LLAMA Transformers Backend failed: {e}")

        # --- STEP B: Llama-cpp-python (Optimized Backend) ---
        print(
            "\n[STEP B] Attempting inference with 'llama-cpp-python'\
            (Strict Templates)..."
        )

        # Qwen Optimized
        qwen_cpp = InferenceCPPService(
            self._qwen_gguf, "qwen", str(model_settings.qwen_tok_path)
        )
        qwen_cpp.run_initial_test(self._processor, self._split)

        # Llama Optimized
        llama_cpp = InferenceCPPService(
            self._llama_gguf, "llama", str(model_settings.llama_tok_path)
        )
        llama_cpp.run_initial_test(self._processor, self._split)
