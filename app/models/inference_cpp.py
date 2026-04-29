import time

from llama_cpp import Llama

from app.config import settings
from app.converters.dialogue_format import ChatTemplateService


class InferenceCPPService:
    """
    Service to handle GGUF inference using llama-cpp-python
    with strict chat templates.
    """

    def __init__(self, model_path, model_label, tokenizer_path):
        self.model_path = model_path
        self.model_label = model_label
        self.tokenizer_path = tokenizer_path
        self.model = None
        # Composition: Use ChatTemplateService for formatting
        self._template_service = ChatTemplateService(tokenizer_path)

    def load_resources(self):
        """Loads the GGUF model into memory using Llama-cpp."""
        print(
            f"[+] Loading GGUF model {self.model_label} \
            (Backend: Metal/MPS)..."
        )
        self.model = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=-1,
            n_ctx=2048,
            verbose=False,
        )
        print(f"[✓] {self.model_label} loaded successfully on Mac GPU.")

    def generate_response(self, messages, max_new_tokens=128, temperature=0.7):
        """Generates a response using explicit chat templates
        for consistency."""
        if not self.model:
            self.load_resources()

        # 1. Apply strict chat template (Clean Code: Consistency with training)
        formatted_prompt = self._template_service.format_messages(messages)

        print(
            f"[+] Generating response with {self.model_label} \
            (strict template)..."
        )

        start_time = time.time()
        # Use create_completion instead of chat_completion
        # because we already formatted the string
        response_data = self.model.create_completion(
            prompt=formatted_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "<|end|>", "</s>"],  # Safety stop tokens
        )
        end_time = time.time()

        # Extract content and metrics
        content = response_data["choices"][0]["text"].strip()
        tokens_generated = response_data["usage"]["completion_tokens"]
        duration = end_time - start_time
        tps = tokens_generated / duration if duration > 0 else 0

        metrics = {
            "tokens": tokens_generated,
            "duration_s": round(duration, 2),
            "tokens_per_second": round(tps, 2),
        }

        return content, metrics

    def run_initial_test(self, dialogue_processor, split_name):
        """Orchestrates an initial inference test with metrics reporting."""
        try:
            test_file = (
                settings.structured_path / split_name / "dialogues_001.json"
            )
            if not test_file.exists():
                return

            dialogues = dialogue_processor.load_json(test_file)
            sample_context = dialogues[0][:3]

            print(
                f"\n--- {self.model_label.upper()} \
                Inference Test (Strict Templates) ---"
            )
            print("[Context]")
            for msg in sample_context:
                print(f"  {msg['role'].upper()}: {msg['content']}")

            response, metrics = self.generate_response(sample_context)

            print("\n[Generated Response]")
            print(f"  ASSISTANT: {response}")

            print("\n[Performance Metrics]")
            print(f"  - Tokens generated: {metrics['tokens']}")
            print(f"  - Time taken: {metrics['duration_s']}s")
            print(f"  - Speed: {metrics['tokens_per_second']} tokens/sec")

        except Exception as e:
            print(f"[!] Llama-cpp Inference failed: {e}")
