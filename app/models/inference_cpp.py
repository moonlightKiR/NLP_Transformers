import time

from llama_cpp import Llama

from app.config import settings


class InferenceCPPService:
    """Service to handle GGUF inference using llama-cpp-python
    with performance metrics."""

    def __init__(self, model_path, model_label):
        self.model_path = model_path
        self.model_label = model_label
        self.model = None

    def load_resources(self):
        """Loads the GGUF model into memory using Llama-cpp."""
        print(
            f"[+] Loading GGUF model {self.model_label} \
                with llama-cpp-python (Backend: Metal/MPS)..."
        )
        self.model = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=-1,  # Offload all layers to GPU (Metal)
            n_ctx=2048,
            verbose=False,
        )
        print(f"[✓] {self.model_label} loaded successfully on Mac GPU.")

    def generate_response(self, messages, max_new_tokens=128, temperature=0.7):
        """Generates a response and returns performance metrics."""
        if not self.model:
            self.load_resources()

        print(
            f"[+] Generating response with {self.model_label} (llama-cpp)..."
        )

        start_time = time.time()
        response_data = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        end_time = time.time()

        # Extract data
        content = response_data["choices"][0]["message"]["content"].strip()
        tokens_generated = response_data["usage"]["completion_tokens"]
        duration = end_time - start_time

        # Calculate tokens per second
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
                    Inference Test (LLAMA-CPP) ---"
            )
            print("[Context]")
            for msg in sample_context:
                print(f"  {msg['role'].upper()}: {msg['content']}")

            response, metrics = self.generate_response(sample_context)

            print("\n[Generated Response (llama-cpp)]")
            print(f"  ASSISTANT: {response}")

            print("\n[Performance Metrics]")
            print(f"  - Tokens generated: {metrics['tokens']}")
            print(f"  - Time taken: {metrics['duration_s']}s")
            print(f"  - Speed: {metrics['tokens_per_second']} tokens/sec")

        except Exception as e:
            print(f"[!] Llama-cpp Inference failed: {e}")
