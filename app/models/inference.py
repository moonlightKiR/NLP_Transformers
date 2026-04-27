import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings


class InferenceService:
    """Service to handle model loading and text generation
    using Transformers."""

    def __init__(self, model_path, tokenizer_path, model_label):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model_label = model_label
        self.model = None
        self.tokenizer = None
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

    def load_resources(self):
        """Loads the tokenizer and the GGUF model into memory."""
        print(
            f"[+] Loading tokenizer for {self.model_label} \
                from: {self.tokenizer_path.name}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[+] Loading GGUF model {self.model_label} on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path.parent,
            gguf_file=self.model_path.name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        print(f"[✓] {self.model_label} loaded successfully.")

    def generate_response(
        self, messages, max_new_tokens=128, temperature=0.7, top_p=0.9
    ):
        """Generates a response given a list of chat messages."""
        if not self.model or not self.tokenizer:
            self.load_resources()

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(input_text, return_tensors="pt").to(
            self.model.device
        )

        print(f"[+] Generating response with {self.model_label}...")
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    def run_initial_test(self, dialogue_processor, split_name):
        """Orchestrates an initial inference test
        using a sample from a structured split."""
        try:
            test_file = (
                settings.structured_path / split_name / "dialogues_001.json"
            )
            if not test_file.exists():
                print(f"[!] Structured file not found: {test_file}")
                return

            dialogues = dialogue_processor.load_json(test_file)
            sample_context = dialogues[0][:3]

            print(
                f"\n--- {self.model_label.upper()} Initial Inference Test ---"
            )
            print("[Context]")
            for msg in sample_context:
                print(f"  {msg['role'].upper()}: {msg['content']}")

            response = self.generate_response(sample_context)
            print("\n[Generated Response]")
            print(f"  ASSISTANT: {response}")

        except Exception as e:
            print(f"[!] Error during initial inference test: {e}")
