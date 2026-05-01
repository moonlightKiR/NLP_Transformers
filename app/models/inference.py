import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings
from app.converters.templates import CHAT_TEMPLATES
from app.models.constants import LLAMA_TRAIN_ID, QWEN_TRAIN_ID
from app.utils.hardware import HardwareDetector


class InferenceService:
    """Service to handle model loading and text generation
    using Transformers."""

    def __init__(self, model_path, tokenizer_path, model_label):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model_label = model_label
        self.model = None
        self.tokenizer = None

        # Detect device using HardwareDetector
        self.device_type = HardwareDetector.get_device_type()
        if self.device_type == "cuda":
            self.device = torch.device("cuda")
        elif self.device_type == "mps" or self.device_type == "mlx":
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def load_resources(self):
        """Loads the tokenizer and the model into memory."""
        print(
            f"[+] Loading tokenizer for {self.model_label} \
            from: {self.tokenizer_path.name}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        # Ensure tokenizer has pad_token
        if self.tokenizer.pad_token is None:
            if getattr(self.tokenizer, "eos_token", None):
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.tokenizer.pad_token = "[PAD]"

        # Ensure a chat_template exists so apply_chat_template works
        if not getattr(self.tokenizer, "chat_template", None):
            self.tokenizer.chat_template = CHAT_TEMPLATES.get(
                self.model_label, CHAT_TEMPLATES["qwen"]
            )

        # NATIVE CUDA OPTIMIZATION: Use safetensors + BitsAndBytes
        if self.device_type == "cuda":
            repo_id = (
                QWEN_TRAIN_ID
                if "qwen" in self.model_label.lower()
                else LLAMA_TRAIN_ID
            )
            print(
                f"[+] Loading native model "
                f"{self.model_label} from {repo_id} on CUDA (4-bit)..."
            )

            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # GGUF Fallback for MPS/CPU
            print(
                f"[+] Loading GGUF model "
                f"{self.model_label} on {self.device}..."
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path.parent,
                gguf_file=self.model_path.name,
                device_map="auto",
                dtype=torch.float16,
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
