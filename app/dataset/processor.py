import json

import torch

from app.config import settings
from app.converters.dialogue_format import format_sgd_to_messages


class DialogueProcessor:
    """Handles loading and delegation of dialogue data."""

    def load_json(self, file_path):
        """Loads any JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Error loading {file_path.name}: {e}")
            return []

    def save_json(self, data, file_path):
        """Saves data to a JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


class StructuringService:
    """Service to convert raw SGD data to structured chat JSONs."""

    def __init__(self, processor):
        self._processor = processor

    def structure_directory(self, input_dir, max_turns=20):
        """Converts raw SGD files to structured chat format
        in .data/structured/."""
        split_name = input_dir.name
        output_dir = settings.structured_path / split_name
        output_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(list(input_dir.glob("dialogues_*.json")))
        print(
            f"[+] Structuring {len(json_files)} files: {split_name} \
            -> structured/{split_name}/"
        )

        for file_path in json_files:
            output_file = output_dir / file_path.name
            if output_file.exists():
                continue

            raw_data = self._processor.load_json(file_path)
            structured_data = format_sgd_to_messages(
                raw_data, max_turns=max_turns
            )
            self._processor.save_json(structured_data, output_file)

        print(f"[✓] Structuring for '{split_name}' completed.")


class TokenizerService:
    """Wrapper for model-specific tokenization logic with
    Chat Template support."""

    # Official templates fallback if not found in local files
    TEMPLATES = {
        "gemma": (
            "{{ bos_token }}{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<start_of_turn>user\\n' + "
            "message['content'] + '<end_of_turn>\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<start_of_turn>model\\n' + "
            "message['content'] + '<end_of_turn>\\n' }}"
            "{% endif %}{% endfor %}"
        ),
        "qwen": (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + "
            "message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
        ),
    }

    def __init__(self, tokenizer, model_label):
        self._tokenizer = tokenizer
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

        # Inject template if missing
        # (common when loading from local JSON files)
        if (
            not hasattr(self._tokenizer, "chat_template")
            or self._tokenizer.chat_template is None
        ):
            print(f"[!] Injecting missing chat template for {model_label}...")
            self._tokenizer.chat_template = self.TEMPLATES.get(
                model_label, self.TEMPLATES["qwen"]
            )

        if self.device.type == "mps":
            print(f"[+] Apple Silicon GPU (MPS) active for {model_label}.")

    def process_conversations(
        self, conversations, max_length=512, padding=True, truncation=True
    ):
        """Applies chat templates and tokenizes conversations."""
        formatted_texts = [
            self._tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            for conv in conversations
        ]

        encoded = self._tokenizer(
            formatted_texts,
            max_length=max_length,
            padding="max_length" if padding else False,
            truncation=truncation,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoded.items()}


class PreprocessingService:
    """High-level service to tokenize structured JSONs into tensors."""

    def __init__(self, processor, model_label, tokenizer_name):
        self._processor = processor
        self._model_label = model_label
        self._tokenizer_name = tokenizer_name
        self._tokenizer_service = None

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        print(
            f"[+] Loading tokenizer for {self._model_label}:\
            {self._tokenizer_name}..."
        )
        raw_tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_name, trust_remote_code=True
        )
        if raw_tokenizer.pad_token is None:
            raw_tokenizer.pad_token = raw_tokenizer.eos_token
        self._tokenizer_service = TokenizerService(
            raw_tokenizer, self._model_label
        )

    def process_structured_directory(self, split_name, max_length=512):
        """Reads structured JSONs and saves tokenized tensors."""
        input_dir = settings.structured_path / split_name
        output_dir = (
            settings.preprocessed_path / self._model_label / split_name
        )

        if not input_dir.exists():
            print(f"[!] Structured data not found for split: {split_name}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        json_files = sorted(list(input_dir.glob("dialogues_*.json")))

        pending_files = [
            f for f in json_files if not (output_dir / f"{f.stem}.pt").exists()
        ]
        if not pending_files:
            print(
                f"[-] Tensors for '{self._model_label}/{split_name}' \
                already exist."
            )
            return

        if self._tokenizer_service is None:
            self._load_tokenizer()

        print(
            f"[+] Tokenizing {len(pending_files)} files \
            for '{self._model_label}'..."
        )
        for i, file_path in enumerate(pending_files, 1):
            conversations = self._processor.load_json(file_path)
            tensors = self._tokenizer_service.process_conversations(
                conversations, max_length=max_length
            )
            torch.save(tensors, output_dir / f"{file_path.stem}.pt")

            if i % 20 == 0 or i == len(pending_files):
                print(
                    f"    - [{i}/{len(pending_files)}] \
                    Processed: {file_path.name}"
                )

        print(
            f"[✓] Tokenization for '{self._model_label}/{split_name}' \
            completed."
        )
