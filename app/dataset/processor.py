import json

import torch

from app.config import settings


class DialogueProcessor:
    """Handles extraction and preparation of dialogue data."""

    def load_dialogues(self, file_path):
        """Loads dialogues from a JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            print(f"[!] Error loading {file_path.name}: {e}")
            return []

    def extract_utterances(self, dialogues):
        """Extracts a list of utterances from dialogues."""
        utterances = []
        if not dialogues:
            return utterances

        for dialogue in dialogues:
            turns = dialogue.get("turns", [])
            for turn in turns:
                utterance = turn.get("utterance", "").strip()
                if utterance:
                    utterances.append(utterance)
        return utterances


class TokenizerService:
    """Wrapper for model-specific tokenization logic
    with Apple Silicon (MPS) support."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        if self.device.type == "mps":
            print("[+] Apple Silicon GPU (MPS) detected and active.")

    def process(self, texts, max_length=512, padding=True, truncation=True):
        """Performs tokenization and moves tensors to the active device."""
        if not texts:
            return None

        encoded = self._tokenizer(
            texts,
            max_length=max_length,
            padding="max_length" if padding else False,
            truncation=truncation,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoded.items()}


class PreprocessingService:
    """High-level service to orchestrate the preprocessing pipeline
    and save results."""

    def __init__(self, processor, model_label, tokenizer_name):
        self._processor = processor
        self._model_label = model_label
        self._tokenizer_name = tokenizer_name
        self._tokenizer_service = None

    def _load_tokenizer(self):
        """Lazy loads the Hugging Face tokenizer."""
        from transformers import AutoTokenizer

        print(
            f"[+] Loading tokenizer for {self._model_label}: \
                {self._tokenizer_name}..."
        )
        raw_tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
        if raw_tokenizer.pad_token is None:
            raw_tokenizer.pad_token = raw_tokenizer.eos_token
        self._tokenizer_service = TokenizerService(raw_tokenizer)

    def process_all(self, directory_path, max_length=512):
        """Processes all JSON files and saves results, \
            skipping existing ones."""
        try:
            if not directory_path.exists():
                print(f"[!] Directory not found: {directory_path}")
                return

            # Create output directory for this specific model
            output_dir = settings.preprocessed_path / self._model_label
            output_dir.mkdir(parents=True, exist_ok=True)

            json_files = sorted(list(directory_path.glob("dialogues_*.json")))

            # Filter out files that already have a corresponding .pt file
            pending_files = [
                f
                for f in json_files
                if not (output_dir / f"{f.stem}.pt").exists()
            ]

            if not pending_files:
                print(
                    f"[-] All files for '{self._model_label}' \
                        are already preprocessed. Skipping."
                )
                return

            # Only load tokenizer if there are pending files
            if self._tokenizer_service is None:
                self._load_tokenizer()

            print(
                f"[+] Processing {len(pending_files)} new files for \
                    '{self._model_label}'."
            )

            total_utterances = 0
            for i, file_path in enumerate(pending_files, 1):
                try:
                    dialogues = self._processor.load_dialogues(file_path)
                    utterances = self._processor.extract_utterances(dialogues)

                    if not utterances:
                        continue

                    # Tokenize
                    tensors = self._tokenizer_service.process(
                        utterances, max_length=max_length
                    )

                    # Save results as a .pt file
                    output_file = output_dir / f"{file_path.stem}.pt"
                    torch.save(tensors, output_file)

                    total_utterances += len(utterances)

                    if i % 20 == 0 or i == len(pending_files):
                        print(
                            f"    - [{i}/{len(pending_files)}] \
                                Saved: {output_file.name}"
                        )

                except Exception as file_error:
                    print(
                        f"[!] Skip file {file_path.name} due to error: \
                            {file_error}"
                    )
                    continue

            print(f"[✓] Preprocessing for '{self._model_label}' completed.")

        except Exception as e:
            print(f"[!] Bulk preprocessing critical failure: {e}")
