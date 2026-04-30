import json
import re
import shutil

from app.config import settings
from app.constants import TEST_SPLIT_STRUCTURED_DIR, TRAIN_SPLIT_STRUCTURED_DIR


class MLXDataConverter:
    """Converts structured chat JSONs to MLX-compatible JSONL format."""

    # Keep a margin under the 320-token training limit to account for
    # chat-template/system overhead added by the model tokenizer.
    MAX_TOKENS_PER_EXAMPLE = 280
    TOKENS_PER_MESSAGE_OVERHEAD = 6

    def _count_nonempty_lines(self, file_path) -> int:
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def _estimate_tokens(self, text: str) -> int:
        """
        Cheap tokenizer-agnostic estimate.
        We count words and punctuation separately so the budget stays
        conservative across different model tokenizers.
        """
        return len(re.findall(r"\w+|[^\w\s]", text))

    def _message_tokens(self, message: dict) -> int:
        role = message.get("role", "")
        content = message.get("content", "")
        return (
            self._estimate_tokens(role)
            + self._estimate_tokens(content)
            + self.TOKENS_PER_MESSAGE_OVERHEAD
        )

    def _split_conversation(self, messages: list[dict]) -> list[list[dict]]:
        """
        Split long conversations into smaller contiguous chunks so training
        does not repeatedly truncate them at runtime.
        """
        chunks: list[list[dict]] = []
        current_chunk: list[dict] = []
        current_tokens = 0

        for message in messages:
            message_tokens = self._message_tokens(message)

            if (
                current_chunk
                and current_tokens + message_tokens
                > self.MAX_TOKENS_PER_EXAMPLE
            ):
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0

            # If a single message is still oversized, keep it alone rather than
            # dropping data. MLX may still truncate that rare case.
            current_chunk.append(message)
            current_tokens += message_tokens

        if current_chunk:
            chunks.append(current_chunk)

        # Avoid one-message trailing chunks when possible by merging them back.
        if len(chunks) >= 2 and len(chunks[-1]) == 1:
            chunks[-2].extend(chunks[-1])
            chunks.pop()

        return chunks

    def convert_split(self, split_name):
        """Merges individual structured JSONs into
        a single train/test.jsonl file with long conversations pre-split."""
        input_dir = settings.structured_path / split_name
        output_file = settings.structured_path / f"{split_name}.jsonl"

        if not input_dir.exists():
            print(f"[!] Input directory not found: {input_dir}")
            return

        json_files = list(input_dir.glob("dialogues_*.json"))
        print(
            f"[+] Converting {len(json_files)} files to MLX format "
            f"with pre-splitting ({split_name}.jsonl)..."
        )

        conversations_written = 0
        chunked_examples_written = 0
        with open(output_file, "w", encoding="utf-8") as outfile:
            for file_path in json_files:
                with open(file_path, "r", encoding="utf-8") as infile:
                    conversations = json.load(infile)
                    for conv in conversations:
                        conversations_written += 1
                        for chunk in self._split_conversation(conv):
                            # MLX expects a dict with a "messages" key
                            outfile.write(
                                json.dumps({"messages": chunk}) + "\n"
                            )
                            chunked_examples_written += 1

        print(
            f"[✓] Created {output_file.name} "
            f"({conversations_written} conversations -> "
            f"{chunked_examples_written} training examples)"
        )
        return output_file

    def ensure_train_valid(
        self, valid_fraction: float = 0.2
    ) -> tuple[str, str]:
        """
        Ensures MLX has non-empty train/valid JSONL files under
        `settings.structured_path`.

        MLX LoRA expects `train.jsonl` and `valid.jsonl` by default.
        We map:
        - train.jsonl  <- structured/train/
        - valid.jsonl  <- first `valid_fraction` of structured/test/
          (fallback: sample from train)
        """
        train_path = self.convert_split(TRAIN_SPLIT_STRUCTURED_DIR)
        valid_path = self.convert_split(TEST_SPLIT_STRUCTURED_DIR)

        # If test split doesn't exist or results in an empty file, create
        # a small validation set from the training JSONL to avoid running
        # without validation.
        valid_file = settings.structured_path / "valid.jsonl"
        train_file = settings.structured_path / "train.jsonl"

        if train_path is not None and train_path.exists():
            # Normalize expected filenames for mlx-lm
            if train_path != train_file:
                shutil.copyfile(train_path, train_file)

        if (
            valid_path is not None
            and valid_path.exists()
            and valid_path.stat().st_size > 0
        ):
            # Use a portion of test.jsonl as validation. Keep test.jsonl intact
            # for final evaluation.
            total = self._count_nonempty_lines(valid_path)
            target = max(1, int(total * valid_fraction)) if total else 0
            wrote = 0
            with (
                open(valid_path, "r", encoding="utf-8") as src,
                open(valid_file, "w", encoding="utf-8") as dst,
            ):
                for line in src:
                    if not line.strip():
                        continue
                    dst.write(line)
                    wrote += 1
                    if wrote >= target:
                        break

            if wrote == 0:
                print(
                    "[!] Warning: test.jsonl had no usable lines; "
                    "validation will remain empty."
                )
            else:
                pct = int(valid_fraction * 100)
                print(
                    f"[✓] Created validation from test: valid.jsonl "
                    f"({wrote} lines, ~{pct}%)"
                )
            return (str(train_file), str(valid_file))

        # Fallback: sample first N lines from train.jsonl
        if not train_file.exists() or train_file.stat().st_size == 0:
            print(
                "[!] Cannot create validation set: "
                "train.jsonl is missing or empty."
            )
            return (str(train_file), str(valid_file))

        max_lines = 200
        wrote = 0
        with (
            open(train_file, "r", encoding="utf-8") as src,
            open(valid_file, "w", encoding="utf-8") as dst,
        ):
            for line in src:
                if not line.strip():
                    continue
                dst.write(line)
                wrote += 1
                if wrote >= max_lines:
                    break

        if wrote == 0:
            print(
                "[!] Warning: train.jsonl had no usable lines; "
                "validation will remain empty."
            )
        else:
            print(
                f"[✓] Created validation fallback: valid.jsonl ({wrote} lines)"
            )

        return (str(train_file), str(valid_file))
