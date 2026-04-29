import json
import shutil

from app.config import settings
from app.constants import TEST_SPLIT_STRUCTURED_DIR, TRAIN_SPLIT_STRUCTURED_DIR


class MLXDataConverter:
    """Converts structured chat JSONs to MLX-compatible JSONL format."""

    def _count_nonempty_lines(self, file_path) -> int:
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def convert_split(self, split_name):
        """Merges individual structured JSONs into
        a single train/test.jsonl file. Skips if already exists."""
        input_dir = settings.structured_path / split_name
        output_file = settings.structured_path / f"{split_name}.jsonl"

        if output_file.exists():
            print(
                f"[-] {output_file.name} already exists. Skipping conversion."
            )
            return output_file

        if not input_dir.exists():
            print(f"[!] Input directory not found: {input_dir}")
            return

        json_files = list(input_dir.glob("dialogues_*.json"))
        print(
            f"[+] Converting {len(json_files)} files to MLX format \
            ({split_name}.jsonl)..."
        )

        with open(output_file, "w", encoding="utf-8") as outfile:
            for file_path in json_files:
                with open(file_path, "r", encoding="utf-8") as infile:
                    conversations = json.load(infile)
                    for conv in conversations:
                        # MLX expects a dict with a "messages" key
                        outfile.write(json.dumps({"messages": conv}) + "\n")

        print(f"[✓] Created {output_file.name}")
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
