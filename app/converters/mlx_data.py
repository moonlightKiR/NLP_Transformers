import json

from app.config import settings


class MLXDataConverter:
    """Converts structured chat JSONs to MLX-compatible JSONL format."""

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
