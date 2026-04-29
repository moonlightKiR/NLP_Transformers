import subprocess


class ModelDownloader:
    """Handles the physical download of model files and tokenizers."""

    def download(self, source, destination):
        if destination.exists():
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[+] Downloading from {source}...")
        try:
            import os

            # Use curl -L to follow redirects (essential for Hugging Face)
            cmd = ["curl", "-L", source, "-o", str(destination)]

            # Add Authorization header
            # if it's a Hugging Face URL and token is available
            token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get(
                "HF_TOKEN"
            )
            if token and "huggingface.co" in source:
                cmd.extend(["-H", f"Authorization: Bearer {token}"])

            subprocess.run(cmd, check=True)
            print(f"[+] Download completed: {destination.name}")
        except subprocess.CalledProcessError:
            raise RuntimeError(f"Failed to download: {source}")


class ModelService:
    """Service to manage model setup and organization."""

    def __init__(self, downloader):
        self._downloader = downloader

    def setup_model(self, url, target_dir):
        """Orchestrates the download of a model to the target directory."""
        filename = url.split("/")[-1]
        target_path = target_dir / filename
        try:
            if not target_path.exists():
                self._downloader.download(url, target_path)
            else:
                print(f"[-] Model {filename} already exists locally.")
        except Exception as e:
            print(f"[!] Error in model service for {filename}: {e}")

    def setup_tokenizer_from_urls(self, urls, target_dir):
        """Downloads specific tokenizer files from a list of URLs."""
        if target_dir.exists() and any(target_dir.iterdir()):
            print(
                f"[-] Tokenizer in {target_dir.name} already exists locally."
            )
            return

        print(f"[+] Setting up tokenizer in {target_dir}...")
        for url in urls:
            filename = url.split("/")[-1]
            target_path = target_dir / filename
            try:
                self._downloader.download(url, target_path)
            except Exception as e:
                print(f"[!] Error downloading tokenizer file {filename}: {e}")

    def ensure_hf_snapshot(self, repo_id, target_dir, allow_patterns=None):
        """
        Ensures a local snapshot of a Hugging Face repository exists.
        Useful for models that need Safetensors (like MLX training).
        """
        if target_dir.exists() and any(target_dir.iterdir()):
            return str(target_dir)

        print(f"[+] Ensuring HF snapshot for {repo_id} in {target_dir}...")
        try:
            import os

            from huggingface_hub import snapshot_download

            token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get(
                "HF_TOKEN"
            )

            local_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                allow_patterns=allow_patterns
                or ["*.safetensors", "*.json", "*.txt"],
                token=token,
            )
            print(f"[✓] Snapshot for {repo_id} ready at {local_path}")
            return local_path
        except Exception as e:
            print(f"[!] Error downloading HF snapshot for {repo_id}: {e}")
            return None
