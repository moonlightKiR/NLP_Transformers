import subprocess


class ModelDownloader:
    """Handles the physical download of model files and tokenizers."""

    def download(self, source, destination):
        if destination.exists():
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[+] Downloading from {source}...")
        try:
            # Use curl -L to follow redirects (essential for Hugging Face)
            subprocess.run(
                ["curl", "-L", source, "-o", str(destination)], check=True
            )
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
