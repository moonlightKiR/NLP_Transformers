import subprocess


class ModelDownloader:
    """Handles the physical download of model files."""

    def download(self, source, destination):
        if destination.exists():
            print(
                f"[-] File already exists at: {destination}. \
                Skipping download."
            )
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[+] Downloading model from {source}...")
        try:
            subprocess.run(
                ["curl", "-L", source, "-o", str(destination)], check=True
            )
            print(f"[+] Download completed successfully at: {destination}")
        except subprocess.CalledProcessError:
            raise RuntimeError(f"Failed to download model: {source}")


class ModelService:
    """Service to manage model setup and organization."""

    def __init__(self, downloader):
        self._downloader = downloader

    def setup_model(self, url, target_dir):
        """Orchestrates the download of a model to the target directory."""
        filename = url.split("/")[-1]
        target_path = target_dir / filename
        try:
            self._downloader.download(url, target_path)
        except Exception as e:
            print(f"[!] Error in model service for {filename}: {e}")
