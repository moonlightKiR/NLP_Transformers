import subprocess
from abc import ABC, abstractmethod


class Downloader(ABC):
    """Abstract interface for data downloaders."""

    @abstractmethod
    def download(self, source, destination):
        pass


class GitDownloader(Downloader):
    """Git-based implementation for downloading repositories."""

    def download(self, source, destination):
        if destination.exists() and any(destination.iterdir()):
            print(
                f"[-] Dataset already exists at: {destination}. \
                    Skipping download."
            )
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[+] Downloading dataset from {source}...")
        try:
            subprocess.run(
                ["git", "clone", source, str(destination)],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"[+] Download completed successfully at: {destination}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone repository: {e.stderr}")


class DatasetService:
    """High-level service to manage datasets."""

    def __init__(self, downloader):
        self._downloader = downloader

    def setup_dataset(self, url, target_path):
        try:
            self._downloader.download(url, target_path)
        except Exception as e:
            print(f"[!] Critical error in dataset service: {e}")
