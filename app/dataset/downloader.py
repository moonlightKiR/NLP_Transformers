import hashlib
import json
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path


class Downloader(ABC):
    """Abstract interface for data downloaders."""

    @abstractmethod
    def download(self, source: str, destination: Path) -> None:
        pass


class IntegrityVerifier:
    """Handles data integrity checks using MD5 hashes and manifests."""

    @staticmethod
    def calculate_md5(file_path: Path) -> str:
        """Calculates the MD5 hash of a single file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def generate_manifest(self, directory: Path) -> dict[str, str]:
        """Generates a dictionary of {relative_path: md5_hash}."""
        manifest = {}
        for root, _, files in os.walk(directory):
            for file in files:
                if ".git" in root or file.startswith(".git"):
                    continue
                file_path = Path(root) / file
                relative_path = str(file_path.relative_to(directory))
                manifest[relative_path] = self.calculate_md5(file_path)
        return manifest

    def save_manifest(self, manifest: dict[str, str], path: Path) -> None:
        """Saves the manifest to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)
        print(f"[+] Manifest saved to {path}")

    def load_manifest(self, path: Path) -> dict[str, str] | None:
        """Loads the manifest from a JSON file."""
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class GitDownloader(Downloader):
    """Git-based implementation for downloading repositories."""

    def download(self, source: str, destination: Path) -> None:
        if self._is_already_downloaded(destination):
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

    def _is_already_downloaded(self, path: Path) -> bool:
        return path.exists() and any(path.iterdir())


class DatasetService:
    """High-level service to manage datasets and their integrity."""

    def __init__(self, downloader: Downloader, verifier: IntegrityVerifier):
        self._downloader = downloader
        self._verifier = verifier

    def setup_dataset(
        self, url: str, target_path: Path, manifest_path: Path
    ) -> None:
        try:
            # 1. Ensure dataset is present
            self._downloader.download(url, target_path)

            # 2. Integrity logic
            print(f"[+] Verifying data integrity for: {target_path.name}...")
            current_manifest = self._verifier.generate_manifest(target_path)
            stored_manifest = self._verifier.load_manifest(manifest_path)

            if stored_manifest is None:
                print("[!] No manifest found. Creating a new one...")
                self._verifier.save_manifest(current_manifest, manifest_path)
            else:
                self._compare_manifests(current_manifest, stored_manifest)

        except Exception as e:
            print(f"[!] Critical error in dataset service: {e}")

    def _compare_manifests(
        self, current: dict[str, str], stored: dict[str, str]
    ) -> None:
        """Compares two manifests and reports discrepancies."""
        errors = []

        # Check for changed or missing files
        for file_path, stored_hash in stored.items():
            if file_path not in current:
                errors.append(f"Missing file: {file_path}")
            elif current[file_path] != stored_hash:
                errors.append(f"Modified file (hash mismatch): {file_path}")

        # Check for new files not in manifest
        for file_path in current:
            if file_path not in stored:
                errors.append(f"New untracked file found: {file_path}")

        if errors:
            print("[✘] Integrity check FAILED:")
            for error in errors:
                print(f"    - {error}")
            raise ValueError("Dataset integrity compromised.")

        print(
            f"[✓] Integrity verified against manifest. {len(current)} \
                  files OK."
        )
