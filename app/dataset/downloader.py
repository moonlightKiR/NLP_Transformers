import subprocess
from abc import ABC, abstractmethod
from pathlib import Path


class Downloader(ABC):
    """Interfaz abstracta para descargadores de datos."""

    @abstractmethod
    def download(self, source: str, destination: Path) -> None:
        pass


class GitDownloader(Downloader):
    """Implementación de descarga mediante Git."""

    def download(self, source: str, destination: Path) -> None:
        if destination.exists():
            print(f"[-] El repositorio ya existe en: {destination}")
            return

        print(f"[+] Clonando desde {source}...")
        result = subprocess.run(
            ["git", "clone", source, str(destination)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Error en git clone: {result.stderr}")

        print(f"[+] Descarga completada en: {destination}")


class DatasetService:
    """Servicio de alto nivel para gestionar datasets."""

    def __init__(self, downloader: Downloader):
        self._downloader = downloader

    def setup_dataset(self, url: str, target_path: Path) -> None:
        try:
            self._downloader.download(url, target_path)
        except Exception as e:
            print(f"[!] Fallo en el servicio de dataset: {e}")
