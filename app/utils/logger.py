import sys
from datetime import datetime
from pathlib import Path


class FileLogger:
    """
    Utility to redirect stdout to both console and a log file.
    """

    def __init__(self, service_name: str):
        self.terminal = sys.stdout
        log_dir = Path("results/logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{service_name}_{timestamp}.log"
        self.log_path = log_dir / log_filename
        self.log_file = open(self.log_path, "w", encoding="utf-8")

        # Immediate header
        header = f"=== Log started for {service_name} at {timestamp} ===\n"
        self.terminal.write(header)
        self.log_file.write(header)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def __del__(self):
        if hasattr(self, "log_file"):
            self.log_file.close()


def setup_logger(service_name: str):
    """
    Intercepts sys.stdout and redirects it to FileLogger.
    """
    sys.stdout = FileLogger(service_name)
    print(f"[Logger] Capturing output to: results/logs/{service_name}_*.log")
