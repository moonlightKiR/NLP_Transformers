# Practica 2: Transformers en NLP

Este proyecto forma parte de la asignatura de Deep Learning y se centra en la exploracion, implementacion y evaluacion comparativa de modelos basados en la arquitectura Transformer para tareas de Procesamiento de Lenguaje Natural (NLP).

## Objectives
- Implement and fine-tune Transformer models for Dialogue Response Generation.
- Perform a comparative evaluation between Gemma 4 E2B-it and Qwen 3.5-9B architectures.
- Use the SGD (Schema-Guided Dialogue) dataset from Google.
- Apply software engineering principles (SOLID, Clean Code).

## Tech Stack
- Language: Python 3.12+
- Package Manager: uv
- Models: Gemma 4 E2B-it, Qwen 3.5-9B (GGUF)
- Hardware Acceleration: Apple Silicon MPS (Metal Performance Shaders)
- Code Quality:
  - pre-commit: Hook automation.
  - ruff: Fast linting and formatting (PEP 8).

## Project Structure
```text
├── app/
│   ├── dataset/          # Data downloading and processing logic
│   ├── models/           # Models logic, constants and config
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── downloader.py
│   │   └── __init__.py
│   ├── config.py         # General project configuration
│   ├── constants.py      # Global project constants
│   └── main.py           # Entry point (Orchestrator)
├── tests/                # Unit and integration tests
├── .data/                # Local datasets and preprocessed tensors (ignored)
├── .models/              # Local GGUF models and tokenizers (ignored)
├── report/               # LaTeX report and documentation
│   └── app -> ../app     # Symlink to app for report listings
├── .pre-commit-config.yaml
├── pyproject.toml        # Dependencies and tool configuration
└── README.md
```

## Installation and Usage

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Setup quality hooks:
   ```bash
   uv run pre-commit install
   ```

3. Download and Preprocess Resources:
   ```bash
   uv run nlp-t
   ```
   Note: The system will automatically:
   - Fetch the SGD dataset and GGUF models.
   - Setup local tokenizers for offline use.
   - Perform bulk preprocessing (Tokenization, Truncation, Padding).
   - Apply long sequence handling strategies.
   - Save processed tensors as .pt files using Apple Silicon GPU acceleration.

## Advanced Features

### Hardware Optimization
The pipeline is optimized for macOS (M1/M2/M3) using the MPS backend, offloading intensive tokenization and tensor operations to the GPU.

### Long Sequence Handling
To manage extensive dialogue histories, the system combines architectural strengths with data strategies:
- Native Support: Utilization of Hybrid Attention (Gemma 4) and Gated Delta Networks (Qwen 3.5) for massive context windows.
- Context Windowing: Automated turn-truncation in the preprocessing phase to prioritize the most recent and relevant conversational context.

## Testing
The project uses structured tests in the `tests/` directory (mirroring the `app/` structure). To run specific tests using `uv`:
```bash
# Test inference
uv run python tests/models/test_inference.py

# Test data processing
uv run python tests/dataset/test_processor.py
```

## Code Standards
The project follows strict typing and style rules. You can validate the code manually with:
```bash
uv run pre-commit run --all-files
```

## Maintainer
- Guillem Gonzalo - [guillemgonzalo2001@gmail.com](mailto:guillemgonzalo2001@gmail.com)
