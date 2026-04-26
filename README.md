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
├── .data/                # Local datasets (ignored by git, hidden)
├── .models/              # Local GGUF models and tokenizers (ignored by git, hidden)
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

3. Download and Preprocess Resources (Dataset and Models):
   ```bash
   uv run nlp-t
   ```
   Note: Resources will be stored in .data/ and .models/ respectively. The system will automatically perform tokenization, truncation, and padding, saving the results as PyTorch tensors.

## Code Standards
The project follows strict typing and style rules. You can validate the code manually with:
```bash
uv run pre-commit run --all-files
```

## Maintainer
- Guillem Gonzalo - [guillemgonzalo2001@gmail.com](mailto:guillemgonzalo2001@gmail.com)
