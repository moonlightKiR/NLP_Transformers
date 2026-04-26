# Práctica 2: Transformers en NLP

Este proyecto forma parte de la asignatura de Deep Learning y se centra en la exploración, implementación y evaluación comparativa de modelos basados en la arquitectura **Transformer** para tareas de Procesamiento de Lenguaje Natural (NLP).

## 🎯 Objectives
- Implement and fine-tune Transformer models for **Dialogue Response Generation**.
- Perform a comparative evaluation between **Decoder-only** (Qwen3) and **Encoder-Decoder** (mT5) architectures.
- Use the **SGD (Schema-Guided Dialogue)** dataset from Google.
- Apply software engineering principles (**SOLID**, **Clean Code**).

## 🛠️ Tech Stack
- **Language:** Python 3.12+
- **Package Manager:** `uv`
- **Models:** Qwen3-1.5B, mT5-base
- **Code Quality:**
  - `pre-commit`: Hook automation.
  - `ruff`: Fast linting and formatting (PEP 8).
  - `mypy`: Static type checking.

## 📁 Project Structure
```text
├── app/
│   ├── dataset/          # Data downloading and processing logic
│   ├── models/           # Local GGUF models (Gemma 4, Qwen 3.5)
│   ├── config.py         # Dynamic configuration and path management
│   ├── constants.py      # Global project constants
│   └── main.py           # Entry point (Orchestrator)
├── .data/                # Local datasets (ignored by git, hidden)
├── report/               # LaTeX report and documentation
├── .pre-commit-config.yaml
├── pyproject.toml        # Dependencies and tool configuration
└── README.md
```

## 🚀 Installation and Usage

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Setup quality hooks:**
   ```bash
   uv run pre-commit install
   ```

3. **Download the Dataset (SGD):**
   ```bash
   uv run nlp-t
   ```
   *Note: The dataset will be downloaded into the `./.data/` directory.*

## 📝 Code Standards
The project follows strict typing and style rules. You can validate the code manually with:
```bash
uv run pre-commit run --all-files
```

## 👤 Maintainer
- **Guillem Gonzalo** - [guillemgonzalo2001@gmail.com](mailto:guillemgonzalo2001@gmail.com)
