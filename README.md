# Practica 2: Transformers en NLP

Este proyecto se centra en la implementación y evaluación comparativa de modelos Transformer (Qwen 3.5 y Phi-4) para la generación de diálogos, optimizado para ejecutarse en hardware local y entornos de alto rendimiento.

## Objectives
- Implement and fine-tune Transformer models for Dialogue Response Generation.
- Perform a comparative evaluation between **Qwen 3.5-2B** and **Phi-4-mini** architectures.
- Apply software engineering principles (**SOLID**, **Clean Code**).
- **Cross-Platform Portability**: Transparent execution on Apple Silicon (MPS) and NVIDIA (CUDA).

## Tech Stack
- **Language**: Python 3.12+
- **Core**: uv, Transformers, PyTorch.
- **Fine-Tuning**: MLX (Apple), PEFT/BitsAndBytes (NVIDIA).
- **Optimization**: Optuna.
- **Hardware Acceleration**: Apple Silicon GPU (Metal) & NVIDIA CUDA.

## Project Structure
```text
├── app/
│   ├── dataset/          # Data ingestion and SGD processing
│   ├── models/           # Models logic and Trainer abstraction
│   │   ├── trainer_factory.py  # Factory Pattern for hardware selection
│   │   ├── trainer_mlx.py      # Apple Silicon specific logic
│   │   ├── trainer_cuda.py     # NVIDIA/Colab specific logic
│   │   ├── orchestrator.py     # Inference management
│   │   └── config.py
│   ├── utils/
│   │   └── hardware.py   # Automatic Device Detector (MPS/CUDA/CPU)
│   ├── training/         # Section 5 & 7 training logic
│   └── main.py           # Pipeline Entry point
├── .adapters/            # LoRA weights and Optuna trials (ignored)
├── .data/                # Local datasets (ignored)
├── .models/              # Local GGUF models (ignored)
├── report/               # LaTeX documentation
└── README.md
```

## Advanced Features

### Hardware-Agnostic Portability (NEW)
The system features an automatic **Hardware Detector** and a **Trainer Factory**. 
- **macOS**: Uses the high-performance **MLX** backend native to Apple Silicon.
- **Linux/Colab**: Automatically switches to the **PyTorch/CUDA** stack with **QLoRA (4-bit)** to maximize NVIDIA T4/A100 efficiency.
This allows moving the project between a local Mac and Google Colab without modifying any code.

### Systematic Optimization
- **Optuna Integration**: Automated hyperparameter sweep (Learning Rate, Rank) evaluating Perplexity, Coherence, and Diversity.
- **Strict Chat Templates**: Uses official model tokenizers to ensure the exact prompt format (`<|im_start|>`) used during original training.

## Usage
```bash
uv sync                      # Install dependencies
uv run nlp-main              # Initial inference & Preprocessing
uv run nlp-train             # Section 5: Optuna Optimization
uv run nlp-train-lora        # Section 7: Final LoRA Training
```

## Maintainer
- Guillem Gonzalo - [guillemgonzalo2001@gmail.com](mailto:guillemgonzalo2001@gmail.com)
