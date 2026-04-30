# Practica 2: Transformers en NLP

Este proyecto se centra en la implementación y evaluación comparativa de modelos Transformer (Qwen 3.5 y Llama 3.2) para la generación de diálogos, optimizado para ejecutarse en hardware local y entornos de alto rendimiento.

## Objectives
- Implement and fine-tune Transformer models for Dialogue Response Generation.
- Perform a comparative evaluation between **Qwen 3.5-2B** and **Llama 3.2-1B** architectures.
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
├── results/              # Persistent metadata (tracked by Git)
│   ├── optuna/           # SQLite databases with trial history
│   ├── lora/             # YAML configuration traces for each experiment
├── .adapters/            # LoRA weights (ignored)
├── .data/                # Local datasets (ignored)
├── .models/              # Local GGUF models (ignored)
├── report/               # LaTeX documentation
└── README.md
```

## Advanced Features

### Hardware-Agnostic Portability
The system features an automatic **Hardware Detector** and a **Trainer Factory**. 
- **macOS**: Uses the **MLX** backend with **isolated subprocess execution** to ensure 100% GPU memory reclamation between trials.
- **Linux/Colab**: Automatically switches to **PyTorch/CUDA** with **QLoRA (4-bit)**.

### Systematic Optimization & Traceability
- **Self-Cleaning Persistence**: Results are saved in SQLite databases (`results/optuna/`). The system automatically synchronizes the database with the local adapters, removing stale data on clean starts.
- **Strict Trial Management**: Ensures a fixed number of successful trials (3) per architecture, preventing database bloat and ensuring consistent report metrics.
- **Config Traces**: Every training run saves a YAML snapshot in `results/lora/` for full reproducibility.

### Technical Visualization
Integrated visualization suite for:
- **Attention Maps**: Heatmaps of self-attention weights to analyze model interpretability.
- **Optimization History**: Plots of hyperparameter convergence from Optuna trials.

## Usage
```bash
uv sync                      # Install dependencies
uv run nlp-main              # Initial inference & Preprocessing
uv run nlp-train             # Section 5: Optuna Optimization (Persistence in results/)
uv run nlp-viz               # Generate Attention Maps & Optuna Plots
uv run nlp-train-lora        # Section 7: Final LoRA Training
```

## Maintainer
- Guillem Gonzalo - [guillemgonzalo2001@gmail.com](mailto:guillemgonzalo2001@gmail.com)
