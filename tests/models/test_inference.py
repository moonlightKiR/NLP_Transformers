import torch
from app.config import settings
from app.models.config import model_settings
from app.models.inference import InferenceService
from app.dataset.processor import DialogueProcessor

def run_initial_inference_test(model_label="qwen"):
    """
    Test script to verify initial inference with a real structured dialogue.
    """
    print(f"--- Initial Inference Test: {model_label.upper()} ---")
    
    # 1. Paths
    if model_label == "qwen":
        model_file = model_settings.gguf_dir / "Qwen3.5-9B-Q3_K_M.gguf"
        tokenizer_path = model_settings.qwen_tok_path
    else:
        model_file = model_settings.gguf_dir / "gemma-4-E2B-it-Q3_K_M.gguf"
        tokenizer_path = model_settings.gemma_tok_path

    # 2. Load a sample dialogue from structured data
    processor = DialogueProcessor()
    sample_file = settings.structured_path / "test" / "dialogues_001.json"
    
    if not sample_file.exists():
        print(f"[!] No structured sample found at {sample_file}. Run uv run nlp-t first.")
        return

    dialogues = processor.load_json(sample_file)
    # Get the first 3 turns of the first dialogue as context
    test_context = dialogues[0][:3]
    
    print("\n[Contexto del Diálogo]")
    for msg in test_context:
        print(f"  {msg['role'].upper()}: {msg['content']}")

    # 3. Setup and Run Inference
    # Note: For local GGUF, from_pretrained needs the directory
    inference = InferenceService(
        model_path=model_file, 
        tokenizer_path=tokenizer_path,
        model_label=model_label
    )
    
    try:
        response = inference.generate_response(test_context)
        print(f"\n[Respuesta Generada por {model_label.upper()}]")
        print(f"  ASSISTANT: {response}")
    except Exception as e:
        print(f"[!] Inference failed: {e}")

if __name__ == "__main__":
    # You can change to "gemma" to test the other model
    run_initial_inference_test("qwen")
