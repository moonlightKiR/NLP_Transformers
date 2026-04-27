from app.dataset.processor import DialogueProcessor
from app.converters.dialogue_format import format_sgd_to_messages

def test_dialogue_extraction():
    """Verifies that the DialogueProcessor and converter work correctly."""
    print("--- Testing Dialogue Extraction and Formatting ---")
    
    # Mock SGD data
    mock_raw_data = [{
        "dialogue_id": "test_01",
        "turns": [
            {"speaker": "USER", "utterance": "Hello"},
            {"speaker": "SYSTEM", "utterance": "Hi there!"}
        ]
    }]
    
    formatted = format_sgd_to_messages(mock_raw_data)
    
    assert len(formatted[0]) == 2
    assert formatted[0][0]["role"] == "user"
    assert formatted[0][1]["role"] == "assistant"
    
    print("[✓] Dialogue extraction test passed.")

if __name__ == "__main__":
    test_dialogue_extraction()
