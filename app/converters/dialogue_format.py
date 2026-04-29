from typing import Dict, List, Optional

from transformers import AutoTokenizer


def format_sgd_to_messages(
    dialogues: List[Dict], max_turns: Optional[int] = None
) -> List[List[Dict[str, str]]]:
    """
    Converts SGD dataset format to a standard list of message dictionaries.
    Logic moved to converters for better modularity.
    """
    all_conversations = []
    if not dialogues:
        return all_conversations

    for dialogue in dialogues:
        turns = dialogue.get("turns", [])

        # Apply context window strategy (Long sequence handling)
        if max_turns and len(turns) > max_turns:
            turns = turns[-max_turns:]

        messages = []
        for turn in turns:
            speaker = turn.get("speaker", "").upper()
            # Map SGD speakers (USER/SYSTEM) to standard roles (user/assistant)
            role = "user" if speaker == "USER" else "assistant"
            content = turn.get("utterance", "").strip()

            if content:
                messages.append({"role": role, "content": content})

        if messages:
            all_conversations.append(messages)

    return all_conversations


class ChatTemplateService:
    """
    Service responsible for applying model-specific chat templates.
    Adheres to SOLID by decoupling formatting from inference logic.
    """

    def __init__(self, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Converts a list of messages (OpenAI format)
        into a model-specific string.
        """
        # apply_chat_template ensures we use
        # <|im_start|>, <|user|>, etc., correctly
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
