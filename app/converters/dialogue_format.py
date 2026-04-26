def format_sgd_to_messages(dialogues, max_turns=None):
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
