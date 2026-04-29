"""
Centralized Chat Templates for different model architectures.
Isolating these long strings prevents linting errors in main logic files.
"""

# Llama 3.2 Official Template
# Reference: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
LLAMA_3_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)

# Qwen / ChatML Template
# Reference: https://huggingface.co/Qwen/Qwen3.5-2B
QWEN_3_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + "
    "message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)

# Global mapping for easy lookup
CHAT_TEMPLATES = {
    "llama": LLAMA_3_TEMPLATE,
    "qwen": QWEN_3_TEMPLATE,
}
