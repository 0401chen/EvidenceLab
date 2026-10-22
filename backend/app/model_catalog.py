"""Server-controlled model selection; a catalogue entry is not access approval."""
from __future__ import annotations

import re

DEFAULT_MODEL = "gpt-4o-mini"

CATALOG = (
    "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5-mini",
    "deepseek-flash", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
)


def valid_model_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}", value))


def transport_options(model: str) -> dict:
    # Reasoning models reject temperature=0 and use completion-token budgets.
    # Compatible gateways may still reject these options: never switch models.
    if re.match(r"^(?:gpt-5(?:-|$)|o[134](?:-|$))", model):
        return {"max_completion_tokens": 6500, "response_format": {"type": "json_object"}}
    if "deepseek" in model.lower():
        return {"max_tokens": 6500}  # JSON is instructed; not assumed supported.
    return {"temperature": 0, "max_tokens": 6500, "response_format": {"type": "json_object"}}
