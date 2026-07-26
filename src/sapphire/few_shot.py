from pathlib import Path

import yaml


def load_few_shot_examples(path: str) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        data = yaml.safe_load(f) or []
    return [{"user": str(x["user"]), "assistant": str(x["assistant"])} for x in data]


def format_few_shot_examples(
    examples: list[dict[str, str]],
    username: str | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for ex in examples:
        content = f"{username}: {ex['user']}" if username else ex["user"]
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    return messages


def inject_few_shot_into_conversation(
    messages: list[dict[str, str]],
    few_shot_messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not messages:
        return list(few_shot_messages)
    system, *rest = messages
    return [system, *few_shot_messages, *rest]
