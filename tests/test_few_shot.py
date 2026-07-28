import pytest
import tempfile
from pathlib import Path
from sapphire.few_shot import (
    load_few_shot_examples,
    format_few_shot_examples,
    inject_few_shot_into_conversation,
)


SAMPLE_YAML = """
- user: hello
  assistant: hi there
- user: how are you
  assistant: i am fine
"""


@pytest.fixture
def yaml_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(SAMPLE_YAML)
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


class TestLoadFewShotExamples:
    def test_loads_examples(self, yaml_file):
        examples = load_few_shot_examples(yaml_file)
        assert len(examples) == 2
        assert examples[0] == {"user": "hello", "assistant": "hi there"}
        assert examples[1] == {"user": "how are you", "assistant": "i am fine"}

    def test_missing_file_returns_empty(self):
        assert load_few_shot_examples("/tmp/nonexistent_file.yml") == []


class TestFormatFewShotExamples:
    def test_without_username(self):
        examples = [{"user": "hello", "assistant": "hi"}]
        messages = format_few_shot_examples(examples)
        assert messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_with_username(self):
        examples = [{"user": "hello", "assistant": "hi"}]
        messages = format_few_shot_examples(examples, "Alice")
        assert messages == [
            {"role": "user", "content": "Alice: hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_empty_examples(self):
        assert format_few_shot_examples([]) == []


class TestInjectFewShot:
    def test_injects_after_system(self):
        messages = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hello"},
        ]
        few_shot = [
            {"role": "user", "content": "example"},
            {"role": "assistant", "content": "response"},
        ]
        result = inject_few_shot_into_conversation(messages, few_shot)
        assert result == [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "example"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "hello"},
        ]

    def test_empty_messages_returns_few_shot_only(self):
        few_shot = [{"role": "user", "content": "hi"}]
        assert inject_few_shot_into_conversation([], few_shot) == few_shot

    def test_empty_few_shot_returns_messages_unchanged(self):
        messages = [{"role": "system", "content": "system"}]
        assert inject_few_shot_into_conversation(messages, []) == messages
