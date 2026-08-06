import json

import pytest

from app.llm import LlmClient


class _FakeBlock:
    def __init__(self, text=None, name=None, input=None):
        self.text = text
        self.type = "tool_use" if name else "text"
        self.name = name
        self.input = input


class _FakeResponse:
    def __init__(self, blocks):
        self.content = blocks


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


def test_complete_returns_concatenated_text():
    fake = _FakeClient(_FakeResponse([_FakeBlock(text="hello "), _FakeBlock(text="world")]))
    client = LlmClient(client=fake)
    assert client.complete(system="s", user="u") == "hello world"


def test_complete_passes_system_and_user_through():
    fake = _FakeClient(_FakeResponse([_FakeBlock(text="ok")]))
    client = LlmClient(client=fake)
    client.complete(system="be terse", user="hi")
    kwargs = fake.messages.last_kwargs
    assert kwargs["system"] == "be terse"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_complete_json_returns_tool_input_dict():
    payload = {"gaps": ["missing founding year"]}
    fake = _FakeClient(
        _FakeResponse([_FakeBlock(name="emit_result", input=payload)])
    )
    client = LlmClient(client=fake)
    schema = {"type": "object", "properties": {"gaps": {"type": "array"}}}
    assert client.complete_json(system="s", user="u", schema=schema) == payload


def test_complete_json_forces_the_emit_tool():
    schema = {"type": "object", "properties": {"gaps": {"type": "array"}}}
    fake = _FakeClient(_FakeResponse([_FakeBlock(name="emit_result", input={})]))
    client = LlmClient(client=fake)
    client.complete_json(system="s", user="u", schema=schema)
    kwargs = fake.messages.last_kwargs
    assert kwargs["tool_choice"] == {"type": "tool", "name": "emit_result"}
    assert kwargs["tools"][0]["name"] == "emit_result"
    # The caller's schema must pass through untouched -- it IS the output
    # contract. Four call sites (identify_gaps, extract_facts,
    # generate_lesson_plan, grade_answer) depend on this being an exact
    # pass-through, not a copy or a reshaped subset.
    assert kwargs["tools"][0]["input_schema"] == schema


def test_complete_json_raises_when_model_returns_no_tool_use():
    fake = _FakeClient(_FakeResponse([_FakeBlock(text="I refuse")]))
    client = LlmClient(client=fake)
    with pytest.raises(ValueError, match="did not return structured output"):
        client.complete_json(system="s", user="u", schema={"type": "object"})
