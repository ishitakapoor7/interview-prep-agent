"""Thin wrapper over the Anthropic SDK.

Two methods only: free text, and schema-enforced JSON. Structured output is done
by defining a single tool and forcing the model to call it — the tool's input
schema becomes the output contract, which is more reliable than asking for JSON
in the prompt and parsing it back.

The real SDK client is created lazily so tests can inject a fake and never need
an API key.
"""

from __future__ import annotations

from typing import Any

from app.config import ANTHROPIC_API_KEY, MODEL

_EMIT_TOOL = "emit_result"


class LlmClient:
    def __init__(self, client: Any = None, model: str = MODEL) -> None:
        self._client = client
        self._model = model

    def _get(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        resp = self._get().messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def complete_json(
        self, system: str, user: str, schema: dict, max_tokens: int = 8192
    ) -> dict:
        """Force a single tool call whose input schema IS the desired output shape."""
        resp = self._get().messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": _EMIT_TOOL,
                    "description": "Emit the structured result. Always use this tool.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": _EMIT_TOOL},
        )
        for block in resp.content:
            if getattr(block, "type", "") == "tool_use" and block.name == _EMIT_TOOL:
                return dict(block.input)
        raise ValueError("model did not return structured output")
