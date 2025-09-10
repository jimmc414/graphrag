"""A simple local provider that simulates OpenAI Codex responses.

This provider is intentionally lightweight so that GraphRAG can be
run without an API key.  It mirrors the interface used by other
language models in the project, but it simply echoes a deterministic
response.  Users can replace the logic in :func:`chat` with calls to a
local Codex implementation if desired.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING, Any

from graphrag.language_model.response.base import (
    BaseModelOutput,
    BaseModelResponse,
    ModelResponse,
)

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from graphrag.config.models.language_model_config import LanguageModelConfig


class OpenAICodexLocal:
    """A minimal ChatModel provider that does not require an API key."""

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: Any | None = None,
        cache: Any | None = None,
        responses: list[str] | None = None,
    ) -> None:
        self.config = config
        self._responses = responses or ["Codex response"]
        self._response_index = 0

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> ModelResponse:
        """Asynchronously generate a response for the given prompt."""

        return self.chat(prompt, history=history, **kwargs)

    def chat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> ModelResponse:
        """Generate a response for the given prompt."""

        response = self._responses[self._response_index % len(self._responses)]
        self._response_index += 1
        return BaseModelResponse(output=BaseModelOutput(content=response))

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Stream the response for the given prompt."""

        yield self.chat(prompt, history=history, **kwargs).output.content

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> Generator[str, None, None]:
        """Streaming is not supported for the synchronous interface."""

        raise NotImplementedError("chat_stream is not supported for CodexLocal")
