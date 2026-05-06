from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, TypeVar, runtime_checkable

from pydantic import BaseModel

Role = Literal["user", "assistant", "system"]
FinishReason = Literal["finished", "max_output_reached", "error"]

_T = TypeVar("_T", bound=BaseModel)


class ImageUrl(TypedDict):
    """OpenAI-style image URL — use a data: URI for base64-encoded images."""

    url: str


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImageUrlPart(TypedDict):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentPart = TextPart | ImageUrlPart


class Message(TypedDict):
    role: Role
    content: str | list[ContentPart]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Result of a single LLM call."""

    content: str
    finish_reason: FinishReason


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic LLM client.

    Implementations must be safe to share across coroutines for `acomplete`.
    """

    @property
    def model(self) -> str:
        """Get the model identifier."""
        ...

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Synchronous text completion.

        Args:
            messages: Conversation messages.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Returns:
            LLMResponse with generated content and finish reason.
        """
        ...

    async def acomplete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Asynchronous text completion.

        Args:
            messages: Conversation messages.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Returns:
            LLMResponse with generated content and finish reason.
        """
        ...

    def complete_structured(
        self,
        messages: list[Message],
        response_model: type[_T],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> _T:
        """Synchronous structured output completion.

        Args:
            messages: Conversation messages.
            response_model: Pydantic model for structured response.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Returns:
            Validated response model instance.
        """
        ...

    async def acomplete_structured(
        self,
        messages: list[Message],
        response_model: type[_T],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> _T:
        """Asynchronous structured output completion.

        Args:
            messages: Conversation messages.
            response_model: Pydantic model for structured response.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Returns:
            Validated response model instance.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to tokenize.

        Returns:
            Number of tokens.
        """
        ...
