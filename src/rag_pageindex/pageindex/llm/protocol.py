from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, runtime_checkable

Role = Literal["user", "assistant", "system"]
FinishReason = Literal["finished", "max_output_reached", "error"]


class Message(TypedDict):
    role: Role
    content: str


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
    def model(self) -> str: ...

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    async def acomplete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    def count_tokens(self, text: str) -> int: ...
