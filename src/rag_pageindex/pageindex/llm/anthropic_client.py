from typing import TypeVar

from anthropic import Anthropic, AsyncAnthropic
from pydantic import BaseModel

from rag_pageindex.pageindex.llm.protocol import (
    FinishReason,
    LLMResponse,
    Message,
)
from rag_pageindex.pageindex.llm.retry import awith_retries, with_retries

_T = TypeVar("_T", bound=BaseModel)


def _check_no_multimodal(messages: list[Message]) -> None:
    """Verify that messages do not contain multimodal content blocks.

    Args:
        messages: List of messages to check.

    Raises:
        NotImplementedError: If any message has a list of content parts.
    """
    for msg in messages:
        if isinstance(msg["content"], list):
            raise NotImplementedError(
                "AnthropicClient does not support multimodal content blocks. "
                "Use OpenAICompatibleClient with a vision-capable model."
            )


def _split_system(
    messages: list[Message],
) -> tuple[str | None, list[Message]]:
    """Anthropic takes the system prompt as a separate kwarg."""
    system: str | None = None
    rest: list[Message] = []
    for msg in messages:
        if msg["role"] == "system":
            content = msg["content"]
            if not isinstance(content, str):
                raise TypeError("System message content must be str for AnthropicClient")
            system = content if system is None else f"{system}\n\n{content}"
        else:
            rest.append(msg)
    return system, rest


def _map_stop_reason(stop_reason: str | None) -> FinishReason:
    """Map Anthropic stop reason to protocol FinishReason.

    Args:
        stop_reason: Stop reason from Anthropic API.

    Returns:
        Normalized finish reason.
    """
    if stop_reason == "max_tokens":
        return "max_output_reached"
    return "finished"


class AnthropicClient:
    """LLMClient implementation backed by Anthropic's SDK with retry logic.

    Supports both sync and async completion. Does not support structured
    output (use OpenAICompatibleClient instead).
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_retries: int = 10,
        retry_delay_s: float = 1.0,
        max_output_tokens: int = 4096,
    ) -> None:
        """Initialize an AnthropicClient.

        Args:
            api_key: Anthropic API key.
            model: Model ID (e.g., 'claude-3-5-sonnet-20241022').
            max_retries: Maximum retries on transient failures.
            retry_delay_s: Base delay between retries in seconds.
            max_output_tokens: Default max tokens for completion.
        """
        self._sync = Anthropic(api_key=api_key)
        self._async = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._retry_delay_s = retry_delay_s
        self._max_output_tokens = max_output_tokens

    @property
    def model(self) -> str:
        """Get the model ID."""
        return self._model

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Synchronous text completion with retry.

        Args:
            messages: List of messages (system must be None or a single dict).
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Max tokens to generate; uses default if None.

        Returns:
            LLMResponse with content and finish_reason.
        """
        _check_no_multimodal(messages)
        system, rest = _split_system(messages)

        def _call() -> LLMResponse:
            kwargs: dict[str, object] = {
                "model": self._model,
                "messages": rest,
                "temperature": temperature,
                "max_tokens": max_tokens or self._max_output_tokens,
            }
            if system is not None:
                kwargs["system"] = system
            response = self._sync.messages.create(**kwargs)  # type: ignore[call-overload]
            text = "".join(
                block.text  # type: ignore[attr-defined]
                for block in response.content
                if getattr(block, "type", None) == "text"
            )
            return LLMResponse(
                content=text,
                finish_reason=_map_stop_reason(response.stop_reason),
            )

        return with_retries(
            _call,
            max_retries=self._max_retries,
            delay_s=self._retry_delay_s,
        )

    async def acomplete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Asynchronous text completion with retry.

        Args:
            messages: List of messages (system must be None or a single dict).
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Max tokens to generate; uses default if None.

        Returns:
            LLMResponse with content and finish_reason.
        """
        _check_no_multimodal(messages)
        system, rest = _split_system(messages)

        async def _call() -> LLMResponse:
            kwargs: dict[str, object] = {
                "model": self._model,
                "messages": rest,
                "temperature": temperature,
                "max_tokens": max_tokens or self._max_output_tokens,
            }
            if system is not None:
                kwargs["system"] = system
            response = await self._async.messages.create(**kwargs)  # type: ignore[call-overload]
            text = "".join(
                block.text  # type: ignore[attr-defined]
                for block in response.content
                if getattr(block, "type", None) == "text"
            )
            return LLMResponse(
                content=text,
                finish_reason=_map_stop_reason(response.stop_reason),
            )

        return await awith_retries(
            _call,
            max_retries=self._max_retries,
            delay_s=self._retry_delay_s,
        )

    def complete_structured(
        self,
        messages: list[Message],
        response_model: type[_T],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> _T:
        """Structured output not supported by AnthropicClient.

        Args:
            messages: Ignored.
            response_model: Ignored.
            temperature: Ignored.
            max_tokens: Ignored.

        Raises:
            NotImplementedError: Always; use OpenAICompatibleClient instead.
        """
        raise NotImplementedError(
            "AnthropicClient does not implement complete_structured. Use OpenAICompatibleClient."
        )

    async def acomplete_structured(
        self,
        messages: list[Message],
        response_model: type[_T],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> _T:
        """Structured output not supported by AnthropicClient.

        Args:
            messages: Ignored.
            response_model: Ignored.
            temperature: Ignored.
            max_tokens: Ignored.

        Raises:
            NotImplementedError: Always; use OpenAICompatibleClient instead.
        """
        raise NotImplementedError(
            "AnthropicClient does not implement acomplete_structured. Use OpenAICompatibleClient."
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in a message using Anthropic's token counter.

        Args:
            text: Text to tokenize.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        result = self._sync.messages.count_tokens(
            model=self._model,
            messages=[{"role": "user", "content": text}],
        )
        return result.input_tokens
