from typing import TypeVar

from langfuse import Langfuse
from pydantic import BaseModel

from rag_pageindex.pageindex.llm.protocol import LLMClient, LLMResponse, Message

_T = TypeVar("_T", bound=BaseModel)


def _serialize_messages(messages: list[Message]) -> list[dict[str, object]]:
    """Pass messages through as plain dicts so Langfuse renders them in the UI.

    Image content parts (data: URIs) are kept inline; the UI renders them.
    """
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def _has_images(messages: list[Message]) -> bool:
    return any(isinstance(m["content"], list) for m in messages)


class TracingLLMClient:
    """Decorator that records every LLMClient call as a Langfuse generation.

    Wraps any concrete `LLMClient`. Each call opens a Langfuse generation
    (nested under the current pipeline span if one is active), passes through
    to the inner client, then records the output before closing the span.
    Failures are surfaced both to Langfuse (as ERROR-level spans) and to the
    caller via re-raising.
    """

    def __init__(self, inner: LLMClient, langfuse: Langfuse) -> None:
        self._inner = inner
        self._lf = langfuse

    @property
    def model(self) -> str:
        return self._inner.model

    def _metadata(self, messages: list[Message], temperature: float) -> dict[str, object]:
        return {
            "provider_class": type(self._inner).__name__,
            "temperature": temperature,
            "multimodal": _has_images(messages),
        }

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        with self._lf.start_as_current_observation(
            name="llm.complete",
            as_type="generation",
            model=self._inner.model,
            input=_serialize_messages(messages),
            metadata=self._metadata(messages, temperature),
            model_parameters={"temperature": temperature, "max_tokens": max_tokens},
        ) as gen:
            try:
                response = self._inner.complete(messages, temperature=temperature, max_tokens=max_tokens)
            except Exception as exc:
                gen.update(level="ERROR", status_message=str(exc))
                raise
            gen.update(output=response.content, metadata={"finish_reason": response.finish_reason})
            return response

    async def acomplete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        with self._lf.start_as_current_observation(
            name="llm.acomplete",
            as_type="generation",
            model=self._inner.model,
            input=_serialize_messages(messages),
            metadata=self._metadata(messages, temperature),
            model_parameters={"temperature": temperature, "max_tokens": max_tokens},
        ) as gen:
            try:
                response = await self._inner.acomplete(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
            except Exception as exc:
                gen.update(level="ERROR", status_message=str(exc))
                raise
            gen.update(output=response.content, metadata={"finish_reason": response.finish_reason})
            return response

    def complete_structured(
        self,
        messages: list[Message],
        response_model: type[_T],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> _T:
        with self._lf.start_as_current_observation(
            name=f"llm.complete_structured[{response_model.__name__}]",
            as_type="generation",
            model=self._inner.model,
            input=_serialize_messages(messages),
            metadata={
                **self._metadata(messages, temperature),
                "response_model": response_model.__name__,
            },
            model_parameters={"temperature": temperature, "max_tokens": max_tokens},
        ) as gen:
            try:
                result = self._inner.complete_structured(
                    messages,
                    response_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                gen.update(level="ERROR", status_message=str(exc))
                raise
            gen.update(output=result.model_dump())
            return result

    async def acomplete_structured(
        self,
        messages: list[Message],
        response_model: type[_T],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> _T:
        with self._lf.start_as_current_observation(
            name=f"llm.acomplete_structured[{response_model.__name__}]",
            as_type="generation",
            model=self._inner.model,
            input=_serialize_messages(messages),
            metadata={
                **self._metadata(messages, temperature),
                "response_model": response_model.__name__,
            },
            model_parameters={"temperature": temperature, "max_tokens": max_tokens},
        ) as gen:
            try:
                result = await self._inner.acomplete_structured(
                    messages,
                    response_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                gen.update(level="ERROR", status_message=str(exc))
                raise
            gen.update(output=result.model_dump())
            return result

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)
