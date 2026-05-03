from typing import TypeVar

import httpx
from pydantic import BaseModel

from rag_pageindex.pageindex.llm.protocol import (
    FinishReason,
    LLMResponse,
    Message,
)
from rag_pageindex.pageindex.llm.retry import awith_retries, with_retries

_T = TypeVar("_T", bound=BaseModel)

_DEFAULT_TIMEOUT = 120.0
_CHARS_PER_TOKEN = 4  # rough estimate used when no tokenizer is available


def _map_finish_reason(reason: str | None) -> FinishReason:
    if reason == "length":
        return "max_output_reached"
    return "finished"


def _build_payload(
    model: str,
    messages: list[Message],
    temperature: float,
    max_tokens: int,
) -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


class OpenAICompatibleClient:
    """LLMClient backed by any OpenAI-compatible endpoint (OpenRouter, vLLM, etc.).

    Set `base_url` to the API root, e.g.:
      - OpenRouter : "https://openrouter.ai/api/v1"
      - Local vLLM : "http://localhost:8000/v1"
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_retries: int = 10,
        retry_delay_s: float = 1.0,
        max_output_tokens: int = 4096,
        timeout: float = _DEFAULT_TIMEOUT,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        self._retry_delay_s = retry_delay_s
        self._max_output_tokens = max_output_tokens

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        self._sync = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        self._async = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout)

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = _build_payload(
            self._model, messages, temperature, max_tokens or self._max_output_tokens
        )

        def _call() -> LLMResponse:
            resp = self._sync.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            return LLMResponse(
                content=choice["message"]["content"] or "",
                finish_reason=_map_finish_reason(choice.get("finish_reason")),
            )

        return with_retries(_call, max_retries=self._max_retries, delay_s=self._retry_delay_s)

    async def acomplete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = _build_payload(
            self._model, messages, temperature, max_tokens or self._max_output_tokens
        )

        async def _call() -> LLMResponse:
            resp = await self._async.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            return LLMResponse(
                content=choice["message"]["content"] or "",
                finish_reason=_map_finish_reason(choice.get("finish_reason")),
            )

        return await awith_retries(_call, max_retries=self._max_retries, delay_s=self._retry_delay_s)

    def complete_structured(
        self,
        messages: list[Message],
        response_model: type[_T],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> _T:
        payload = {
            **_build_payload(self._model, messages, temperature, max_tokens or self._max_output_tokens),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": False,
                    "schema": response_model.model_json_schema(),
                },
            },
        }

        def _call() -> _T:
            resp = self._sync.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            return response_model.model_validate_json(content)

        return with_retries(_call, max_retries=self._max_retries, delay_s=self._retry_delay_s)

    async def acomplete_structured(
        self,
        messages: list[Message],
        response_model: type[_T],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> _T:
        payload = {
            **_build_payload(self._model, messages, temperature, max_tokens or self._max_output_tokens),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": False,
                    "schema": response_model.model_json_schema(),
                },
            },
        }

        async def _call() -> _T:
            resp = await self._async.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            return response_model.model_validate_json(content)

        return await awith_retries(_call, max_retries=self._max_retries, delay_s=self._retry_delay_s)

    def count_tokens(self, text: str) -> int:
        return len(text) // _CHARS_PER_TOKEN
