from __future__ import annotations

import json as _json
from typing import Any

from loguru import logger

from rag_pageindex.pageindex.llm.protocol import LLMClient
from rag_pageindex.pageindex.prompts import render
from rag_pageindex.pageindex.structured_responses import (
    TocIndexResponse,
    TocPageNumberResponse,
    TocTransformResponse,
)
from rag_pageindex.pageindex.toc.helpers import convert_page_to_int


def toc_transformer(
    toc_content: str, *, llm: LLMClient, max_tokens: int | None = None
) -> list[dict[str, Any]]:
    """Transform raw TOC text into a structured list-of-dicts via structured output."""
    logger.debug("toc_transformer: start")
    prompt = render("toc_to_json.j2", toc_content=toc_content)
    result = llm.complete_structured(
        [{"role": "user", "content": prompt}],
        TocTransformResponse,
        max_tokens=max_tokens,
    )
    return convert_page_to_int([e.model_dump() for e in result.table_of_contents])


def toc_index_extractor(
    toc: object,
    content: str,
    *,
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """Tag a TOC with physical indices using pages tagged with <physical_index_X>."""
    logger.debug("toc_index_extractor: start")
    prompt = render(
        "toc_index_extractor.j2",
        toc=str(toc),
        content=content,
    )
    result = llm.complete_structured([{"role": "user", "content": prompt}], TocIndexResponse)
    return [e.model_dump() for e in result.items]


def add_page_number_to_toc(
    part: str,
    structure: list[dict[str, Any]],
    *,
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """Ask the LLM whether each TOC item starts in `part`; emit physical_index."""
    structure_str = _json.dumps(structure, indent=2)
    prompt = render(
        "add_page_number_to_toc.j2",
        part=part,
        structure=structure_str,
    )
    result = llm.complete_structured([{"role": "user", "content": prompt}], TocPageNumberResponse)
    return [e.model_dump(exclude={"start"}) for e in result.items]


async def add_page_number_to_toc_async(
    part: str,
    structure: list[dict[str, Any]],
    *,
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """Async variant of add_page_number_to_toc."""
    structure_str = _json.dumps(structure, indent=2)
    prompt = render(
        "add_page_number_to_toc.j2",
        part=part,
        structure=structure_str,
    )
    result = await llm.acomplete_structured([{"role": "user", "content": prompt}], TocPageNumberResponse)
    return [e.model_dump(exclude={"start"}) for e in result.items]
