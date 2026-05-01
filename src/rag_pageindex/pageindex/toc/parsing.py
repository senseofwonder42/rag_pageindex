from __future__ import annotations

from typing import Any

from loguru import logger

from rag_pageindex.pageindex.json_extract import extract_json, get_json_content
from rag_pageindex.pageindex.llm.protocol import LLMClient
from rag_pageindex.pageindex.prompts import render
from rag_pageindex.pageindex.toc.detection import (
    check_if_toc_transformation_is_complete,
)
from rag_pageindex.pageindex.toc.helpers import convert_page_to_int


def toc_transformer(
    toc_content: str, *, llm: LLMClient
) -> list[dict[str, Any]]:
    """Transform raw TOC text into a structured list-of-dicts.

    Loops with continuation prompts until the model emits a complete JSON.
    """
    logger.debug("toc_transformer: start")
    prompt = render("toc_to_json.j2", toc_content=toc_content)
    response = llm.complete([{"role": "user", "content": prompt}])
    last_complete_text = response.content
    finish_reason = response.finish_reason

    if (
        check_if_toc_transformation_is_complete(
            toc_content, last_complete_text, llm=llm
        )
        == "yes"
        and finish_reason == "finished"
    ):
        parsed = extract_json(last_complete_text)
        return convert_page_to_int(parsed["table_of_contents"])

    last_complete_text = get_json_content(last_complete_text)

    attempt = 0
    if_complete = "no"
    while not (if_complete == "yes" and finish_reason == "finished"):
        attempt += 1
        if attempt > 5:
            raise RuntimeError(
                "Failed to complete toc transformation after maximum retries"
            )
        position = last_complete_text.rfind("}")
        if position != -1:
            last_complete_text = last_complete_text[: position + 2]

        continue_prompt = render(
            "toc_to_json_continue.j2",
            raw_toc=toc_content,
            partial_json=last_complete_text,
        )
        new = llm.complete([{"role": "user", "content": continue_prompt}])
        if new.content.startswith("```json"):
            last_complete_text += get_json_content(new.content)
        finish_reason = new.finish_reason
        if_complete = check_if_toc_transformation_is_complete(
            toc_content, last_complete_text, llm=llm
        )

    parsed = extract_json(last_complete_text)
    return convert_page_to_int(parsed["table_of_contents"])


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
    response = llm.complete([{"role": "user", "content": prompt}])
    return extract_json(response.content)


def add_page_number_to_toc(
    part: str,
    structure: list[dict[str, Any]],
    *,
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """Ask the LLM whether each TOC item starts in `part`; emit physical_index."""
    import json as _json

    structure_str = _json.dumps(structure, indent=2)
    prompt = render(
        "add_page_number_to_toc.j2",
        part=part,
        structure=structure_str,
    )
    response = llm.complete([{"role": "user", "content": prompt}])
    json_result = extract_json(response.content)
    for item in json_result:
        item.pop("start", None)
    return json_result
