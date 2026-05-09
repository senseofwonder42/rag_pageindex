from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from rag_pageindex.pageindex.llm.protocol import ContentPart, LLMClient
from rag_pageindex.pageindex.observability import observe
from rag_pageindex.pageindex.pdf.reader import Page, PdfSource
from rag_pageindex.pageindex.pdf.renderer import image_part, render_pages
from rag_pageindex.pageindex.prompts import render
from rag_pageindex.pageindex.structured_responses import PhysicalIndexResponse
from rag_pageindex.pageindex.toc.helpers import convert_physical_index_to_int
from rag_pageindex.pageindex.toc.verification import check_title_appearance


async def _vlm_locate_title(
    title: str,
    page_indices: list[int],
    rendered: dict[int, bytes],
    *,
    llm: LLMClient,
    max_images_per_call: int = 10,
) -> tuple[int | None, str]:
    """Send page images to VLM; return (physical_index, confidence)."""
    if not page_indices:
        return None, "low"

    chunk_size = max(1, max_images_per_call)
    for chunk_start in range(0, len(page_indices), chunk_size):
        chunk = page_indices[chunk_start : chunk_start + chunk_size]
        prompt_text = render(
            "single_toc_item_index_fixer_vlm.j2",
            section_title=title,
            start_page=chunk[0],
            num_images=len(chunk),
        )
        parts: list[ContentPart] = [{"type": "text", "text": prompt_text}]
        for idx in chunk:
            if idx in rendered:
                parts.append(image_part(rendered[idx]))

        result = await llm.acomplete_structured(
            [{"role": "user", "content": parts}], PhysicalIndexResponse
        )
        physical_index = convert_physical_index_to_int(result.physical_index)
        if physical_index is not None:
            return physical_index, result.confidence
    return None, "low"


@observe(name="fix_incorrect_toc_with_vlm")
async def fix_incorrect_toc_with_vlm(
    toc_items: list[dict[str, Any]],
    pages: list[Page],
    incorrect_results: list[dict[str, Any]],
    source: PdfSource,
    *,
    start_index: int = 1,
    dpi: int = 144,
    max_images_per_call: int = 10,
    llm: LLMClient,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """VLM-based second-stage fixer for items that the text fixer couldn't place.

    Lazily renders only the candidate page range for each failing item.
    """
    incorrect_indices = {r["list_index"] for r in incorrect_results}
    end_index = len(pages) + start_index - 1

    # Collect all page ranges we'll need, then render once
    ranges: dict[int, list[int]] = {}
    for item in incorrect_results:
        list_index = item["list_index"]
        if not (0 <= list_index < len(toc_items)):
            continue

        prev_correct = start_index
        for j in range(list_index - 1, -1, -1):
            if j not in incorrect_indices:
                pi = toc_items[j].get("physical_index")
                if pi is not None:
                    prev_correct = pi
                    break

        next_correct = end_index
        for j in range(list_index + 1, len(toc_items)):
            if j not in incorrect_indices:
                pi = toc_items[j].get("physical_index")
                if pi is not None:
                    next_correct = min(pi, end_index)
                    break

        ranges[list_index] = list(range(prev_correct, min(next_correct + 1, end_index + 1)))

    all_pages = sorted({p for r in ranges.values() for p in r})
    rendered = render_pages(source, all_pages, dpi=dpi) if all_pages else {}

    async def _fix_one(incorrect_item: dict[str, Any]) -> dict[str, Any]:
        list_index = incorrect_item["list_index"]
        page_indices = ranges.get(list_index, [])

        physical_index, confidence = await _vlm_locate_title(
            incorrect_item["title"],
            page_indices,
            rendered,
            llm=llm,
            max_images_per_call=max_images_per_call,
        )

        in_range = physical_index is not None and start_index <= physical_index <= end_index
        if confidence == "high" and in_range:
            return {
                "list_index": list_index,
                "title": incorrect_item["title"],
                "physical_index": physical_index,
                "is_valid": True,
            }
        check_item = incorrect_item.copy()
        check_item["physical_index"] = physical_index
        check_result = await check_title_appearance(check_item, pages, start_index=start_index, llm=llm)
        return {
            "list_index": list_index,
            "title": incorrect_item["title"],
            "physical_index": physical_index,
            "is_valid": check_result["answer"] == "yes",
        }

    raw_results = await asyncio.gather(
        *[_fix_one(item) for item in incorrect_results], return_exceptions=True
    )

    still_invalid: list[dict[str, Any]] = []
    fixed = 0
    for item, result in zip(incorrect_results, raw_results, strict=False):
        if isinstance(result, BaseException):
            logger.error("fix_incorrect_toc_with_vlm error for {}: {}", item["title"], result)
            still_invalid.append(item)
        elif result["is_valid"]:
            li = result["list_index"]
            if 0 <= li < len(toc_items):
                toc_items[li]["physical_index"] = result["physical_index"]
                fixed += 1
        else:
            still_invalid.append(result)

    logger.info(
        "fix_incorrect_toc_with_vlm: {} fixed, {} still invalid",
        fixed,
        len(still_invalid),
    )
    return toc_items, still_invalid
