from __future__ import annotations

import asyncio
import random
from typing import Any

from loguru import logger

from rag_pageindex.pageindex.llm.protocol import LLMClient
from rag_pageindex.pageindex.observability import observe
from rag_pageindex.pageindex.pdf.reader import Page
from rag_pageindex.pageindex.prompts import render
from rag_pageindex.pageindex.structured_responses import (
    PhysicalIndexResponse,
    TitleAppearanceResponse,
    TitleStartResponse,
)
from rag_pageindex.pageindex.toc.helpers import convert_physical_index_to_int


def validate_and_truncate_physical_indices(
    toc_items: list[dict[str, Any]],
    page_list_length: int,
    *,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    """Set physical_index to None for items that reference beyond the document."""
    if not toc_items:
        return toc_items
    max_allowed = page_list_length + start_index - 1
    truncated = 0
    for item in toc_items:
        if item.get("physical_index") is not None and item["physical_index"] > max_allowed:
            item["physical_index"] = None
            truncated += 1
    if truncated:
        logger.info(
            "validate_and_truncate: removed {} items beyond doc end (max={})",
            truncated,
            max_allowed,
        )
    return toc_items


async def check_title_appearance(
    item: dict[str, Any],
    pages: list[Page],
    *,
    start_index: int = 1,
    llm: LLMClient,
) -> dict[str, Any]:
    """Check whether a TOC item's title appears at its physical_index page."""
    title = item["title"]
    if item.get("physical_index") is None:
        return {
            "list_index": item.get("list_index"),
            "answer": "no",
            "title": title,
            "page_number": None,
        }
    page_number = item["physical_index"]
    page_text = pages[page_number - start_index].text
    prompt = render("check_title_appearance.j2", title=title, page_text=page_text)
    result = await llm.acomplete_structured(
        [{"role": "user", "content": prompt}], TitleAppearanceResponse
    )
    return {
        "list_index": item.get("list_index"),
        "answer": result.answer,
        "title": title,
        "page_number": page_number,
    }


async def check_title_appearance_in_start(
    title: str,
    page_text: str,
    *,
    llm: LLMClient,
) -> str:
    """Check if `title` is the first content on `page_text`. Returns 'yes'/'no'."""
    prompt = render("check_title_appearance_in_start.j2", title=title, page_text=page_text)
    result = await llm.acomplete_structured([{"role": "user", "content": prompt}], TitleStartResponse)
    return result.start_begin


@observe(name="check_title_appearance_in_start_concurrent")
async def check_title_appearance_in_start_concurrent(
    structure: list[dict[str, Any]],
    pages: list[Page],
    *,
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """Set `appear_start` on every structure item concurrently."""
    for item in structure:
        if item.get("physical_index") is None:
            item["appear_start"] = "no"

    valid_items = [item for item in structure if item.get("physical_index") is not None]
    tasks = [
        check_title_appearance_in_start(
            item["title"],
            pages[item["physical_index"] - 1].text,
            llm=llm,
        )
        for item in valid_items
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(valid_items, results, strict=False):
        if isinstance(result, Exception):
            logger.error(
                "check_title_appearance_in_start error for {}: {}",
                item["title"],
                result,
            )
            item["appear_start"] = "no"
        else:
            item["appear_start"] = result

    return structure


@observe(name="verify_toc")
async def verify_toc(
    pages: list[Page],
    list_result: list[dict[str, Any]],
    *,
    start_index: int = 1,
    sample_n: int | None = None,
    llm: LLMClient,
) -> tuple[float, list[dict[str, Any]]]:
    """Spot-check TOC item page locations. Returns (accuracy, incorrect_items)."""
    last_physical_index: int | None = None
    for item in reversed(list_result):
        if item.get("physical_index") is not None:
            last_physical_index = item["physical_index"]
            break

    if last_physical_index is None or last_physical_index < len(pages) / 2:
        return 0.0, []

    if sample_n is None:
        sample_indices: list[int] = list(range(len(list_result)))
    else:
        n = min(sample_n, len(list_result))
        sample_indices = random.sample(range(len(list_result)), n)

    indexed_items = []
    for idx in sample_indices:
        item = list_result[idx]
        if item.get("physical_index") is not None:
            item_copy = item.copy()
            item_copy["list_index"] = idx
            indexed_items.append(item_copy)

    tasks = [
        check_title_appearance(item, pages, start_index=start_index, llm=llm) for item in indexed_items
    ]
    results: list[dict[str, Any]] = list(await asyncio.gather(*tasks))

    correct_count = sum(1 for r in results if r["answer"] == "yes")
    incorrect = [r for r in results if r["answer"] != "yes"]
    accuracy = correct_count / len(results) if results else 0.0
    logger.info("verify_toc accuracy={:.2%} incorrect={}", accuracy, len(incorrect))
    return accuracy, incorrect


async def single_toc_item_index_fixer(
    section_title: str,
    content: str,
    *,
    llm: LLMClient,
) -> int | None:
    """Ask the LLM to locate `section_title` within `content`."""
    prompt = render(
        "single_toc_item_index_fixer.j2",
        section_title=section_title,
        content=content,
    )
    result = await llm.acomplete_structured([{"role": "user", "content": prompt}], PhysicalIndexResponse)
    return convert_physical_index_to_int(result.physical_index)


async def fix_incorrect_toc(
    toc_items: list[dict[str, Any]],
    pages: list[Page],
    incorrect_results: list[dict[str, Any]],
    *,
    start_index: int = 1,
    llm: LLMClient,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attempt to fix each incorrect TOC item's physical_index."""
    incorrect_indices = {r["list_index"] for r in incorrect_results}
    end_index = len(pages) + start_index - 1

    async def _fix_one(incorrect_item: dict[str, Any]) -> dict[str, Any]:
        list_index = incorrect_item["list_index"]
        if not (0 <= list_index < len(toc_items)):
            return {
                "list_index": list_index,
                "title": incorrect_item["title"],
                "physical_index": incorrect_item.get("physical_index"),
                "is_valid": False,
            }

        prev_correct = 0
        for j in range(list_index - 1, -1, -1):
            if j not in incorrect_indices and 0 <= j < len(toc_items):
                pi = toc_items[j].get("physical_index")
                if pi is not None:
                    prev_correct = pi
                    break

        next_correct = end_index
        for j in range(list_index + 1, len(toc_items)):
            if j not in incorrect_indices and 0 <= j < len(toc_items):
                pi = toc_items[j].get("physical_index")
                if pi is not None:
                    next_correct = pi
                    break

        page_contents = []
        for pi in range(prev_correct, next_correct + 1):
            li = pi - start_index
            if 0 <= li < len(pages):
                page_contents.append(
                    f"<physical_index_{pi}>\n{pages[li].text}\n<physical_index_{pi}>\n\n"
                )
        content_range = "".join(page_contents)

        physical_index = await single_toc_item_index_fixer(
            incorrect_item["title"], content_range, llm=llm
        )
        check_item = incorrect_item.copy()
        check_item["physical_index"] = physical_index
        check_result = await check_title_appearance(check_item, pages, start_index=start_index, llm=llm)
        return {
            "list_index": list_index,
            "title": incorrect_item["title"],
            "physical_index": physical_index,
            "is_valid": check_result["answer"] == "yes",
        }

    tasks = [_fix_one(item) for item in incorrect_results]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[dict[str, Any]] = []
    for item, result in zip(incorrect_results, raw_results, strict=False):
        if isinstance(result, BaseException):
            logger.error("fix_incorrect_toc error for {}: {}", item, result)
        else:
            results.append(result)

    invalid: list[dict[str, Any]] = []
    for result in results:
        li = result["list_index"]
        if result["is_valid"] and 0 <= li < len(toc_items):
            toc_items[li]["physical_index"] = result["physical_index"]
        else:
            invalid.append(result)

    logger.info(
        "fix_incorrect_toc: {} fixed, {} still invalid",
        len(results) - len(invalid),
        len(invalid),
    )
    return toc_items, invalid


@observe(name="fix_incorrect_toc_with_retries")
async def fix_incorrect_toc_with_retries(
    toc_items: list[dict[str, Any]],
    pages: list[Page],
    incorrect_results: list[dict[str, Any]],
    *,
    start_index: int = 1,
    max_attempts: int = 3,
    llm: LLMClient,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retry fix_incorrect_toc up to `max_attempts` times."""
    current_toc = toc_items
    current_incorrect = incorrect_results
    attempt = 0
    while current_incorrect and attempt < max_attempts:
        logger.info(
            "fix_incorrect_toc attempt {}/{} ({} remaining)",
            attempt + 1,
            max_attempts,
            len(current_incorrect),
        )
        current_toc, current_incorrect = await fix_incorrect_toc(
            current_toc,
            pages,
            current_incorrect,
            start_index=start_index,
            llm=llm,
        )
        attempt += 1
    if current_incorrect:
        logger.info(
            "fix_incorrect_toc_with_retries: {} items remain incorrect",
            len(current_incorrect),
        )
    return current_toc, current_incorrect
