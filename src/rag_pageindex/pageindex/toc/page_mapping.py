from __future__ import annotations

import asyncio
import copy
import json as _json
import math
from typing import Any

from loguru import logger

from rag_pageindex.pageindex.llm.protocol import LLMClient
from rag_pageindex.pageindex.observability import observe
from rag_pageindex.pageindex.pdf.reader import Page
from rag_pageindex.pageindex.prompts import render
from rag_pageindex.pageindex.structured_responses import TocGeneratedResponse
from rag_pageindex.pageindex.toc.helpers import (
    convert_physical_index_to_int,
    remove_page_number,
)
from rag_pageindex.pageindex.toc.parsing import (
    add_page_number_to_toc,
    add_page_number_to_toc_async,
    toc_index_extractor,
    toc_transformer,
)


def _generate_toc_init(part: str, *, llm: LLMClient) -> list[dict[str, Any]]:
    """Generate initial TOC structure from document text.

    Args:
        part: Document text to extract TOC from.
        llm: LLM client for extraction.

    Returns:
        List of TOC items (dicts with structure, title, physical_index).
    """
    prompt = render("generate_toc_init.j2", part=part)
    result = llm.complete_structured([{"role": "user", "content": prompt}], TocGeneratedResponse)
    return [e.model_dump() for e in result.items]


def _generate_toc_continue(
    toc_content: list[dict[str, Any]],
    part: str,
    *,
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """Continue generating TOC structure given previous partial results.

    Args:
        toc_content: Previously generated TOC items (for context).
        part: Next document text chunk.
        llm: LLM client for extraction.

    Returns:
        List of additional TOC items.
    """
    prompt = render(
        "generate_toc_continue.j2",
        part=part,
        toc_content=_json.dumps(toc_content, indent=2),
    )
    result = llm.complete_structured([{"role": "user", "content": prompt}], TocGeneratedResponse)
    return [e.model_dump() for e in result.items]


def page_list_to_group_text(
    pages: list[Page],
    *,
    max_tokens: int = 20_000,
    overlap_page: int = 1,
    start_index: int = 1,
) -> list[str]:
    """Divide pages into text chunks, each within token limit, with overlap.

    Groups pages to balance load, maintaining a small overlap window for
    context between groups. Useful for TOC generation on large documents.

    Args:
        pages: List of PDF pages with token counts.
        max_tokens: Target token limit per group.
        overlap_page: Number of pages to overlap between groups.
        start_index: Starting page number for physical_index tags (1-indexed).

    Returns:
        List of text strings, each with physical_index markers.
    """
    page_contents = [
        (f"<physical_index_{start_index + i}>\n{p.text}\n<physical_index_{start_index + i}>\n\n")
        for i, p in enumerate(pages)
    ]
    token_lengths = [p.token_length for p in pages]
    num_tokens = sum(token_lengths)

    if num_tokens <= max_tokens:
        return ["".join(page_contents)]

    subsets: list[str] = []
    current_subset: list[str] = []
    current_token_count = 0
    expected_parts = math.ceil(num_tokens / max_tokens)
    average_tokens = math.ceil(((num_tokens / expected_parts) + max_tokens) / 2)

    for i, (page_content, page_tokens) in enumerate(zip(page_contents, token_lengths, strict=True)):
        if current_token_count + page_tokens > average_tokens:
            subsets.append("".join(current_subset))
            overlap_start = max(i - overlap_page, 0)
            current_subset = page_contents[overlap_start:i]
            current_token_count = sum(token_lengths[overlap_start:i])
        current_subset.append(page_content)
        current_token_count += page_tokens

    if current_subset:
        subsets.append("".join(current_subset))

    logger.debug("page_list_to_group_text: {} groups", len(subsets))
    return subsets


def extract_matching_page_pairs(
    toc_page: list[dict[str, Any]],
    toc_physical_index: list[dict[str, Any]],
    start_page_index: int,
) -> list[dict[str, Any]]:
    """Match TOC items from two sources by title to get (page, physical_index) pairs.

    Helps calibrate the offset between page numbers in TOC and physical page indices.

    Args:
        toc_page: TOC items extracted from TOC pages.
        toc_physical_index: TOC items extracted from page content.
        start_page_index: Minimum acceptable physical index.

    Returns:
        List of matched items with title, page, and physical_index.
    """
    pairs = []
    for phy_item in toc_physical_index:
        for page_item in toc_page:
            if phy_item.get("title") == page_item.get("title"):
                physical_index = phy_item.get("physical_index")
                if physical_index is not None and int(physical_index) >= start_page_index:
                    pairs.append(
                        {
                            "title": phy_item.get("title"),
                            "page": page_item.get("page"),
                            "physical_index": physical_index,
                        }
                    )
    return pairs


def calculate_page_offset(pairs: list[dict[str, Any]]) -> int | None:
    """Calculate the most common page number offset from matched pairs.

    Returns the mode of (physical_index - page) differences, useful for
    converting between page numbers in TOC and physical document indices.

    Args:
        pairs: List of matched {title, page, physical_index} dicts.

    Returns:
        Most common offset, or None if no valid pairs.
    """
    differences: list[int] = []
    for pair in pairs:
        try:
            differences.append(pair["physical_index"] - pair["page"])
        except (KeyError, TypeError):
            continue
    if not differences:
        return None
    counts: dict[int, int] = {}
    for d in differences:
        counts[d] = counts.get(d, 0) + 1
    return max(counts, key=lambda k: counts[k])


def _evenly_spaced_sample(items: list[dict[str, Any]], *, n: int) -> list[dict[str, Any]]:
    """Pick up to `n` evenly-spaced items by index. Returns all items if len(items) <= n."""
    if len(items) <= n:
        return list(items)
    step = (len(items) - 1) / (n - 1)
    indices = sorted({int(round(i * step)) for i in range(n)})
    return [items[i] for i in indices]


def _add_page_offset_to_toc_json(data: list[dict[str, Any]], offset: int) -> list[dict[str, Any]]:
    for item in data:
        if item.get("page") is not None and isinstance(item["page"], int):
            item["physical_index"] = item["page"] + offset
            del item["page"]
    return data


async def process_none_page_numbers(
    toc_items: list[dict[str, Any]],
    pages: list[Page],
    *,
    start_index: int = 1,
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """For TOC items missing a physical_index, ask the LLM to find each in parallel.

    Each item is searched within a window bounded by its already-known
    neighbours (using values present at call time only — we don't propagate
    freshly-resolved indices mid-pass). This loses the sequential refinement
    of the previous implementation but unlocks O(N) → O(1) wall-clock.
    """
    pending_indices = [i for i, item in enumerate(toc_items) if "physical_index" not in item]
    if not pending_indices:
        return toc_items

    async def _resolve_one(i: int) -> tuple[int, int | None]:
        prev_idx = 0
        for j in range(i - 1, -1, -1):
            if toc_items[j].get("physical_index") is not None:
                prev_idx = toc_items[j]["physical_index"]
                break

        next_idx = len(pages) + start_index - 1
        for j in range(i + 1, len(toc_items)):
            if toc_items[j].get("physical_index") is not None:
                next_idx = toc_items[j]["physical_index"]
                break

        page_contents = []
        for page_idx in range(prev_idx, next_idx + 1):
            list_i = page_idx - start_index
            if 0 <= list_i < len(pages):
                page_contents.append(
                    f"<physical_index_{page_idx}>\n{pages[list_i].text}\n<physical_index_{page_idx}>\n\n"
                )

        item_copy = copy.deepcopy(toc_items[i])
        item_copy.pop("page", None)
        result = await add_page_number_to_toc_async("".join(page_contents), [item_copy], llm=llm)
        if result:
            raw_pi = result[0].get("physical_index")
            if isinstance(raw_pi, str) and raw_pi.startswith("<physical_index"):
                return i, int(raw_pi.split("_")[-1].rstrip(">").strip())
        return i, None

    tasks = [_resolve_one(i) for i in pending_indices]
    resolutions = await asyncio.gather(*tasks, return_exceptions=True)
    for res in resolutions:
        if isinstance(res, BaseException):
            logger.error("process_none_page_numbers error: {}", res)
            continue
        i, physical_index = res
        if physical_index is not None:
            toc_items[i]["physical_index"] = physical_index
            toc_items[i].pop("page", None)

    return toc_items


@observe(name="process_toc_with_page_numbers")
async def process_toc_with_page_numbers(
    toc_content: str,
    toc_page_list: list[int],
    pages: list[Page],
    *,
    toc_check_page_num: int,
    llm: LLMClient,
    start_index: int = 1,
    toc_max_output_tokens: int | None = None,
    toc_transformed: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build TOC with physical indices when the TOC already has page numbers."""
    if toc_transformed is None:
        toc_with_page_number = toc_transformer(toc_content, llm=llm, max_tokens=toc_max_output_tokens)
    else:
        toc_with_page_number = copy.deepcopy(toc_transformed)
    logger.info("toc_transformer: {}", toc_with_page_number)

    toc_no_page_number = remove_page_number(copy.deepcopy(toc_with_page_number))

    start_page_index = toc_page_list[-1] + 1
    main_content_parts = []
    for pi in range(
        start_page_index,
        min(start_page_index + toc_check_page_num, len(pages)),
    ):
        main_content_parts.append(
            f"<physical_index_{pi + 1}>\n{pages[pi].text}\n<physical_index_{pi + 1}>\n\n"
        )
    main_content = "".join(main_content_parts)

    toc_no_page_number_list: list[dict[str, Any]] = toc_no_page_number  # type: ignore[assignment]
    sample = _evenly_spaced_sample(toc_no_page_number_list, n=8)
    logger.info("offset sampling {}/{} entries", len(sample), len(toc_no_page_number_list))

    toc_with_pi = toc_index_extractor(sample, main_content, llm=llm)
    logger.info("toc_with_physical_index: {}", toc_with_pi)

    toc_with_pi = convert_physical_index_to_int(toc_with_pi)

    matching_pairs = extract_matching_page_pairs(toc_with_page_number, toc_with_pi, start_page_index)
    logger.info("matching_pairs: {}", matching_pairs)

    offset = calculate_page_offset(matching_pairs) if len(matching_pairs) >= 3 else None
    logger.info("offset: {}", offset)

    if offset is not None:
        toc_with_page_number = _add_page_offset_to_toc_json(toc_with_page_number, offset)

    toc_with_page_number = await process_none_page_numbers(
        toc_with_page_number, pages, start_index=start_index, llm=llm
    )
    logger.info("process_none_page_numbers done")

    return toc_with_page_number


@observe(name="process_toc_no_page_numbers")
def process_toc_no_page_numbers(
    toc_content: str,
    pages: list[Page],
    *,
    start_index: int = 1,
    llm: LLMClient,
    max_tokens: int = 20_000,
    toc_max_output_tokens: int | None = None,
    toc_transformed: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build TOC with physical indices when the TOC has no page numbers."""
    if toc_transformed is None:
        toc_content_list = toc_transformer(toc_content, llm=llm, max_tokens=toc_max_output_tokens)
    else:
        toc_content_list = copy.deepcopy(toc_transformed)
    logger.info("toc_transformer: {}", toc_content_list)

    groups = page_list_to_group_text(pages, max_tokens=max_tokens, start_index=start_index)
    logger.info("process_toc_no_page_numbers: {} groups", len(groups))

    toc_with_page_number = copy.deepcopy(toc_content_list)
    for group_text in groups:
        toc_with_page_number = add_page_number_to_toc(group_text, toc_with_page_number, llm=llm)
    logger.info("add_page_number_to_toc done")

    return convert_physical_index_to_int(toc_with_page_number)


@observe(name="process_no_toc")
def process_no_toc(
    pages: list[Page],
    *,
    start_index: int = 1,
    llm: LLMClient,
    max_tokens: int = 20_000,
) -> list[dict[str, Any]]:
    """Build TOC from scratch (no TOC in the document)."""
    groups = page_list_to_group_text(pages, max_tokens=max_tokens, start_index=start_index)
    logger.info("process_no_toc: {} groups", len(groups))

    toc: list[dict[str, Any]] = _generate_toc_init(groups[0], llm=llm)
    for group_text in groups[1:]:
        additional = _generate_toc_continue(toc, group_text, llm=llm)
        toc.extend(additional)
    logger.info("generate_toc done: {} items", len(toc))

    return convert_physical_index_to_int(toc)
