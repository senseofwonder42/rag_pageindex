from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from rag_pageindex.pageindex.llm.protocol import LLMClient, Message
from rag_pageindex.pageindex.observability import observe
from rag_pageindex.pageindex.prompts import render
from rag_pageindex.pageindex.structured_responses import (
    CompletionCheckResponse,
    PageIndexInTocResponse,
    TocDetectedResponse,
)

if TYPE_CHECKING:
    from rag_pageindex.pageindex.pdf import Page


@dataclass(frozen=True, slots=True)
class TocDetection:
    """Outcome of scanning the front of a document for a TOC."""

    toc_content: str | None
    toc_page_list: list[int]
    page_index_given_in_toc: str  # "yes" | "no"


async def toc_detector_single_page(content: str, *, llm: LLMClient) -> str:
    """Check if a single page contains a table of contents."""
    prompt = render("check_toc.j2", content=content)
    result = await llm.acomplete_structured([{"role": "user", "content": prompt}], TocDetectedResponse)
    return result.toc_detected


async def find_toc_pages(
    pages: list[Page],
    *,
    start_page_index: int,
    toc_check_page_num: int,
    llm: LLMClient,
) -> list[int]:
    """Find contiguous pages containing a table of contents.

    Fans out detection across [start_page_index, toc_check_page_num) in
    parallel, then walks the boolean array to extract the contiguous run
    of TOC pages (first 'yes' through last 'yes' before a 'no').
    """
    end = min(toc_check_page_num, len(pages))
    if start_page_index >= end:
        logger.info("No TOC found")
        return []

    tasks = [toc_detector_single_page(pages[i].text, llm=llm) for i in range(start_page_index, end)]
    results = await asyncio.gather(*tasks)

    toc_page_list: list[int] = []
    started = False
    for offset, verdict in enumerate(results):
        i = start_page_index + offset
        if verdict == "yes":
            toc_page_list.append(i)
            started = True
            logger.info("Page {} contains TOC", i)
        elif started:
            logger.info("Last TOC page detected: {}", i - 1)
            break

    if not toc_page_list:
        logger.info("No TOC found")
    return toc_page_list


def _transform_dots_to_colon(text: str) -> str:
    """Normalize TOC dot leaders to colon separators."""
    text = re.sub(r"\.{5,}", ": ", text)
    text = re.sub(r"(?:\. ){5,}\.?", ": ", text)
    return text


async def detect_page_index(toc_content: str, *, llm: LLMClient) -> str:
    """Check if TOC contains explicit page numbers."""
    logger.debug("detect_page_index")
    prompt = render("detect_page_index.j2", toc_content=toc_content)
    result = await llm.acomplete_structured(
        [{"role": "user", "content": prompt}], PageIndexInTocResponse
    )
    return result.page_index_given_in_toc


async def extract_toc_from_pages(
    pages: list[Page],
    toc_page_list: list[int],
    *,
    llm: LLMClient,
) -> dict[str, str]:
    """Concatenate TOC pages, normalize formatting, and detect page numbers."""
    toc_content = "".join(pages[i].text for i in toc_page_list)
    toc_content = _transform_dots_to_colon(toc_content)
    page_index_given_in_toc = await detect_page_index(toc_content, llm=llm)
    return {
        "toc_content": toc_content,
        "page_index_given_in_toc": page_index_given_in_toc,
    }


def check_if_toc_extraction_is_complete(
    content: str,
    toc: str,
    *,
    llm: LLMClient,
) -> str:
    """Check if extracted TOC content is complete."""
    prompt = render(
        "check_toc_extraction_complete.j2",
        content=content,
        toc=toc,
    )
    result = llm.complete_structured([{"role": "user", "content": prompt}], CompletionCheckResponse)
    return result.completed


def check_if_toc_transformation_is_complete(
    raw_toc: str,
    cleaned_toc: str,
    *,
    llm: LLMClient,
) -> str:
    """Check if TOC transformation/cleaning is complete."""
    prompt = render(
        "check_toc_transformation_complete.j2",
        raw_toc=raw_toc,
        cleaned_toc=cleaned_toc,
    )
    result = llm.complete_structured([{"role": "user", "content": prompt}], CompletionCheckResponse)
    return result.completed


@observe(name="check_toc")
async def check_toc(
    pages: list[Page],
    *,
    toc_check_page_num: int,
    llm: LLMClient,
) -> TocDetection:
    """Scan document for a TOC and extract its content."""
    toc_page_list = await find_toc_pages(
        pages,
        start_page_index=0,
        toc_check_page_num=toc_check_page_num,
        llm=llm,
    )
    if not toc_page_list:
        logger.info("no toc found")
        return TocDetection(toc_content=None, toc_page_list=[], page_index_given_in_toc="no")

    logger.info("toc found")
    toc = await extract_toc_from_pages(pages, toc_page_list, llm=llm)
    if toc["page_index_given_in_toc"] == "yes":
        logger.info("page index found in toc")
        return TocDetection(
            toc_content=toc["toc_content"],
            toc_page_list=toc_page_list,
            page_index_given_in_toc="yes",
        )

    current_start_index = toc_page_list[-1] + 1
    while (
        toc["page_index_given_in_toc"] == "no"
        and current_start_index < len(pages)
        and current_start_index < toc_check_page_num
    ):
        additional = await find_toc_pages(
            pages,
            start_page_index=current_start_index,
            toc_check_page_num=toc_check_page_num,
            llm=llm,
        )
        if not additional:
            break
        additional_toc = await extract_toc_from_pages(pages, additional, llm=llm)
        if additional_toc["page_index_given_in_toc"] == "yes":
            logger.info("page index found in extended toc")
            return TocDetection(
                toc_content=additional_toc["toc_content"],
                toc_page_list=additional,
                page_index_given_in_toc="yes",
            )
        current_start_index = additional[-1] + 1

    logger.info("page index not found in toc")
    return TocDetection(
        toc_content=toc["toc_content"],
        toc_page_list=toc_page_list,
        page_index_given_in_toc="no",
    )


@observe(name="extract_toc_content")
def extract_toc_content(content: str, *, llm: LLMClient) -> str:
    """Extract a TOC from document text via multi-turn continuation."""
    prompt = render("extract_toc_content.j2", content=content)
    response = llm.complete([{"role": "user", "content": prompt}])
    text = response.content
    finish_reason = response.finish_reason
    continue_prompt = render("extract_toc_continue.j2")
    last_user_msg = prompt

    for _ in range(6):
        if finish_reason == "finished" and (
            check_if_toc_transformation_is_complete(content, text, llm=llm) == "yes"
        ):
            return text
        msgs: list[Message] = [
            {"role": "user", "content": last_user_msg},
            {"role": "assistant", "content": text},
            {"role": "user", "content": continue_prompt},
        ]
        follow = llm.complete(msgs)
        text += follow.content
        finish_reason = follow.finish_reason
        last_user_msg = continue_prompt

    raise RuntimeError("Failed to complete table of contents after maximum retries")
