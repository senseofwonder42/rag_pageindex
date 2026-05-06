from __future__ import annotations

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


def toc_detector_single_page(content: str, *, llm: LLMClient) -> str:
    """Check if a single page contains a table of contents.

    Args:
        content: Text content of the page.
        llm: LLM client for classification.

    Returns:
        'yes' if page contains TOC, 'no' otherwise.
    """
    prompt = render("check_toc.j2", content=content)
    result = llm.complete_structured([{"role": "user", "content": prompt}], TocDetectedResponse)
    return result.toc_detected


def find_toc_pages(
    pages: list[Page],
    *,
    start_page_index: int,
    toc_check_page_num: int,
    llm: LLMClient,
) -> list[int]:
    """Find contiguous pages containing a table of contents.

    Scans forward from start_page_index until toc_check_page_num, collecting
    pages marked as TOC. Stops at the first non-TOC page after TOC pages end.

    Args:
        pages: List of PDF pages.
        start_page_index: Page index to start scanning from.
        toc_check_page_num: Maximum page to scan up to.
        llm: LLM client for TOC detection.

    Returns:
        List of page indices containing the TOC (empty if none found).
    """
    last_page_was_toc = False
    toc_page_list: list[int] = []
    i = start_page_index

    while i < len(pages):
        if i >= toc_check_page_num and not last_page_was_toc:
            break
        result = toc_detector_single_page(pages[i].text, llm=llm)
        if result == "yes":
            logger.info("Page {} contains TOC", i)
            toc_page_list.append(i)
            last_page_was_toc = True
        elif result == "no" and last_page_was_toc:
            logger.info("Last TOC page detected: {}", i - 1)
            break
        i += 1

    if not toc_page_list:
        logger.info("No TOC found")
    return toc_page_list


def _transform_dots_to_colon(text: str) -> str:
    """Normalize TOC dot leaders to colon separators.

    Replaces multiple dots or repeated dot-space patterns with ': '
    to normalize TOC formatting from OCR or PDF extraction.

    Args:
        text: TOC text potentially containing dot leaders.

    Returns:
        Text with normalized separators.
    """
    text = re.sub(r"\.{5,}", ": ", text)
    text = re.sub(r"(?:\. ){5,}\.?", ": ", text)
    return text


def detect_page_index(toc_content: str, *, llm: LLMClient) -> str:
    """Check if TOC contains explicit page numbers.

    Args:
        toc_content: Extracted TOC text.
        llm: LLM client for classification.

    Returns:
        'yes' if page numbers are present, 'no' otherwise.
    """
    logger.debug("detect_page_index")
    prompt = render("detect_page_index.j2", toc_content=toc_content)
    result = llm.complete_structured([{"role": "user", "content": prompt}], PageIndexInTocResponse)
    return result.page_index_given_in_toc


def extract_toc_from_pages(
    pages: list[Page],
    toc_page_list: list[int],
    *,
    llm: LLMClient,
) -> dict[str, str]:
    """Concatenate TOC pages, normalize formatting, and detect page numbers.

    Args:
        pages: List of PDF pages.
        toc_page_list: Indices of pages containing the TOC.
        llm: LLM client for page number detection.

    Returns:
        Dict with 'toc_content' and 'page_index_given_in_toc' ('yes'/'no').
    """
    toc_content = "".join(pages[i].text for i in toc_page_list)
    toc_content = _transform_dots_to_colon(toc_content)
    page_index_given_in_toc = detect_page_index(toc_content, llm=llm)
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
    """Check if extracted TOC content is complete.

    Args:
        content: Original page content.
        toc: Extracted TOC text.
        llm: LLM client for verification.

    Returns:
        'yes' if extraction is complete, 'no' otherwise.
    """
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
    """Check if TOC transformation/cleaning is complete.

    Args:
        raw_toc: Original TOC text.
        cleaned_toc: Transformed TOC text.
        llm: LLM client for verification.

    Returns:
        'yes' if transformation is complete, 'no' otherwise.
    """
    prompt = render(
        "check_toc_transformation_complete.j2",
        raw_toc=raw_toc,
        cleaned_toc=cleaned_toc,
    )
    result = llm.complete_structured([{"role": "user", "content": prompt}], CompletionCheckResponse)
    return result.completed


@observe(name="check_toc")
def check_toc(
    pages: list[Page],
    *,
    toc_check_page_num: int,
    llm: LLMClient,
) -> TocDetection:
    """Scan document for a table of contents and extract its content.

    Iterates forward through pages looking for contiguous TOC pages. If
    the initial TOC has no page numbers, continues scanning to find an
    extended TOC with page numbers.

    Args:
        pages: List of PDF pages.
        toc_check_page_num: Maximum page to scan for TOC.
        llm: LLM client for detection and analysis.

    Returns:
        TocDetection with toc_content, toc_page_list, and page_index_given_in_toc.
    """
    toc_page_list = find_toc_pages(
        pages,
        start_page_index=0,
        toc_check_page_num=toc_check_page_num,
        llm=llm,
    )
    if not toc_page_list:
        logger.info("no toc found")
        return TocDetection(toc_content=None, toc_page_list=[], page_index_given_in_toc="no")

    logger.info("toc found")
    toc = extract_toc_from_pages(pages, toc_page_list, llm=llm)
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
        additional = find_toc_pages(
            pages,
            start_page_index=current_start_index,
            toc_check_page_num=toc_check_page_num,
            llm=llm,
        )
        if not additional:
            break
        additional_toc = extract_toc_from_pages(pages, additional, llm=llm)
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
    """Extract and complete a table of contents from document text.

    Uses multi-turn conversation with the LLM to extract the TOC verbatim
    and iteratively request continuation until the extraction is complete
    (max 5 continuation attempts).

    Args:
        content: Document text to extract TOC from.
        llm: LLM client for extraction.

    Returns:
        Complete extracted TOC text.

    Raises:
        RuntimeError: If completion fails after maximum retry attempts.
    """
    prompt = render("extract_toc_content.j2", content=content)
    response = llm.complete([{"role": "user", "content": prompt}])
    text = response.content
    finish_reason = response.finish_reason

    if (
        check_if_toc_transformation_is_complete(content, text, llm=llm) == "yes"
        and finish_reason == "finished"
    ):
        return text

    chat_history: list[dict[str, str]] = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": text},
    ]
    continue_prompt = render("extract_toc_continue.j2")
    msgs: list[Message] = [
        {"role": m["role"], "content": m["content"]}  # type: ignore[typeddict-item]
        for m in chat_history
    ]
    msgs.append({"role": "user", "content": continue_prompt})
    follow = llm.complete(msgs)
    text += follow.content
    finish_reason = follow.finish_reason

    attempt = 0
    while not (
        check_if_toc_transformation_is_complete(content, text, llm=llm) == "yes"
        and finish_reason == "finished"
    ):
        attempt += 1
        if attempt > 5:
            raise RuntimeError("Failed to complete table of contents after maximum retries")
        chat_history = [
            {"role": "user", "content": continue_prompt},
            {"role": "assistant", "content": text},
        ]
        msgs = [
            {"role": m["role"], "content": m["content"]}  # type: ignore[typeddict-item]
            for m in chat_history
        ]
        msgs.append({"role": "user", "content": continue_prompt})
        follow = llm.complete(msgs)
        text += follow.content
        finish_reason = follow.finish_reason

    return text
