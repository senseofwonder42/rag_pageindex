from __future__ import annotations

from typing import Any

from loguru import logger

from rag_pageindex.pageindex.llm.protocol import ContentPart, LLMClient
from rag_pageindex.pageindex.pdf.reader import PdfSource
from rag_pageindex.pageindex.pdf.renderer import image_part, render_pages
from rag_pageindex.pageindex.prompts import render
from rag_pageindex.pageindex.structured_responses import TocGeneratedResponse
from rag_pageindex.pageindex.toc.helpers import convert_physical_index_to_int


async def regenerate_path_c_range_vlm(
    source: PdfSource,
    start_page: int,
    end_page: int,
    *,
    structure_offset: str = "1",
    dpi: int = 144,
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """Re-extract TOC for pages [start_page, end_page] using VLM images.

    Returns a flat list in TocItem format (structure, title, physical_index as int).
    Called when text-based Path C verification falls below threshold.
    """
    if start_page > end_page:
        return []

    page_indices = list(range(start_page, end_page + 1))
    logger.info(
        "regenerate_path_c_range_vlm: rendering {} pages ({}-{})",
        len(page_indices),
        start_page,
        end_page,
    )

    rendered = render_pages(source, page_indices, dpi=dpi)

    prompt_text = render(
        "generate_toc_vlm_range.j2",
        start_page=start_page,
        num_images=len(page_indices),
        structure_offset=structure_offset,
    )
    parts: list[ContentPart] = [{"type": "text", "text": prompt_text}]
    for idx in page_indices:
        if idx in rendered:
            parts.append(image_part(rendered[idx]))

    result = await llm.acomplete_structured([{"role": "user", "content": parts}], TocGeneratedResponse)

    items: list[dict[str, Any]] = []
    for entry in result.items:
        physical_index = convert_physical_index_to_int(entry.physical_index)
        if physical_index is not None:
            items.append(
                {
                    "structure": entry.structure,
                    "title": entry.title,
                    "physical_index": physical_index,
                }
            )

    logger.info("regenerate_path_c_range_vlm: extracted {} items", len(items))
    return items
