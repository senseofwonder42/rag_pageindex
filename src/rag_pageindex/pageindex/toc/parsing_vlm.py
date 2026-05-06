from __future__ import annotations

from typing import Any

from loguru import logger

from rag_pageindex.pageindex.llm.protocol import ContentPart, LLMClient
from rag_pageindex.pageindex.observability import observe
from rag_pageindex.pageindex.pdf.reader import PdfSource
from rag_pageindex.pageindex.pdf.renderer import image_part, render_pages
from rag_pageindex.pageindex.prompts import render
from rag_pageindex.pageindex.structured_responses import TocGeneratedResponse
from rag_pageindex.pageindex.toc.helpers import convert_physical_index_to_int


@observe(name="regenerate_path_c_range_vlm")
async def regenerate_path_c_range_vlm(
    source: PdfSource,
    start_page: int,
    end_page: int,
    *,
    structure_offset: str = "1",
    dpi: int = 144,
    max_images_per_call: int = 10,
    max_output_tokens: int | None = None,
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """Re-extract TOC for pages [start_page, end_page] using VLM images.

    Returns a flat list in TocItem format (structure, title, physical_index as int).
    Called when text-based Path C verification falls below threshold.

    The page range is split into chunks of at most ``max_images_per_call``
    images so we stay under per-prompt image caps imposed by some providers
    (e.g. OpenRouter → Nvidia: 10 images max per prompt).
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

    chunk_size = max(1, max_images_per_call)
    items: list[dict[str, Any]] = []
    for chunk_start in range(0, len(page_indices), chunk_size):
        chunk = page_indices[chunk_start : chunk_start + chunk_size]
        prompt_text = render(
            "generate_toc_vlm_range.j2",
            start_page=chunk[0],
            num_images=len(chunk),
            structure_offset=structure_offset,
        )
        parts: list[ContentPart] = [{"type": "text", "text": prompt_text}]
        for idx in chunk:
            if idx in rendered:
                parts.append(image_part(rendered[idx]))

        result = await llm.acomplete_structured(
            [{"role": "user", "content": parts}],
            TocGeneratedResponse,
            max_tokens=max_output_tokens,
        )
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
