from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from rag_pageindex.pageindex.llm.protocol import LLMClient
from rag_pageindex.pageindex.observability import observe
from rag_pageindex.pageindex.pdf.reader import (
    PdfSource,
    get_pdf_name,
    read_pages,
)
from rag_pageindex.pageindex.tree.builder import add_node_text, write_node_id
from rag_pageindex.pageindex.tree.summaries import (
    generate_doc_description,
    generate_parent_summaries,
    remove_structure_text,
)
from rag_pageindex.pageindex.vlm.extract import extract_tree, strip_internal_fields
from rag_pageindex.pageindex.vlm.summaries import roll_up_leaf_summaries

if TYPE_CHECKING:
    from rag_pageindex.core.config import Settings

_KEY_ORDER = [
    "title",
    "node_id",
    "start_index",
    "end_index",
    "summary",
    "text",
    "nodes",
]

_Tree = dict[str, Any] | list[Any]


def _format_structure(structure: _Tree, *, order: list[str]) -> _Tree:
    """Reorder dict keys recursively and remove empty node lists."""
    if isinstance(structure, dict):
        if "nodes" in structure:
            structure["nodes"] = _format_structure(structure["nodes"], order=order)
        if not structure.get("nodes"):
            structure.pop("nodes", None)
        return {k: structure[k] for k in order if k in structure}
    if isinstance(structure, list):
        return [_format_structure(item, order=order) for item in structure]
    return structure


@observe(name="page_index_builder")
async def apage_index(
    source: PdfSource,
    *,
    llm: LLMClient,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Build the hierarchical page index for a PDF using the VLM pipeline.

    Each page is rendered to an image, sent in batches to a vision model
    that returns the headings starting on the page (with visual rank) plus
    a 1-2 sentence content description. The tree is then assembled from
    those records, leaf summaries are rolled up from the per-page
    descriptions, and parent summaries are reduced bottom-up.

    Returns a dict shaped like `IndexResult`:
        {doc_name, structure, doc_description?}
    """
    if settings is None:
        from rag_pageindex.core.config import settings as _default_settings

        settings = _default_settings

    structure = await extract_tree(source, llm=llm, settings=settings)

    if settings.pageindex_add_node_id:
        write_node_id(structure)

    if settings.pageindex_add_node_summary:
        await roll_up_leaf_summaries(structure, llm=llm)
        await generate_parent_summaries(structure, llm=llm)

    if settings.pageindex_add_node_text:
        pages = read_pages(source, llm=llm, parser="PyMuPDF")
        add_node_text(structure, pages)

    strip_internal_fields(structure)

    result: dict[str, Any] = {
        "doc_name": get_pdf_name(source),
        "structure": _format_structure(structure, order=_KEY_ORDER),
    }

    if settings.pageindex_add_doc_description and settings.pageindex_add_node_summary:
        result["doc_description"] = generate_doc_description(structure, llm=llm)

    if not settings.pageindex_add_node_text:
        remove_structure_text(result["structure"])  # defensive

    return result


def page_index(
    source: PdfSource,
    *,
    llm: LLMClient,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper around `apage_index`."""
    return asyncio.run(apage_index(source, llm=llm, settings=settings))
