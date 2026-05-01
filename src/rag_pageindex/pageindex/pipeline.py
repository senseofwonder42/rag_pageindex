from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from rag_pageindex.pageindex.llm.protocol import LLMClient
from rag_pageindex.pageindex.pdf.reader import (
    PdfSource,
    get_pdf_name,
)
from rag_pageindex.pageindex.tree.builder import (
    add_node_text,
    tree_parser,
    write_node_id,
)
from rag_pageindex.pageindex.tree.summaries import (
    generate_doc_description,
    generate_summaries_for_structure,
    remove_structure_text,
)

if TYPE_CHECKING:
    from rag_pageindex.core.config import Settings

_KEY_ORDER = [
    "title", "node_id", "start_index", "end_index", "summary", "text", "nodes",
]

_Tree = dict[str, Any] | list[Any]


def _format_structure(structure: _Tree, *, order: list[str]) -> _Tree:
    """Reorder dict keys in every node; drop empty `nodes` lists."""
    if isinstance(structure, dict):
        if "nodes" in structure:
            structure["nodes"] = _format_structure(structure["nodes"], order=order)
        if not structure.get("nodes"):
            structure.pop("nodes", None)
        return {k: structure[k] for k in order if k in structure}
    if isinstance(structure, list):
        return [_format_structure(item, order=order) for item in structure]
    return structure


async def _build(
    source: PdfSource,
    *,
    llm: LLMClient,
    settings: Settings,
) -> dict[str, Any]:
    from rag_pageindex.pageindex.pdf.reader import read_pages

    pages = read_pages(source, llm=llm, parser="PyPDF2")
    logger.info(
        "page_index: {} pages, {} tokens total",
        len(pages),
        sum(p.token_length for p in pages),
    )

    structure = await tree_parser(pages, llm=llm, settings=settings)

    if settings.pageindex_add_node_id:
        write_node_id(structure)

    if settings.pageindex_add_node_summary:
        add_node_text(structure, pages)
        await generate_summaries_for_structure(structure, llm=llm)
        if not settings.pageindex_add_node_text:
            remove_structure_text(structure)
    elif settings.pageindex_add_node_text:
        add_node_text(structure, pages)

    result: dict[str, Any] = {
        "doc_name": get_pdf_name(source),
        "structure": _format_structure(structure, order=_KEY_ORDER),
    }

    if (
        settings.pageindex_add_doc_description
        and settings.pageindex_add_node_summary
    ):
        result["doc_description"] = generate_doc_description(
            structure, llm=llm
        )

    return result


def page_index(
    source: PdfSource,
    *,
    llm: LLMClient,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Index a PDF and return a structured tree dict.

    Synchronous entry point; runs the async pipeline internally.
    """
    if settings is None:
        from rag_pageindex.core.config import settings as _default_settings

        settings = _default_settings

    return asyncio.run(_build(source, llm=llm, settings=settings))
