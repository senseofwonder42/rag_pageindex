from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from loguru import logger

from rag_pageindex.agent.vlm import answer_with_images
from rag_pageindex.core.config import settings
from rag_pageindex.pageindex.llm.factory import get_default_client
from rag_pageindex.pageindex.llm.protocol import LLMClient
from rag_pageindex.pageindex.pdf.renderer import render_pages

_llm: LLMClient | None = None


def _get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = get_default_client(settings)
    return _llm


def _max_end_index(structure: list[dict[str, Any]]) -> int:
    best = 0
    for node in structure:
        end = node.get("end_index") or 0
        if end > best:
            best = end
        children = node.get("nodes") or []
        if children:
            sub = _max_end_index(children)
            if sub > best:
                best = sub
    return best


def _strip_text(structure: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in structure:
        clean = {k: v for k, v in node.items() if k != "text"}
        children = clean.get("nodes")
        if children:
            clean["nodes"] = _strip_text(children)
        out.append(clean)
    return out


def _iter_index_files() -> list[tuple[Path, dict[str, Any]]]:
    """Return (path, parsed) for every JSON in results dir that looks like an IndexResult."""
    items: list[tuple[Path, dict[str, Any]]] = []
    results_dir = settings.pageindex_results_dir
    if not results_dir.exists():
        return items
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("skipping {}: {}", path.name, exc)
            continue
        if isinstance(data, dict) and "doc_name" in data and "structure" in data:
            items.append((path, data))
    return items


def _load_doc(doc_name: str) -> dict[str, Any]:
    for _path, data in _iter_index_files():
        if data["doc_name"] == doc_name:
            return data
    stem = Path(doc_name).stem.lower()
    for _path, data in _iter_index_files():
        if Path(data["doc_name"]).stem.lower() == stem:
            return data
    raise FileNotFoundError(
        f"No structure JSON found for doc_name={doc_name!r} in {settings.pageindex_results_dir}"
    )


def _resolve_pdf(doc_name: str) -> Path:
    results_dir = settings.pageindex_results_dir
    name = doc_name if doc_name.lower().endswith(".pdf") else f"{doc_name}.pdf"
    return results_dir / name


def _list_documents_sync() -> list[dict[str, Any]]:
    return [
        {
            "doc_name": data["doc_name"],
            "doc_description": data.get("doc_description"),
            "num_pages": _max_end_index(data["structure"]),
        }
        for _path, data in _iter_index_files()
    ]


def _get_document_structure_sync(doc_name: str) -> dict[str, Any]:
    data = _load_doc(doc_name)
    return {
        "doc_name": data["doc_name"],
        "doc_description": data.get("doc_description"),
        "structure": _strip_text(data["structure"]),
    }


def _prepare_pages_sync(
    doc_name: str, page_indices: list[int]
) -> tuple[Path | None, list[int], dict[int, bytes]]:
    pdf_path = _resolve_pdf(doc_name)
    if not pdf_path.exists():
        return pdf_path, [], {}

    seen: dict[int, None] = {}
    for idx in page_indices:
        if idx >= 1:
            seen.setdefault(idx, None)
    ordered = list(seen.keys())
    if not ordered:
        return pdf_path, [], {}

    rendered = render_pages(pdf_path, ordered, dpi=settings.pageindex_vision_dpi)
    return pdf_path, ordered, rendered


@tool
async def list_documents() -> list[dict[str, Any]]:
    """List indexed documents available to query.

    Returns one entry per document with its name, optional description, and
    total page count (max end_index across the tree). Call this first when
    the user has not specified which document to query.
    """
    return await asyncio.to_thread(_list_documents_sync)


@tool
async def get_document_structure(doc_name: str) -> dict[str, Any]:
    """Return the table-of-contents tree for one document.

    Use the returned tree (titles + start_index/end_index + optional summaries)
    to decide which pages to fetch with `answer_from_pages`. The `text` field
    on nodes is stripped to keep this response small.

    Args:
        doc_name: Document name as returned by `list_documents`
            (e.g. 'earthmover.pdf').
    """
    return await asyncio.to_thread(_get_document_structure_sync, doc_name)


@tool
async def answer_from_pages(
    doc_name: str,
    page_indices: list[int],
    question: str,
) -> str:
    """Render the listed PDF pages and ask the vision model to answer `question`.

    Args:
        doc_name: Document name (e.g. 'earthmover.pdf').
        page_indices: 1-based page numbers from the document tree. Order is
            preserved; duplicates are removed.
        question: The question to answer using only those pages as context.

    Returns:
        A grounded answer string, or a clear error message if the source PDF
        is missing.
    """
    pdf_path, ordered, rendered = await asyncio.to_thread(
        _prepare_pages_sync, doc_name, page_indices
    )
    if not ordered:
        if pdf_path is not None and not pdf_path.exists():
            return (
                f"Source PDF not found at {pdf_path}. The structure JSON "
                "exists but the PDF must sit alongside it for page rendering."
            )
        return "No valid 1-based page indices were provided."

    images = [rendered[i] for i in ordered]
    return await answer_with_images(_get_llm(), question=question, images_png=images)


TOOLS = [list_documents, get_document_structure, answer_from_pages]
