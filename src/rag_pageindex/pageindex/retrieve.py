from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import PyPDF2

_Tree = dict[str, Any] | list[Any]


def _parse_pages(pages: str) -> list[int]:
    """Parse page specification into a sorted list of unique page numbers.

    Supports individual pages ('3'), ranges ('5-7'), and comma-separated
    combinations ('3,8,12'). Deduplicates and sorts the result.

    Args:
        pages: Page specification string.

    Returns:
        Sorted list of unique page numbers (1-indexed).

    Raises:
        ValueError: If range start > end or format is invalid.
    """
    result: list[int] = []
    for part in pages.split(","):
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s.strip()), int(end_s.strip())
            if start > end:
                raise ValueError(f"Invalid range '{part}': start must be <= end")
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def _get_pdf_page_content(doc_info: dict[str, Any], page_nums: list[int]) -> list[dict[str, Any]]:
    """Extract text content for specific pages from a PDF.

    Uses cached page content if available; otherwise reads the PDF file
    from disk. Returns only pages that exist in the document.

    Args:
        doc_info: Document metadata dict with 'path' and optional 'pages'.
        page_nums: List of page numbers to extract (1-indexed).

    Returns:
        List of {page, content} dicts for requested pages.
    """
    cached = doc_info.get("pages")
    if cached:
        page_map = {p["page"]: p["content"] for p in cached}
        return [{"page": p, "content": page_map[p]} for p in page_nums if p in page_map]
    path = Path(doc_info["path"])
    with path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        total = len(reader.pages)
        return [
            {"page": p, "content": reader.pages[p - 1].extract_text() or ""}
            for p in page_nums
            if 1 <= p <= total
        ]


def _remove_text_fields(node: _Tree) -> _Tree:
    """Recursively remove 'text' fields from tree nodes.

    Args:
        node: Tree dict, list, or scalar value.

    Returns:
        Tree with all 'text' keys removed at all levels.
    """
    if isinstance(node, dict):
        return {k: _remove_text_fields(v) for k, v in node.items() if k != "text"}
    if isinstance(node, list):
        return [_remove_text_fields(item) for item in node]
    return node


def get_document(documents: dict[str, Any], doc_id: str) -> str:
    """Get document metadata as JSON.

    Args:
        documents: Documents dict (from PageIndexClient).
        doc_id: Document ID.

    Returns:
        JSON string with doc_id, doc_name, doc_description, type, status, page_count.
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"})
    return json.dumps(
        {
            "doc_id": doc_id,
            "doc_name": doc_info.get("doc_name", ""),
            "doc_description": doc_info.get("doc_description", ""),
            "type": doc_info.get("type", ""),
            "status": "completed",
            "page_count": doc_info.get("page_count", 0),
        }
    )


def get_document_structure(documents: dict[str, Any], doc_id: str) -> str:
    """Get document tree structure as JSON (text content removed).

    Args:
        documents: Documents dict (from PageIndexClient).
        doc_id: Document ID.

    Returns:
        JSON string with hierarchical tree structure (no text fields).
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"})
    return json.dumps(
        _remove_text_fields(doc_info.get("structure", [])),
        ensure_ascii=False,
    )


def get_page_content(documents: dict[str, Any], doc_id: str, pages: str) -> str:
    """Get content of specific pages as JSON.

    Args:
        documents: Documents dict (from PageIndexClient).
        doc_id: Document ID.
        pages: Page specification ('3', '5-7', '3,8,12', etc.).

    Returns:
        JSON string with list of {page, content} dicts; {error: message} on failure.
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"})
    try:
        page_nums = _parse_pages(pages)
    except (ValueError, AttributeError) as e:
        return json.dumps(
            {"error": (f"Invalid pages format: {pages!r}. Use '5-7', '3,8', or '12'. Error: {e}")}
        )
    try:
        content = _get_pdf_page_content(doc_info, page_nums)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"Failed to read page content: {e}"})
    return json.dumps(content, ensure_ascii=False)
