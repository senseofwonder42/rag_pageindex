from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import PyPDF2

_Tree = dict[str, Any] | list[Any]


def _parse_pages(pages: str) -> list[int]:
    """Parse '5-7', '3,8', or '12' into a sorted unique list of ints."""
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
    if isinstance(node, dict):
        return {k: _remove_text_fields(v) for k, v in node.items() if k != "text"}
    if isinstance(node, list):
        return [_remove_text_fields(item) for item in node]
    return node


def get_document(documents: dict[str, Any], doc_id: str) -> str:
    """Return JSON with document metadata."""
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
    """Return tree structure JSON with text fields stripped."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"})
    return json.dumps(
        _remove_text_fields(doc_info.get("structure", [])),
        ensure_ascii=False,
    )


def get_page_content(documents: dict[str, Any], doc_id: str, pages: str) -> str:
    """Return page content JSON for the given pages string."""
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
