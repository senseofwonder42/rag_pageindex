from __future__ import annotations

import base64
from io import BytesIO
from typing import Iterable

import pymupdf

from rag_pageindex.pageindex.llm.protocol import ImageUrl, ImageUrlPart
from rag_pageindex.pageindex.pdf.reader import PdfSource

_MAX_LONG_EDGE_PX = 1568


def _open_doc(source: PdfSource) -> pymupdf.Document:
    if isinstance(source, BytesIO):
        return pymupdf.open(stream=source, filetype="pdf")
    return pymupdf.open(str(source))


def _render_single(doc: pymupdf.Document, page_index: int, dpi: int) -> bytes:
    """Render a 1-based page index to PNG bytes, capping the long edge."""
    page = doc[page_index - 1]
    rect = page.rect
    long_edge_pts = max(rect.width, rect.height)
    # Points → pixels at given dpi, then cap to _MAX_LONG_EDGE_PX
    long_edge_px = long_edge_pts * dpi / 72.0
    scale = min(1.0, _MAX_LONG_EDGE_PX / long_edge_px) if long_edge_px > 0 else 1.0
    effective = dpi * scale
    mat = pymupdf.Matrix(effective / 72, effective / 72)
    pix = page.get_pixmap(matrix=mat)
    return bytes(pix.tobytes("png"))


def render_page(source: PdfSource, page_index: int, *, dpi: int = 144) -> bytes:
    """Render a single 1-based page to PNG bytes."""
    doc = _open_doc(source)
    return _render_single(doc, page_index, dpi)


def render_pages(
    source: PdfSource,
    page_indices: Iterable[int],
    *,
    dpi: int = 144,
) -> dict[int, bytes]:
    """Render multiple 1-based page indices to PNG bytes, keyed by page index."""
    doc = _open_doc(source)
    return {idx: _render_single(doc, idx, dpi) for idx in page_indices}


def image_part(png_bytes: bytes) -> ImageUrlPart:
    """Wrap PNG bytes as an OpenAI-compatible image_url content part."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    url: ImageUrl = {"url": f"data:image/png;base64,{b64}"}
    return {"type": "image_url", "image_url": url}
