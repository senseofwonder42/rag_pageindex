from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import pymupdf
import PyPDF2

from rag_pageindex.pageindex.llm.protocol import LLMClient

PdfParser = Literal["PyPDF2", "PyMuPDF"]
PdfSource = str | Path | BytesIO


@dataclass(frozen=True, slots=True)
class Page:
    """One page's extracted text and token count."""

    text: str
    token_length: int


def _sanitize_filename(filename: str, replacement: str = "-") -> str:
    """Replace forward slashes in filename with a safe replacement character.

    Args:
        filename: Original filename.
        replacement: Character to replace slashes with (default "-").

    Returns:
        Filename with slashes removed.
    """
    return filename.replace("/", replacement)


def get_pdf_name(source: PdfSource) -> str:
    """Best-effort filename for a PDF; reads metadata for BytesIO inputs."""
    if isinstance(source, (str, Path)):
        return Path(source).name
    pdf_reader = PyPDF2.PdfReader(source)
    meta = pdf_reader.metadata
    title = meta.title if meta and meta.title else "Untitled"
    return _sanitize_filename(title)


def read_pages(
    source: PdfSource,
    *,
    llm: LLMClient,
    parser: PdfParser = "PyPDF2",
) -> list[Page]:
    """Extract text + token count for every page."""
    if parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(source)
        return [
            Page(
                text=(text := page.extract_text() or ""),
                token_length=llm.count_tokens(text),
            )
            for page in pdf_reader.pages
        ]
    if parser == "PyMuPDF":
        if isinstance(source, BytesIO):
            doc = pymupdf.open(stream=source, filetype="pdf")
        else:
            doc = pymupdf.open(str(source))
        return [
            Page(
                text=(text := page.get_text()),  # type: ignore[attr-defined]
                token_length=llm.count_tokens(text),
            )
            for page in doc  # type: ignore[attr-defined]
        ]
    raise ValueError(f"Unsupported PDF parser: {parser}")


def get_text_of_pages(
    pages: list[Page],
    start_page: int,
    end_page: int,
    *,
    with_labels: bool = False,
) -> str:
    """Concatenate text of pages in `[start_page, end_page]` (1-indexed)."""
    parts: list[str] = []
    for page_num in range(start_page - 1, end_page):
        page_text = pages[page_num].text
        if with_labels:
            parts.append(
                f"<physical_index_{page_num + 1}>\n{page_text}\n<physical_index_{page_num + 1}>\n"
            )
        else:
            parts.append(page_text)
    return "".join(parts)
