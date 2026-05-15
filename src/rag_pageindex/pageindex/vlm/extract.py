from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

import pymupdf
from loguru import logger

from rag_pageindex.pageindex.llm.protocol import ContentPart, LLMClient
from rag_pageindex.pageindex.observability import observe
from rag_pageindex.pageindex.pdf.reader import PdfSource
from rag_pageindex.pageindex.pdf.renderer import image_part, render_pages
from rag_pageindex.pageindex.prompts import render
from rag_pageindex.pageindex.structured_responses import (
    PageBatchResponse,
    PageInfo,
    VisualRank,
)

if TYPE_CHECKING:
    from io import BytesIO

    from rag_pageindex.core.config import Settings


# Largest visual rank → smallest. Index in this tuple defines depth ordering.
_RANK_ORDER: tuple[VisualRank, ...] = ("xlarge", "large", "medium", "small", "xsmall")

# "1.", "1.2", "1.2.3", "5.1 ", "A.1" — numeric (with optional alpha top level)
# section prefix at the start of a heading. Trailing dot is optional.
_NUMBERING_RE = re.compile(r"^([A-Z]?\d+(?:\.\d+)*)\.?(?=\s|$)")

# Figure / table / algorithm / equation captions that the VLM often
# misclassifies as headings. Drop these before tree-building.
_CAPTION_RE = re.compile(
    r"^(table|figure|fig\.|algorithm|alg\.|equation|eq\.|listing|scheme)\s*\d",
    re.IGNORECASE,
)


def _depth_from_numbering(text: str) -> int | None:
    """Return tree depth implied by a heading's section numbering, or None.

    `"1. Intro"` → 1, `"2.3 Methods"` → 2, `"4.1.2 Detail"` → 3, `"A.1"` → 2.
    Returns None for headings without a recognisable numeric prefix.
    """
    match = _NUMBERING_RE.match(text.strip())
    if not match:
        return None
    return match.group(1).count(".") + 1


def _is_caption_noise(text: str) -> bool:
    """True when a 'heading' is really a figure / table / algorithm caption."""
    return bool(_CAPTION_RE.match(text.strip()))


def _page_count(source: PdfSource) -> int:
    """Return the number of pages in the PDF source."""
    if hasattr(source, "read") and hasattr(source, "seek"):
        # BytesIO-like
        doc = pymupdf.open(stream=source, filetype="pdf")  # type: ignore[arg-type]
    else:
        doc = pymupdf.open(str(source))
    try:
        return doc.page_count
    finally:
        doc.close()


def _chunk(seq: list[int], size: int) -> list[list[int]]:
    """Split a list into consecutive chunks of at most `size` items."""
    size = max(1, size)
    return [seq[i : i + size] for i in range(0, len(seq), size)]


async def _call_batch(
    batch: list[int],
    rendered: dict[int, bytes],
    *,
    llm: LLMClient,
) -> list[PageInfo]:
    """Send one batch of rendered pages to the VLM and return its records."""
    if not batch:
        return []
    prompt_text = render(
        "vlm_page_batch.j2",
        num_pages=len(batch),
        page_indices=batch,
    )
    parts: list[ContentPart] = [{"type": "text", "text": prompt_text}]
    for idx in batch:
        png = rendered.get(idx)
        if png is None:
            logger.warning("missing rendered page {} — skipping in batch", idx)
            continue
        parts.append(image_part(png))

    response = await llm.acomplete_structured(
        [{"role": "user", "content": parts}],
        PageBatchResponse,
    )
    # Defensive: filter records whose page_index is outside the batch.
    valid_indices = set(batch)
    records = [p for p in response.pages if p.page_index in valid_indices]
    if len(records) != len(batch):
        logger.warning(
            "VLM batch returned {} records for {} requested pages: {}",
            len(records),
            len(batch),
            batch,
        )
    return records


def _build_rank_map(records: list[PageInfo]) -> dict[VisualRank, int]:
    """Map every rank actually seen in the document to a 1-based depth.

    Ranks are ordered by intended visual size (`xlarge` largest → `xsmall`
    smallest). Only ranks that appear at least once get a depth, so depths
    are dense (no gaps), which matches what the tree-builder expects.
    """
    seen: set[VisualRank] = set()
    for rec in records:
        for h in rec.headings:
            seen.add(h.visual_rank)
    return {rank: depth for depth, rank in enumerate((r for r in _RANK_ORDER if r in seen), start=1)}


def _build_tree(
    records: list[PageInfo],
    *,
    rank_to_depth: dict[VisualRank, int],
    last_page: int,
) -> list[dict[str, Any]]:
    """Walk per-page records in order and produce the nested tree.

    Front matter (pages before the first heading) becomes a synthetic
    top-level leaf node titled "Front Matter".
    """
    root: list[dict[str, Any]] = []
    stack: list[tuple[int, dict[str, Any]]] = []

    def _open_node(title: str, depth: int, page: int) -> dict[str, Any]:
        node: dict[str, Any] = {
            "title": title,
            "start_index": page,
            "_depth": depth,
            "_page_descriptions": [],
            "nodes": [],
        }
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if stack:
            stack[-1][1]["nodes"].append(node)
        else:
            root.append(node)
        stack.append((depth, node))
        return node

    first_heading_page: int | None = None
    for rec in records:
        for h in rec.headings:
            title = h.text.strip()
            if not title or _is_caption_noise(title):
                continue
            depth = _depth_from_numbering(title)
            if depth is None:
                depth = rank_to_depth.get(h.visual_rank, 1)
            if first_heading_page is None:
                first_heading_page = rec.page_index
            _open_node(title, depth, rec.page_index)

    if first_heading_page is None:
        # No headings detected at all: whole doc becomes one leaf.
        whole = {
            "title": "Document",
            "start_index": 1,
            "end_index": last_page,
            "_depth": 1,
            "_page_descriptions": [r.description for r in records],
            "nodes": [],
        }
        return [whole]

    if first_heading_page > 1:
        root.insert(
            0,
            {
                "title": "Front Matter",
                "start_index": 1,
                "end_index": first_heading_page - 1,
                "_depth": 1,
                "_page_descriptions": [],
                "nodes": [],
            },
        )

    _assign_end_indices(root, last_page)
    _attach_page_descriptions(root, records)
    return root


def _assign_end_indices(structure: list[dict[str, Any]], last_page: int) -> None:
    """Set `end_index` for every node so siblings tile their parent's range.

    Pre-order walk: a node ends at (next sibling's start - 1) if any, else at
    the parent's end. The traversal is iterative to avoid recursion on deep
    trees.
    """

    def _walk(siblings: list[dict[str, Any]], parent_end: int) -> None:
        for i, node in enumerate(siblings):
            if i + 1 < len(siblings):
                node_end = siblings[i + 1]["start_index"] - 1
            else:
                node_end = parent_end
            # Front Matter already has end_index set explicitly; respect it.
            if "end_index" not in node:
                node["end_index"] = max(node_end, node["start_index"])
            if node.get("nodes"):
                _walk(node["nodes"], node["end_index"])

    _walk(structure, last_page)


def _attach_page_descriptions(
    structure: list[dict[str, Any]],
    records: list[PageInfo],
) -> None:
    """For each leaf, collect per-page descriptions for pages in its range."""
    page_to_desc: dict[int, str] = {r.page_index: r.description for r in records}

    def _walk(node: dict[str, Any]) -> None:
        if node.get("nodes"):
            for child in node["nodes"]:
                _walk(child)
            return
        start = node["start_index"]
        end = node["end_index"]
        existing = node.get("_page_descriptions") or []
        if existing:
            return
        node["_page_descriptions"] = [
            {"page": p, "description": page_to_desc[p]}
            for p in range(start, end + 1)
            if p in page_to_desc
        ]

    for n in structure:
        _walk(n)


def strip_internal_fields(structure: list[dict[str, Any]]) -> None:
    """Remove `_depth` / `_page_descriptions` from every node (pre-serialization)."""

    def _walk(node: dict[str, Any]) -> None:
        node.pop("_depth", None)
        node.pop("_page_descriptions", None)
        for child in node.get("nodes", []) or []:
            _walk(child)

    for n in structure:
        _walk(n)


@observe(name="vlm_extract_tree")
async def extract_tree(
    source: PdfSource | BytesIO,
    *,
    llm: LLMClient,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Build the document tree by sending batched page images to a VLM.

    Returns a list of nested node dicts with `title`, `start_index`,
    `end_index`, `nodes`, plus a transient `_page_descriptions` field used
    by the summary stage. Strip the transient field before serialization
    using `_strip_internal_fields`.
    """
    total = _page_count(source)
    if total == 0:
        return []

    dpi = settings.pageindex_vlm_dpi
    batch_size = max(
        1,
        min(
            settings.pageindex_vlm_pages_per_batch,
            settings.pageindex_vlm_max_images_per_call,
        ),
    )

    page_indices = list(range(1, total + 1))
    rendered = render_pages(source, page_indices, dpi=dpi)
    batches = _chunk(page_indices, batch_size)
    logger.info(
        "vlm extract: {} pages → {} batches of ≤{} (dpi={})",
        total,
        len(batches),
        batch_size,
        dpi,
    )

    results = await asyncio.gather(
        *(_call_batch(b, rendered, llm=llm) for b in batches),
        return_exceptions=True,
    )

    all_records: list[PageInfo] = []
    for batch, res in zip(batches, results, strict=True):
        if isinstance(res, BaseException):
            logger.error("VLM batch {} failed: {}", batch, res)
            continue
        all_records.extend(res)
    all_records.sort(key=lambda r: r.page_index)

    rank_to_depth = _build_rank_map(all_records)
    logger.info("rank→depth mapping: {}", rank_to_depth)

    structure = _build_tree(all_records, rank_to_depth=rank_to_depth, last_page=total)
    return structure
