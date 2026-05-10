from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from rag_pageindex.pageindex.llm.protocol import LLMClient
from rag_pageindex.pageindex.observability import observe
from rag_pageindex.pageindex.pdf.reader import Page, PdfSource
from rag_pageindex.pageindex.toc.detection import check_toc
from rag_pageindex.pageindex.toc.page_mapping import (
    process_no_toc,
    process_toc_no_page_numbers,
    process_toc_with_page_numbers,
)
from rag_pageindex.pageindex.toc.parsing import toc_transformer
from rag_pageindex.pageindex.toc.verification import (
    check_title_appearance_in_start_concurrent,
    fix_incorrect_toc_with_retries,
    validate_and_truncate_physical_indices,
    verify_toc,
)

if TYPE_CHECKING:
    from rag_pageindex.core.config import Settings


def list_to_tree(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a flat list with `structure` keys into a nested tree.

    Uses the 'structure' field (e.g., '1', '1.2', '1.2.3') to determine
    parent-child relationships and build a hierarchical tree. Removes empty
    'nodes' lists from leaf nodes.

    Args:
        data: Flat list of dicts with 'structure', 'title', 'start_index', 'end_index'.

    Returns:
        Nested tree structure with root nodes as a list.
    """

    def _parent_structure(structure: str | None) -> str | None:
        """Get parent structure index from a dotted structure string."""
        if not structure:
            return None
        parts = str(structure).split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else None

    nodes: dict[str | None, dict[str, Any]] = {}
    root_nodes: list[dict[str, Any]] = []

    for item in data:
        structure = item.get("structure")
        node: dict[str, Any] = {
            "title": item.get("title"),
            "start_index": item.get("start_index"),
            "end_index": item.get("end_index"),
            "nodes": [],
        }
        nodes[structure] = node
        parent = _parent_structure(structure)
        if parent and parent in nodes:
            nodes[parent]["nodes"].append(node)
        else:
            root_nodes.append(node)

    def _clean(node: dict[str, Any]) -> dict[str, Any]:
        """Recursively remove empty 'nodes' lists from tree nodes."""
        if not node["nodes"]:
            del node["nodes"]
        else:
            for child in node["nodes"]:
                _clean(child)
        return node

    return [_clean(n) for n in root_nodes]


def add_preface_if_needed(
    data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Insert a 'Preface' node if the first section doesn't start at page 1.

    Args:
        data: TOC items with 'physical_index' fields.

    Returns:
        Original data with 'Preface' node inserted if needed.
    """
    if not data:
        return data
    first_pi = data[0].get("physical_index")
    if first_pi is not None and first_pi > 1:
        data.insert(
            0,
            {
                "structure": "0",
                "title": "Preface",
                "physical_index": 1,
            },
        )
    return data


def post_processing(
    structure: list[dict[str, Any]],
    end_physical_index: int,
) -> list[dict[str, Any]]:
    """Assign page indices to TOC items and convert to hierarchical tree.

    Sets start_index and end_index for each item based on physical_index
    ordering and appear_start flags, then builds a nested tree structure.

    Args:
        structure: Flat list of TOC items with physical_index.
        end_physical_index: Last page number in the document.

    Returns:
        Hierarchical tree structure (or flat list if tree conversion failed).
    """
    for i, item in enumerate(structure):
        item["start_index"] = item.get("physical_index")
        if i < len(structure) - 1:
            next_item = structure[i + 1]
            if next_item.get("appear_start") == "yes":
                item["end_index"] = next_item["physical_index"] - 1
            else:
                item["end_index"] = next_item["physical_index"]
        else:
            item["end_index"] = end_physical_index

    tree = list_to_tree(structure)
    if tree:
        return tree

    for node in structure:
        node.pop("appear_start", None)
        node.pop("physical_index", None)
    return structure


def write_node_id(data: dict[str, Any] | list[Any]) -> None:
    """Assign zero-padded node IDs to tree nodes in depth-first pre-order.

    Modifies the tree in-place, adding 'node_id' fields with sequential
    zero-padded numbers (0000, 0001, etc.) in pre-order traversal.

    Args:
        data: Tree dict or list to annotate with node IDs.
    """
    counter = [0]

    def _write(node: dict[str, Any] | list[Any]) -> None:
        """Recursively assign node IDs."""
        if isinstance(node, dict):
            node["node_id"] = str(counter[0]).zfill(4)
            counter[0] += 1
            for key in list(node.keys()):
                if "nodes" in key:
                    _write(node[key])
        elif isinstance(node, list):
            for item in node:
                _write(item)

    _write(data)


def add_node_text(node: dict[str, Any] | list[Any], pages: list[Page]) -> None:
    """Attach concatenated page text to LEAF tree nodes only.

    Internal nodes never get a `text` field; their summaries are derived
    from child summaries (see generate_summaries_for_structure). This
    keeps memory peak proportional to the leaf set rather than O(D × text).
    """
    if isinstance(node, dict):
        if node.get("nodes"):
            add_node_text(node["nodes"], pages)
            return
        start = node.get("start_index")
        end = node.get("end_index")
        if start is not None and end is not None:
            node["text"] = "".join(pages[i].text for i in range(start - 1, end))
    elif isinstance(node, list):
        for item in node:
            add_node_text(item, pages)


@observe(name="meta_processor")
async def meta_processor(
    pages: list[Page],
    mode: str,
    *,
    toc_content: str | None = None,
    toc_page_list: list[int] | None = None,
    start_index: int = 1,
    llm: LLMClient,
    settings: Settings,
    source: PdfSource | None = None,
    toc_transformed: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run TOC extraction and verification, with fallback to simpler modes.

    Orchestrates text-based TOC extraction, verification, and optional
    VLM re-extraction. Cascades through modes (with page numbers →
    without page numbers → no TOC) if verification fails.

    Args:
        pages: List of PDF pages.
        mode: Extraction mode ('process_toc_with_page_numbers',
            'process_toc_no_page_numbers', 'process_no_toc').
        toc_content: Extracted TOC text if already available.
        toc_page_list: Page indices containing TOC.
        start_index: Starting page number for this range (1-indexed).
        llm: LLM client for extraction and verification.
        settings: Pipeline configuration.
        source: PDF source for VLM vision fallback.

    Returns:
        List of verified TOC items with physical_index fields.
    """
    logger.info("meta_processor mode={} start_index={}", mode, start_index)

    # Memoise toc_transformer across cascade fall-through: both
    # process_toc_with_page_numbers and process_toc_no_page_numbers
    # transform the same toc_content; compute once on first entry.
    if (
        toc_transformed is None
        and toc_content is not None
        and toc_content.strip()
        and mode in ("process_toc_with_page_numbers", "process_toc_no_page_numbers")
    ):
        try:
            toc_transformed = toc_transformer(
                toc_content,
                llm=llm,
                max_tokens=settings.pageindex_toc_max_output_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("toc_transformer failed: {}; cascade will retry", exc)

    extraction_failed = False
    try:
        if mode == "process_toc_with_page_numbers":
            toc = await process_toc_with_page_numbers(
                toc_content,  # type: ignore[arg-type]
                toc_page_list,  # type: ignore[arg-type]
                pages,
                toc_check_page_num=settings.pageindex_toc_check_page_num,
                llm=llm,
                start_index=start_index,
                toc_max_output_tokens=settings.pageindex_toc_max_output_tokens,
                toc_transformed=toc_transformed,
                toc_index_max_tokens=settings.pageindex_toc_index_max_tokens,
                toc_resolve_max_tokens=settings.pageindex_toc_resolve_max_tokens,
            )
        elif mode == "process_toc_no_page_numbers":
            toc = process_toc_no_page_numbers(
                toc_content,  # type: ignore[arg-type]
                pages,
                start_index=start_index,
                llm=llm,
                max_tokens=settings.pageindex_max_tokens_per_node,
                toc_max_output_tokens=settings.pageindex_toc_max_output_tokens,
                toc_transformed=toc_transformed,
            )
        else:
            toc = process_no_toc(
                pages,
                start_index=start_index,
                llm=llm,
                max_tokens=settings.pageindex_max_tokens_per_node,
            )
    except Exception as exc:
        logger.warning(
            "meta_processor: text extraction failed mode={} ({}); falling through to next stage",
            mode,
            exc,
        )
        toc = []
        extraction_failed = True

    toc = [item for item in toc if item.get("physical_index") is not None]
    toc = validate_and_truncate_physical_indices(toc, len(pages), start_index=start_index)

    if extraction_failed:
        accuracy, incorrect = 0.0, []
    else:
        accuracy, incorrect = await verify_toc(pages, toc, start_index=start_index, llm=llm)
        logger.info(
            "meta_processor verify mode={} accuracy={:.2%} incorrect={}",
            mode,
            accuracy,
            len(incorrect),
        )

    if accuracy == 1.0 and not incorrect:
        return toc

    if accuracy > settings.pageindex_vision_fallback_threshold and incorrect:
        toc, still_incorrect = await fix_incorrect_toc_with_retries(
            toc,
            pages,
            incorrect,
            start_index=start_index,
            llm=llm,
        )
        if still_incorrect and source is not None and settings.pageindex_vision_mode == "fallback":
            from rag_pageindex.pageindex.toc.verification_vlm import fix_incorrect_toc_with_vlm

            toc, _ = await fix_incorrect_toc_with_vlm(
                toc,
                pages,
                still_incorrect,
                source,
                start_index=start_index,
                dpi=settings.pageindex_vision_dpi,
                max_images_per_call=settings.pageindex_vision_max_images_per_call,
                llm=llm,
            )
        return toc

    if mode == "process_toc_with_page_numbers":
        return await meta_processor(
            pages,
            "process_toc_no_page_numbers",
            toc_content=toc_content,
            toc_page_list=toc_page_list,
            start_index=start_index,
            llm=llm,
            settings=settings,
            source=source,
            toc_transformed=toc_transformed,
        )
    if mode == "process_toc_no_page_numbers":
        return await meta_processor(
            pages,
            "process_no_toc",
            start_index=start_index,
            llm=llm,
            settings=settings,
            source=source,
        )

    # process_no_toc failed verification — try VLM re-extraction if enabled
    if source is not None and settings.pageindex_vision_mode == "fallback":
        from rag_pageindex.pageindex.toc.parsing_vlm import regenerate_path_c_range_vlm

        end_page = len(pages) + start_index - 1
        logger.info(
            "meta_processor: VLM re-extraction for pages {}-{} (accuracy={:.2%})",
            start_index,
            end_page,
            accuracy,
        )
        vlm_toc = await regenerate_path_c_range_vlm(
            source,
            start_page=start_index,
            end_page=end_page,
            dpi=settings.pageindex_vision_dpi,
            max_images_per_call=settings.pageindex_vision_max_images_per_call,
            max_output_tokens=settings.pageindex_toc_max_output_tokens,
            llm=llm,
        )
        if vlm_toc:
            vlm_toc = validate_and_truncate_physical_indices(
                vlm_toc, len(pages), start_index=start_index
            )
            return vlm_toc

    logger.warning(
        "meta_processor: process_no_toc verification failed (accuracy={:.2%}, incorrect={})"
        " — returning best-effort toc",
        accuracy,
        len(incorrect),
    )
    return toc


@observe(name="process_large_node")
async def process_large_node_recursively(
    node: dict[str, Any],
    pages: list[Page],
    *,
    llm: LLMClient,
    settings: Settings,
    source: PdfSource | None = None,
) -> dict[str, Any]:
    """Recursively split oversized nodes into sub-trees.

    If a node exceeds max_pages_per_node and max_tokens_per_node, uses
    the no-TOC extraction path to subdivide it. Recursively processes
    child nodes and returns the modified node with sub-nodes added.

    Args:
        node: Tree node to process.
        pages: All pages in the document.
        llm: LLM client for extraction.
        settings: Configuration (page/token limits).
        source: PDF source for VLM fallback.

    Returns:
        Modified node with sub-nodes added if splitting occurred.
    """
    node_pages = pages[node["start_index"] - 1 : node["end_index"]]
    token_num = sum(p.token_length for p in node_pages)

    page_span = node["end_index"] - node["start_index"]
    if (
        page_span > settings.pageindex_max_pages_per_node
        and token_num >= settings.pageindex_max_tokens_per_node
    ):
        logger.info(
            "splitting large node: {} start={} end={} tokens={}",
            node["title"],
            node["start_index"],
            node["end_index"],
            token_num,
        )
        sub_toc = await meta_processor(
            node_pages,
            "process_no_toc",
            start_index=node["start_index"],
            llm=llm,
            settings=settings,
            source=source,
        )
        sub_toc = await check_title_appearance_in_start_concurrent(sub_toc, pages, llm=llm)
        valid_sub = [item for item in sub_toc if item.get("physical_index") is not None]

        titles_match = valid_sub and node["title"].strip() == valid_sub[0]["title"].strip()
        if titles_match:
            node["nodes"] = post_processing(valid_sub[1:], node["end_index"])
            if len(valid_sub) > 1:
                node["end_index"] = valid_sub[1]["start_index"]
        else:
            node["nodes"] = post_processing(valid_sub, node["end_index"])
            if valid_sub:
                node["end_index"] = valid_sub[0]["start_index"]

    if node.get("nodes"):
        tasks = [
            process_large_node_recursively(child, pages, llm=llm, settings=settings, source=source)
            for child in node["nodes"]
        ]
        await asyncio.gather(*tasks)

    return node


@observe(name="tree_parser")
async def tree_parser(
    pages: list[Page],
    *,
    llm: LLMClient,
    settings: Settings,
    source: PdfSource | None = None,
) -> list[dict[str, Any]]:
    """Main entry point: detect TOC, build tree, verify, and split large nodes.

    Orchestrates the full tree-building pipeline: checks for a TOC,
    extracts and verifies it, builds the tree structure, and recursively
    splits any oversized nodes.

    Args:
        pages: List of PDF pages.
        llm: LLM client for extraction and verification.
        settings: Pipeline configuration.
        source: PDF source for VLM vision fallback.

    Returns:
        Hierarchical tree structure (list of root nodes).
    """
    toc_detection = await check_toc(
        pages,
        toc_check_page_num=settings.pageindex_toc_check_page_num,
        llm=llm,
    )
    logger.info("toc_detection: {}", toc_detection)

    if (
        toc_detection.toc_content
        and toc_detection.toc_content.strip()
        and toc_detection.page_index_given_in_toc == "yes"
    ):
        toc = await meta_processor(
            pages,
            "process_toc_with_page_numbers",
            start_index=1,
            toc_content=toc_detection.toc_content,
            toc_page_list=toc_detection.toc_page_list,
            llm=llm,
            settings=settings,
            source=source,
        )
    else:
        toc = await meta_processor(
            pages,
            "process_no_toc",
            start_index=1,
            llm=llm,
            settings=settings,
            source=source,
        )

    toc = add_preface_if_needed(toc)
    toc = await check_title_appearance_in_start_concurrent(toc, pages, llm=llm)
    valid_toc = [item for item in toc if item.get("physical_index") is not None]
    tree = post_processing(valid_toc, len(pages))

    tasks = [
        process_large_node_recursively(node, pages, llm=llm, settings=settings, source=source)
        for node in tree
    ]
    await asyncio.gather(*tasks)

    return tree
