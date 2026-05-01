from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from rag_pageindex.pageindex.llm.protocol import LLMClient
from rag_pageindex.pageindex.pdf.reader import Page
from rag_pageindex.pageindex.toc.detection import check_toc
from rag_pageindex.pageindex.toc.page_mapping import (
    process_no_toc,
    process_toc_no_page_numbers,
    process_toc_with_page_numbers,
)
from rag_pageindex.pageindex.toc.verification import (
    check_title_appearance_in_start_concurrent,
    fix_incorrect_toc_with_retries,
    validate_and_truncate_physical_indices,
    verify_toc,
)

if TYPE_CHECKING:
    from rag_pageindex.core.config import Settings


def list_to_tree(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a flat list with `structure` keys into a nested tree."""

    def _parent_structure(structure: str | None) -> str | None:
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
    """Insert a 'Preface' node if the first real section doesn't start at page 1."""
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
    """Assign start_index/end_index to each flat item then build the tree."""
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
    """Assign zero-padded node_id values in-place (depth-first pre-order)."""
    counter = [0]

    def _write(node: dict[str, Any] | list[Any]) -> None:
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


def add_node_text(
    node: dict[str, Any] | list[Any], pages: list[Page]
) -> None:
    """Attach concatenated page text to each node in-place."""
    if isinstance(node, dict):
        start = node.get("start_index")
        end = node.get("end_index")
        if start is not None and end is not None:
            node["text"] = "".join(
                pages[i].text for i in range(start - 1, end)
            )
        if "nodes" in node:
            add_node_text(node["nodes"], pages)
    elif isinstance(node, list):
        for item in node:
            add_node_text(item, pages)


async def meta_processor(
    pages: list[Page],
    mode: str,
    *,
    toc_content: str | None = None,
    toc_page_list: list[int] | None = None,
    start_index: int = 1,
    llm: LLMClient,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Run the appropriate TOC extraction path and return verified items."""
    logger.info("meta_processor mode={} start_index={}", mode, start_index)

    if mode == "process_toc_with_page_numbers":
        toc = process_toc_with_page_numbers(
            toc_content,  # type: ignore[arg-type]
            toc_page_list,  # type: ignore[arg-type]
            pages,
            toc_check_page_num=settings.pageindex_toc_check_page_num,
            llm=llm,
            start_index=start_index,
        )
    elif mode == "process_toc_no_page_numbers":
        toc = process_toc_no_page_numbers(
            toc_content,  # type: ignore[arg-type]
            pages,
            start_index=start_index,
            llm=llm,
            max_tokens=settings.pageindex_max_tokens_per_node,
        )
    else:
        toc = process_no_toc(
            pages,
            start_index=start_index,
            llm=llm,
            max_tokens=settings.pageindex_max_tokens_per_node,
        )

    toc = [item for item in toc if item.get("physical_index") is not None]
    toc = validate_and_truncate_physical_indices(
        toc, len(pages), start_index=start_index
    )

    accuracy, incorrect = await verify_toc(
        pages, toc, start_index=start_index, llm=llm
    )
    logger.info(
        "meta_processor verify mode={} accuracy={:.2%} incorrect={}",
        mode,
        accuracy,
        len(incorrect),
    )

    if accuracy == 1.0 and not incorrect:
        return toc

    if accuracy > 0.6 and incorrect:
        toc, _ = await fix_incorrect_toc_with_retries(
            toc,
            pages,
            incorrect,
            start_index=start_index,
            max_attempts=3,
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
        )
    if mode == "process_toc_no_page_numbers":
        return await meta_processor(
            pages,
            "process_no_toc",
            start_index=start_index,
            llm=llm,
            settings=settings,
        )
    raise RuntimeError("meta_processor: all fallback modes exhausted")


async def process_large_node_recursively(
    node: dict[str, Any],
    pages: list[Page],
    *,
    llm: LLMClient,
    settings: Settings,
) -> dict[str, Any]:
    """Recursively split oversized nodes using the no-TOC extraction path."""
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
        )
        sub_toc = await check_title_appearance_in_start_concurrent(
            sub_toc, pages, llm=llm
        )
        valid_sub = [
            item for item in sub_toc if item.get("physical_index") is not None
        ]

        titles_match = (
            valid_sub
            and node["title"].strip() == valid_sub[0]["title"].strip()
        )
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
            process_large_node_recursively(
                child, pages, llm=llm, settings=settings
            )
            for child in node["nodes"]
        ]
        await asyncio.gather(*tasks)

    return node


async def tree_parser(
    pages: list[Page],
    *,
    llm: LLMClient,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Main async entry: detect TOC, build tree, verify, split large nodes."""
    toc_detection = check_toc(
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
        )
    else:
        toc = await meta_processor(
            pages,
            "process_no_toc",
            start_index=1,
            llm=llm,
            settings=settings,
        )

    toc = add_preface_if_needed(toc)
    toc = await check_title_appearance_in_start_concurrent(
        toc, pages, llm=llm
    )
    valid_toc = [
        item for item in toc if item.get("physical_index") is not None
    ]
    tree = post_processing(valid_toc, len(pages))

    tasks = [
        process_large_node_recursively(node, pages, llm=llm, settings=settings)
        for node in tree
    ]
    await asyncio.gather(*tasks)

    return tree
