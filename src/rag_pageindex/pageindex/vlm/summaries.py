from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from rag_pageindex.pageindex.llm.protocol import LLMClient
from rag_pageindex.pageindex.observability import observe
from rag_pageindex.pageindex.prompts import render


def _collect_leaves_with_path(
    structure: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], list[str]]]:
    """Return [(leaf_node, ancestor_titles)] in pre-order."""
    out: list[tuple[dict[str, Any], list[str]]] = []

    def _walk(nodes: list[dict[str, Any]], path: list[str]) -> None:
        for node in nodes:
            if node.get("nodes"):
                _walk(node["nodes"], [*path, node.get("title", "")])
            else:
                out.append((node, path))

    _walk(structure, [])
    return out


async def _summarize_leaf(
    node: dict[str, Any],
    path: list[str],
    *,
    llm: LLMClient,
) -> str:
    page_descriptions = node.get("_page_descriptions") or []
    if not page_descriptions:
        return ""
    prompt = render(
        "roll_up_leaf_summary.j2",
        node_title=node.get("title", ""),
        path_titles=path,
        page_descriptions=page_descriptions,
    )
    response = await llm.acomplete([{"role": "user", "content": prompt}])
    return response.content.strip()


@observe(name="roll_up_leaf_summaries")
async def roll_up_leaf_summaries(
    structure: list[dict[str, Any]],
    *,
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """Generate a summary for every leaf node from its per-page descriptions.

    The transient `_page_descriptions` field set by `extract_tree` is read
    here; it stays on each node so callers can inspect it before
    serialization (see `_strip_internal_fields`).
    """
    leaves = _collect_leaves_with_path(structure)
    if not leaves:
        return structure

    async def _one(node: dict[str, Any], path: list[str]) -> tuple[dict[str, Any], str]:
        return node, await _summarize_leaf(node, path, llm=llm)

    results = await asyncio.gather(
        *(_one(n, p) for n, p in leaves),
        return_exceptions=True,
    )
    for entry, (node, _) in zip(results, leaves, strict=True):
        if isinstance(entry, BaseException):
            logger.error("leaf summary failed for {}: {}", node.get("title"), entry)
            node["summary"] = ""
        else:
            _, summary = entry
            node["summary"] = summary

    return structure
