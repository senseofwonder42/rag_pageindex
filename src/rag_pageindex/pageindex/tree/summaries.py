from __future__ import annotations

import asyncio
import json as _json
from typing import Any

from loguru import logger

from rag_pageindex.pageindex.llm.protocol import LLMClient
from rag_pageindex.pageindex.observability import observe
from rag_pageindex.pageindex.prompts import render

_Tree = dict[str, Any] | list[Any]


def structure_to_list(structure: _Tree) -> list[dict[str, Any]]:
    """Flatten a tree structure into a depth-first list of nodes."""
    if isinstance(structure, dict):
        nodes: list[dict[str, Any]] = [structure]
        if "nodes" in structure:
            nodes.extend(structure_to_list(structure["nodes"]))
        return nodes
    if isinstance(structure, list):
        result: list[dict[str, Any]] = []
        for item in structure:
            result.extend(structure_to_list(item))
        return result
    return []


def remove_structure_text(data: _Tree) -> _Tree:
    """Recursively remove 'text' fields from all nodes in-place."""
    if isinstance(data, dict):
        data.pop("text", None)
        if "nodes" in data:
            remove_structure_text(data["nodes"])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


async def _generate_parent_summary(node: dict[str, Any], *, llm: LLMClient) -> str:
    """Generate a parent-node summary by reducing its child summaries."""
    children: list[dict[str, Any]] = node.get("nodes") or []
    child_summaries = [{"title": c.get("title", ""), "summary": c.get("summary", "")} for c in children]
    prompt = render(
        "generate_node_summary_from_children.j2",
        node_title=node.get("title", ""),
        child_summaries=child_summaries,
    )
    response = await llm.acomplete([{"role": "user", "content": prompt}])
    return response.content


def _group_by_depth(structure: _Tree) -> list[list[dict[str, Any]]]:
    """Return nodes grouped by depth (depth 0 = root)."""
    levels: list[list[dict[str, Any]]] = []

    def _walk(node: _Tree, depth: int) -> None:
        if isinstance(node, dict):
            while len(levels) <= depth:
                levels.append([])
            levels[depth].append(node)
            if "nodes" in node:
                _walk(node["nodes"], depth + 1)
        elif isinstance(node, list):
            for item in node:
                _walk(item, depth)

    _walk(structure, 0)
    return levels


@observe(name="generate_parent_summaries")
async def generate_parent_summaries(
    structure: list[dict[str, Any]],
    *,
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """Bottom-up roll-up: every internal node's summary is reduced from its
    children's summaries. Leaves must already carry a `summary`; they are
    not touched here.
    """
    levels = _group_by_depth(structure)
    for depth in range(len(levels) - 1, -1, -1):
        nodes = [n for n in levels[depth] if n.get("nodes")]
        if not nodes:
            continue

        async def _summarize(node: dict[str, Any]) -> tuple[dict[str, Any], str]:
            return node, await _generate_parent_summary(node, llm=llm)

        results = await asyncio.gather(*(_summarize(n) for n in nodes), return_exceptions=True)
        for entry, node in zip(results, nodes, strict=True):
            if isinstance(entry, BaseException):
                logger.error("parent summary failed for {}: {}", node.get("title"), entry)
                node["summary"] = ""
            else:
                _, summary = entry
                node["summary"] = summary

    return structure


def create_clean_structure_for_description(structure: _Tree) -> _Tree:
    """Extract relevant fields from tree for document description generation."""
    if isinstance(structure, dict):
        clean: dict[str, Any] = {}
        for key in ("title", "node_id", "summary", "prefix_summary"):
            if key in structure:
                clean[key] = structure[key]
        if structure.get("nodes"):
            clean["nodes"] = create_clean_structure_for_description(structure["nodes"])
        return clean
    if isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure]
    return structure


@observe(name="generate_doc_description")
def generate_doc_description(structure: _Tree, *, llm: LLMClient) -> str:
    """Generate a one-sentence document description from tree structure."""
    clean = create_clean_structure_for_description(structure)
    prompt = render(
        "generate_doc_description.j2",
        structure=_json.dumps(clean, indent=2),
    )
    response = llm.complete([{"role": "user", "content": prompt}])
    return response.content
