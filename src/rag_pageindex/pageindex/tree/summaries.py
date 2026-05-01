from __future__ import annotations

import asyncio
import json as _json
from typing import Any

from rag_pageindex.pageindex.llm.protocol import LLMClient
from rag_pageindex.pageindex.prompts import render

_Tree = dict[str, Any] | list[Any]


def structure_to_list(structure: _Tree) -> list[dict[str, Any]]:
    """Flatten a tree into a list of all nodes (depth-first)."""
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
    """Strip the 'text' key from all nodes in-place."""
    if isinstance(data, dict):
        data.pop("text", None)
        if "nodes" in data:
            remove_structure_text(data["nodes"])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


async def generate_node_summary(node: dict[str, Any], *, llm: LLMClient) -> str:
    """Generate a description for a single tree node."""
    prompt = render("generate_node_summary.j2", node_text=node["text"])
    response = await llm.acomplete([{"role": "user", "content": prompt}])
    return response.content


async def generate_summaries_for_structure(
    structure: list[dict[str, Any]],
    *,
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """Generate summaries for all nodes concurrently, adding `summary` in-place."""
    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, llm=llm) for node in nodes]
    summaries = await asyncio.gather(*tasks)
    for node, summary in zip(nodes, summaries, strict=True):
        node["summary"] = summary
    return structure


def create_clean_structure_for_description(structure: _Tree) -> _Tree:
    """Subset of tree with only fields needed to generate a doc description."""
    if isinstance(structure, dict):
        clean: dict[str, Any] = {}
        for key in ("title", "node_id", "summary", "prefix_summary"):
            if key in structure:
                clean[key] = structure[key]
        if structure.get("nodes"):
            clean["nodes"] = create_clean_structure_for_description(
                structure["nodes"]
            )
        return clean
    if isinstance(structure, list):
        return [
            create_clean_structure_for_description(item) for item in structure
        ]
    return structure


def generate_doc_description(structure: _Tree, *, llm: LLMClient) -> str:
    """Generate a one-sentence document description from the tree structure."""
    clean = create_clean_structure_for_description(structure)
    prompt = render(
        "generate_doc_description.j2",
        structure=_json.dumps(clean, indent=2),
    )
    response = llm.complete([{"role": "user", "content": prompt}])
    return response.content
