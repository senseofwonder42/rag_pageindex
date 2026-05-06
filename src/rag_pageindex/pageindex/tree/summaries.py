from __future__ import annotations

import asyncio
import json as _json
from typing import Any

from rag_pageindex.pageindex.llm.protocol import LLMClient
from rag_pageindex.pageindex.observability import observe
from rag_pageindex.pageindex.prompts import render

_Tree = dict[str, Any] | list[Any]


def structure_to_list(structure: _Tree) -> list[dict[str, Any]]:
    """Flatten a tree structure into a depth-first list of nodes.

    Args:
        structure: Tree dict, list, or scalar.

    Returns:
        List of all dict nodes in depth-first order.
    """
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
    """Recursively remove 'text' fields from all nodes in-place.

    Args:
        data: Tree dict or list to modify.

    Returns:
        Modified tree without 'text' fields.
    """
    if isinstance(data, dict):
        data.pop("text", None)
        if "nodes" in data:
            remove_structure_text(data["nodes"])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


async def generate_node_summary(node: dict[str, Any], *, llm: LLMClient) -> str:
    """Generate a concise summary for a tree node's content.

    Args:
        node: Tree node dict with 'text' field.
        llm: LLM client for generation.

    Returns:
        Generated summary string.
    """
    prompt = render("generate_node_summary.j2", node_text=node["text"])
    response = await llm.acomplete([{"role": "user", "content": prompt}])
    return response.content


@observe(name="generate_summaries_for_structure")
async def generate_summaries_for_structure(
    structure: list[dict[str, Any]],
    *,
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """Generate summaries for all tree nodes concurrently.

    Flattens the tree, generates summaries for each node in parallel,
    and adds the 'summary' field to each node in-place.

    Args:
        structure: Tree structure (list of dicts).
        llm: LLM client for summary generation.

    Returns:
        Modified structure with 'summary' fields added.
    """
    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, llm=llm) for node in nodes]
    summaries = await asyncio.gather(*tasks)
    for node, summary in zip(nodes, summaries, strict=True):
        node["summary"] = summary
    return structure


def create_clean_structure_for_description(structure: _Tree) -> _Tree:
    """Extract relevant fields from tree for document description generation.

    Recursively keeps only title, node_id, summary, and prefix_summary fields
    to reduce context when generating a document-level description.

    Args:
        structure: Full tree structure.

    Returns:
        Subset tree with only relevant fields.
    """
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
    """Generate a one-sentence document description from tree structure.

    Uses the tree's titles, IDs, and summaries to create a concise
    overview of the document's contents and organization.

    Args:
        structure: Tree structure (with summaries already generated).
        llm: LLM client for description generation.

    Returns:
        Single-sentence document description.
    """
    clean = create_clean_structure_for_description(structure)
    prompt = render(
        "generate_doc_description.j2",
        structure=_json.dumps(clean, indent=2),
    )
    response = llm.complete([{"role": "user", "content": prompt}])
    return response.content
