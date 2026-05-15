from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TreeNode(BaseModel):
    """Public-facing hierarchical tree node returned to consumers."""

    title: str
    start_index: int
    end_index: int
    node_id: str | None = None
    summary: str | None = None
    text: str | None = None
    nodes: list[TreeNode] = Field(default_factory=list)


TreeNode.model_rebuild()


class IndexResult(BaseModel):
    """Top-level result of indexing one document."""

    doc_name: str
    doc_description: str | None = None
    structure: list[dict[str, Any]]
