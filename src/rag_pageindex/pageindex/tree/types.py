from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class TocItem(TypedDict, total=False):
    """In-flight TOC entry the pipeline mutates as it builds the tree.

    Fields appear and disappear across pipeline stages, so all are optional.
    """

    structure: str
    title: str
    page: int | None
    physical_index: int | None
    start_index: int | None
    end_index: int | None
    appear_start: str
    list_index: int
    node_id: str
    text: str
    summary: str
    nodes: list[TocItem]


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
