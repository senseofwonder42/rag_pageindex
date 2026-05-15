from __future__ import annotations

from typing import Any

from rag_pageindex.pageindex.pdf.reader import Page


def write_node_id(data: dict[str, Any] | list[Any]) -> None:
    """Assign zero-padded node IDs to tree nodes in depth-first pre-order.

    Modifies the tree in-place, adding 'node_id' fields with sequential
    zero-padded numbers (0000, 0001, etc.) in pre-order traversal.
    """
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


def add_node_text(node: dict[str, Any] | list[Any], pages: list[Page]) -> None:
    """Attach concatenated page text to LEAF tree nodes only.

    Internal nodes never get a `text` field; their summaries are derived
    from child summaries (see generate_summaries_for_structure).
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
