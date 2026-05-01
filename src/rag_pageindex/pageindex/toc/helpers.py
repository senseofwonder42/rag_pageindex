from __future__ import annotations

from typing import Any


def convert_physical_index_to_int(
    data: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Normalize <physical_index_X> tokens into plain ints in-place."""
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "physical_index" in item:
                value = item["physical_index"]
                if isinstance(value, str):
                    if value.startswith("<physical_index_"):
                        item["physical_index"] = int(
                            value.split("_")[-1].rstrip(">").strip()
                        )
                    elif value.startswith("physical_index_"):
                        item["physical_index"] = int(
                            value.split("_")[-1].strip()
                        )
        return data
    if isinstance(data, str):
        if data.startswith("<physical_index_"):
            return int(data.split("_")[-1].rstrip(">").strip())
        if data.startswith("physical_index_"):
            return int(data.split("_")[-1].strip())
        return None
    if isinstance(data, int):
        return data
    return data


def convert_page_to_int(
    data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cast stringified page numbers to int; leave on failure."""
    for item in data:
        if "page" in item and isinstance(item["page"], str):
            try:
                item["page"] = int(item["page"])
            except ValueError:
                pass
    return data


def remove_page_number(
    data: dict[str, Any] | list[Any],
) -> dict[str, Any] | list[Any]:
    """Strip 'page_number' / 'page' keys throughout a nested structure."""
    if isinstance(data, dict):
        data.pop("page_number", None)
        for key in list(data.keys()):
            if "nodes" in key:
                remove_page_number(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_page_number(item)
    return data
