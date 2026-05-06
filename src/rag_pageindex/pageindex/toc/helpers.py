from __future__ import annotations

from typing import Any


def convert_physical_index_to_int(
    data: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Normalize physical_index values to integers.

    Handles various string formats (<physical_index_X>, physical_index_X, or plain ints).
    Modifies lists/dicts in-place; returns integers and scalars as-is.

    Args:
        data: Scalar, string, dict, or list to normalize.

    Returns:
        Modified data with physical_index fields as ints (or None on parse failure).
    """
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "physical_index" in item:
                value = item["physical_index"]
                if isinstance(value, str):
                    if value.startswith("<physical_index_"):
                        item["physical_index"] = int(value.split("_")[-1].rstrip(">").strip())
                    elif value.startswith("physical_index_"):
                        item["physical_index"] = int(value.split("_")[-1].strip())
                    else:
                        try:
                            item["physical_index"] = int(value.strip())
                        except ValueError:
                            item["physical_index"] = None
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
    """Convert stringified page numbers to integers in-place.

    Silently leaves page numbers that cannot be parsed as invalid strings.

    Args:
        data: List of dicts with potential 'page' fields.

    Returns:
        Modified list with parsed page numbers.
    """
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
    """Recursively remove 'page_number' fields from nested structures.

    Args:
        data: Dict, list, or nested combination to modify.

    Returns:
        Modified data without 'page_number' fields.
    """
    if isinstance(data, dict):
        data.pop("page_number", None)
        for key in list(data.keys()):
            if "nodes" in key:
                remove_page_number(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_page_number(item)
    return data
