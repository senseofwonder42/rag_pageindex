import json
from typing import Any

from loguru import logger


def get_json_content(response: str) -> str:
    """Extract JSON content by removing markdown code fences.

    Strips the ```json ... ``` fence markers if present, returning the
    raw JSON content between them (or the entire string if no fences).

    Args:
        response: Model response potentially containing JSON code fences.

    Returns:
        JSON content with fences removed and whitespace stripped.
    """
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]

    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]

    return response.strip()


def extract_json(content: str) -> Any:  # noqa: ANN401
    """Parse JSON from model output, tolerating common formatting issues.

    Handles JSON code fences, replaces Python None with null, normalizes
    whitespace, and recovers from trailing commas. Returns empty dict on
    parse failure to prevent downstream crashes on `.get()` calls.

    Args:
        content: Raw model output potentially containing JSON.

    Returns:
        Parsed JSON object/array, or empty dict on unrecoverable error.
    """
    try:
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            json_content = content.strip()

        json_content = json_content.replace("None", "null")
        json_content = json_content.replace("\n", " ").replace("\r", " ")
        json_content = " ".join(json_content.split())

        return json.loads(json_content)
    except json.JSONDecodeError as exc:
        logger.error("Failed to extract JSON: {}", exc)
        try:
            json_content = json_content.replace(",]", "]").replace(",}", "}")
            return json.loads(json_content)
        except Exception:  # noqa: BLE001
            logger.error("Failed to parse JSON even after cleanup")
            return {}
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error while extracting JSON: {}", exc)
        return {}
