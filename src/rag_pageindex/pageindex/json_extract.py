import json
from typing import Any

from loguru import logger


def get_json_content(response: str) -> str:
    """Strip ```json ... ``` fences from a model response."""
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]

    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]

    return response.strip()


def extract_json(content: str) -> Any:  # noqa: ANN401
    """Parse model-emitted JSON, tolerating common formatting issues.

    Returns `{}` on unrecoverable failures so downstream `.get()` calls don't
    crash — matching upstream behavior.
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
