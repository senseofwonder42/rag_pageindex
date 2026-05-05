from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

from rag_pageindex.core.config import settings
from rag_pageindex.core.logging import setup_logging
from rag_pageindex.pageindex.llm.factory import get_default_client
from rag_pageindex.pageindex.pipeline import page_index

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rag_pageindex.core.config import Settings


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index a PDF with PageIndex and print the tree as JSON."
    )
    parser.add_argument("--pdf-path", required=True, help="Path to the PDF file")
    return parser.parse_args(argv)


@contextmanager
def _tracing(settings: Settings) -> Iterator[None]:
    """Initialize Langfuse if enabled; flush on exit so partial runs aren't lost."""
    if not settings.tracing_enabled:
        yield
        return

    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        raise RuntimeError(
            "tracing_enabled=True but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set."
        )

    from langfuse import Langfuse

    client = Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )
    try:
        yield
    finally:
        client.flush()


def main(argv: list[str] | None = None) -> None:
    setup_logging(settings.log_level)
    args = _parse_args(argv)
    with _tracing(settings):
        llm = get_default_client(settings)
        result = page_index(args.pdf_path, llm=llm, settings=settings)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
