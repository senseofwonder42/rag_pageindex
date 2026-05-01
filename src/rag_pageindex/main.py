from __future__ import annotations

import argparse
import json
import sys

from rag_pageindex.core.config import settings
from rag_pageindex.core.logging import setup_logging
from rag_pageindex.pageindex.llm.factory import get_default_client
from rag_pageindex.pageindex.pipeline import page_index


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index a PDF with PageIndex and print the tree as JSON."
    )
    parser.add_argument(
        "--pdf-path", required=True, help="Path to the PDF file"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    setup_logging(settings.log_level)
    args = _parse_args(argv)
    llm = get_default_client(settings)
    result = page_index(args.pdf_path, llm=llm, settings=settings)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
