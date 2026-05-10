from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from rag_pageindex.core.config import settings
from rag_pageindex.pageindex.llm.factory import get_default_client
from rag_pageindex.pageindex.llm.protocol import LLMClient
from rag_pageindex.pageindex.pipeline import apage_index

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rag_pageindex.core.config import Settings

_console = Console()
_LOG_FORMAT = "{time:HH:mm:ss} | {level: <8} | {name} - {message}"


class _Result(NamedTuple):
    name: str
    out: Path | None
    elapsed: float
    ok: bool


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the PDF indexing application.

    Args:
        argv: List of command-line arguments. If None, uses sys.argv[1:].

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Index PDF(s) with PageIndex and save the tree as JSON."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pdf-path", type=Path, help="Path to a single PDF file"
    )
    group.add_argument(
        "--input-dir", type=Path, help="Directory of PDF files to index"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write JSON files (default: same directory as each PDF)",
    )
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


def _collect_pdfs(args: argparse.Namespace) -> list[Path]:
    """Return sorted list of PDF paths derived from parsed arguments.

    Args:
        args: Parsed argument namespace with pdf_path or input_dir set.

    Returns:
        Sorted list of PDF paths to index.
    """
    if args.pdf_path is not None:
        return [args.pdf_path]
    return sorted(args.input_dir.glob("*.pdf"))


def _output_path(pdf: Path, output_dir: Path | None) -> Path:
    """Compute the destination JSON path for a given PDF.

    Args:
        pdf: Source PDF path.
        output_dir: Override output directory, or None to use the PDF's directory.

    Returns:
        Path where the JSON index should be written.
    """
    dest = output_dir if output_dir is not None else pdf.parent
    return dest / (pdf.stem + ".json")


def _setup_logging(level: str) -> None:
    """Route loguru through the shared rich Console to avoid live-display conflicts.

    Args:
        level: Minimum log level to emit.
    """
    logger.remove()
    logger.add(
        lambda msg: _console.print(msg, end="", markup=False, highlight=False),
        level=level,
        format=_LOG_FORMAT,
        colorize=False,
    )


def _print_summary(results: list[_Result]) -> None:
    """Render a results table after processing multiple PDFs.

    Args:
        results: Per-PDF indexing outcomes.
    """
    table = Table(
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
        title="[bold]Summary[/bold]",
    )
    table.add_column("", justify="center", width=3)
    table.add_column("File", style="dim")
    table.add_column("Output")
    table.add_column("Time", justify="right")

    for r in results:
        mins, secs = divmod(int(r.elapsed), 60)
        table.add_row(
            "[green]✓[/green]" if r.ok else "[red]✗[/red]",
            r.name,
            str(r.out) if r.ok and r.out is not None else "[red]failed[/red]",
            f"{mins}:{secs:02d}",
        )

    _console.print(table)


async def _index_one(
    pdf: Path,
    *,
    llm: LLMClient,
    output_dir: Path | None,
    progress: Progress,
    task_id: TaskID,
) -> _Result:
    """Index one PDF inside the shared event loop and return its outcome."""
    progress.update(task_id, description=pdf.name)
    t0 = time.monotonic()
    try:
        result = await apage_index(pdf, llm=llm, settings=settings)
        out = _output_path(pdf, output_dir)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return _Result(pdf.name, out, time.monotonic() - t0, True)
    except Exception as exc:  # noqa: BLE001
        progress.console.print(f"[bold red]✗ {pdf.name}[/bold red]: {exc}")
        return _Result(pdf.name, None, time.monotonic() - t0, False)
    finally:
        progress.advance(task_id)


async def _run_batch(
    pdfs: list[Path],
    *,
    llm: LLMClient,
    output_dir: Path | None,
) -> list[_Result]:
    """Process every PDF sequentially under one event loop."""
    results: list[_Result] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=_console,
    ) as progress:
        task_id = progress.add_task(pdfs[0].name, total=len(pdfs))
        for pdf in pdfs:
            results.append(
                await _index_one(
                    pdf,
                    llm=llm,
                    output_dir=output_dir,
                    progress=progress,
                    task_id=task_id,
                )
            )
    return results


def main(argv: list[str] | None = None) -> None:
    """Index one or more PDFs and save each result as a JSON file.

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].
    """
    args = _parse_args(argv)
    _setup_logging(settings.log_level)

    pdfs = _collect_pdfs(args)
    if not pdfs:
        _console.print("[bold red]No PDF files found.[/bold red]")
        sys.exit(1)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    n = len(pdfs)
    out_label = str(args.output_dir) if args.output_dir else "alongside each PDF"
    _console.print(
        Panel(
            f"[bold]Indexing {n} PDF{'s' if n > 1 else ''}[/bold]  →  {out_label}",
            title="[bold cyan]PageIndex[/bold cyan]",
            expand=False,
        )
    )

    llm = get_default_client(settings)

    with _tracing(settings):
        results = asyncio.run(
            _run_batch(pdfs, llm=llm, output_dir=args.output_dir)
        )

    if n > 1:
        _print_summary(results)
    elif results[0].ok and results[0].out is not None:
        _console.print(
            f"[green]✓[/green] Saved to [bold]{results[0].out}[/bold]"
        )

    if any(not r.ok for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
