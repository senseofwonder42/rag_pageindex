# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

RAG system using a refactored fork of [PageIndex](https://github.com/VectifyAI/PageIndex) (vendored under [src/rag_pageindex/pageindex/](src/rag_pageindex/pageindex/)) as the retrieval backend. PageIndex is *reasoning-based* (no embeddings): an LLM extracts a hierarchical tree of pages from a PDF, then queries are routed down that tree.

## Commands

- `uv sync` — install deps after a checkout
- `uv add <pkg>` / `uv add <pkg> --dev` / `uv remove <pkg>` — only sanctioned way to change deps. **Never** `uv pip install`, **never** manually `uv venv`. The `uv add/remove` commands keep `pyproject.toml`, `uv.lock`, and the venv in sync.
- `uv run start-app --pdf-path <file>` — entrypoint (resolves to `rag_pageindex.main:main`)
- `uv run pytest` — all tests; `uv run pytest tests/test_foo.py::test_bar` — single test
- `uv run ruff check src tests` — lint
- `uv run ruff check --fix && uv run ruff format` — autofix + format
- `uv run mypy` — type-check (configured to scan `src/` only)

## Toolchain conventions

- Python ≥3.12, line length 79, indent 4.
- Ruff lint set is strict: `E, W, F, I, B, PTH, ANN, ARG`. `ANN` (flake8-annotations) means **all functions need type annotations**. `PTH` means **prefer `pathlib` over `os.path`**. Tests are exempt from `ANN`/`ARG` per `pyproject.toml`.
- mypy uses `pydantic.mypy`, `check_untyped_defs`, `no_implicit_optional`; excludes `tests/`.
- pytest auto-injects `ENVIRONMENT=test` and `LOG_LEVEL=INFO` via `pytest-env` — don't hardcode these in tests.

## Architecture

`src/` layout, single package [src/rag_pageindex/](src/rag_pageindex/).

### Cross-cutting (`core/`)
- [core/config.py](src/rag_pageindex/core/config.py) — `Settings` (pydantic-settings) loads from `.env`. Access via the cached `settings` singleton or `get_settings()`. **All env vars must be added as fields on `Settings`** — never read `os.environ` directly. Replaces the upstream `pageindex/config.yaml` entirely.
- [core/logging.py](src/rag_pageindex/core/logging.py) — `setup_logging(level)` configures **loguru** (not stdlib logging). Call once at program start from `main.py`. Use `from loguru import logger` everywhere; never use `print()` for diagnostic output.
- [core/constants.py](src/rag_pageindex/core/constants.py) — `PROJECT_ROOT: Path` for filesystem-anchored paths.

### PageIndex pipeline ([pageindex/](src/rag_pageindex/pageindex/))
Vendored from upstream commit `a51d97f` (MIT, Vectify AI) and refactored. Upstream was a single ~1100-line monolith; we split it into focused modules:

| Module | Owns |
|---|---|
| [llm/](src/rag_pageindex/pageindex/llm/) | `LLMClient` Protocol, `AnthropicClient` impl, retry helper, factory |
| [prompts/](src/rag_pageindex/pageindex/prompts/) | One `.j2` Jinja template per LLM call + `render()` loader |
| [pdf/reader.py](src/rag_pageindex/pageindex/pdf/reader.py) | PyMuPDF/PyPDF2 wrapper, per-page tokenization |
| [toc/](src/rag_pageindex/pageindex/toc/) | TOC detection, parsing (TOC text→JSON), page-mapping, verification |
| [tree/](src/rag_pageindex/pageindex/tree/) | Pydantic tree types, builder, post-processing |
| `pipeline.py` | Top-level `page_index_builder()` orchestrator |
| `client.py` | Single-doc workspace (multi-doc store deferred) |
| `json_extract.py` | Tolerant parser for model-emitted JSON (handles ```json fences, trailing commas, etc.) |

### Hard rules from the refactor

1. **No `litellm` and no `openai` SDK.** Every LLM call goes through the `LLMClient` protocol ([pageindex/llm/protocol.py](src/rag_pageindex/pageindex/llm/protocol.py)). Pass an `LLMClient` instance into pipeline functions; never reach for a global. To add a new provider, write another implementation alongside `AnthropicClient`.
2. **No hardcoded prompts in Python.** Every prompt lives as a `.j2` file under [pageindex/prompts/](src/rag_pageindex/pageindex/prompts/) and is rendered via `prompts.render("name.j2", **ctx)`. The Jinja env uses `StrictUndefined`, so a typo in a context variable raises at render time.
3. **No `pyyaml` config file.** All tuning knobs live as fields on `Settings` (`pageindex_max_pages_per_node`, `pageindex_max_tokens_per_node`, `pageindex_token_ceiling`, `pageindex_add_node_summary`, etc.). The upstream `config.yaml` is intentionally not vendored.
4. **Loguru only.** Replace any upstream `print()` / stdlib `logging` with `logger.debug/info/warning/error` from loguru.
5. **In-flight pipeline state uses `TocItem` TypedDicts** ([tree/types.py](src/rag_pageindex/pageindex/tree/types.py)). The *output* uses pydantic models (`TreeNode`, `IndexResult`). Don't try to convert internal dicts to pydantic mid-pipeline — the algorithms mutate them heavily across stages.

### Out of scope (deferred for a follow-up)
- Multi-document corpus store / cross-doc query routing.
- Markdown ingestion path (upstream `page_index_md.py`).
- Adaptive chunking past the `pageindex_token_ceiling` (default 110k).
- Async-throughout rewrite of `client.py`'s sync/async juggling.

### Upstream reference
A read-only clone of upstream PageIndex at the vendored commit lives at `/tmp/pageindex_upstream/` (not in the repo). Useful when porting more upstream code or comparing behavior. Re-clone with `git clone --depth 1 https://github.com/VectifyAI/PageIndex.git /tmp/pageindex_upstream` if it's missing.
