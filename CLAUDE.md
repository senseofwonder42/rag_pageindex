# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

RAG system using a refactored fork of [PageIndex](https://github.com/VectifyAI/PageIndex) (vendored under [src/rag_pageindex/pageindex/](src/rag_pageindex/pageindex/)) as the retrieval backend. PageIndex is *reasoning-based* (no embeddings): an LLM extracts a hierarchical tree of pages from a PDF, then queries are routed down that tree. A LangGraph ReAct agent exposes this to users via tool calls.

## Commands

- `uv sync` — install deps after a checkout
- `uv add <pkg>` / `uv add <pkg> --dev` / `uv remove <pkg>` — only sanctioned way to change deps. **Never** `uv pip install`, **never** manually `uv venv`.
- `uv run start-app --pdf-path <file>` — index one PDF; `--input-dir <dir>` for batch mode
- `uv run start-app --input-dir <dir> --output-dir <dir>` — batch index with separate output
- `uv run langgraph dev` — start the LangGraph agent API server (for agent-chat-ui)
- `uv run pytest` — all tests; `uv run pytest tests/test_foo.py::test_bar` — single test
- `uv run ruff check src tests` — lint
- `uv run ruff check --fix && uv run ruff format` — autofix + format
- `uv run mypy` — type-check (configured to scan `src/` only)

## Toolchain conventions

- Python ≥3.12, line length **105**, indent 4.
- Ruff lint set: `E, W, F, I, B, PTH, ANN, ARG`. `ANN` means **all functions need type annotations**. `PTH` means **prefer `pathlib` over `os.path`**. Tests are exempt from `ANN`/`ARG`.
- mypy uses `pydantic.mypy`, `check_untyped_defs`, `no_implicit_optional`; excludes `tests/`.
- pytest auto-injects `ENVIRONMENT=test` and `LOG_LEVEL=INFO` via `pytest-env`.

## Configuration

Settings are loaded in priority order: init args → env vars → `.env` file → `config.yaml` → defaults.

- **[core/config.py](src/rag_pageindex/core/config.py)** — `Settings` (pydantic-settings). Access via the cached `settings` singleton. **All env vars must be fields on `Settings`** — never read `os.environ` directly.
- **`config.yaml`** (project root) — non-secret defaults (model name, pipeline tuning knobs, vision mode). Loaded via `YamlConfigSettingsSource`. Secrets (API keys) must never appear here; use `.env` instead.
- **`.env`** — secrets and local overrides. Copy `.env.example` to start.

## Architecture

`src/` layout, single package [src/rag_pageindex/](src/rag_pageindex/).

### Cross-cutting (`core/`)
- [core/config.py](src/rag_pageindex/core/config.py) — `Settings` singleton; see above.
- [core/logging.py](src/rag_pageindex/core/logging.py) — `setup_logging(level)` configures **loguru**. Use `from loguru import logger` everywhere; never `print()` for diagnostics.
- [core/constants.py](src/rag_pageindex/core/constants.py) — `PROJECT_ROOT: Path`.

### LangGraph agent (`agent/`)
A LangGraph ReAct agent that wraps the indexed documents as tools. Entry point for `langgraph dev` is defined in `langgraph.json` → `agent/graph.py:graph`.

- [agent/graph.py](src/rag_pageindex/agent/graph.py) — builds the compiled `StateGraph` using `create_react_agent` with `ChatOpenAI` (via `langchain-openai`).
- [agent/tools.py](src/rag_pageindex/agent/tools.py) — three async LangChain tools: `list_documents`, `get_document_structure`, `answer_from_pages`. Tools read index JSONs from `settings.pageindex_results_dir` and render PDF pages via `pdf/renderer.py`.
- [agent/vlm.py](src/rag_pageindex/agent/vlm.py) — `answer_with_images()`: sends rendered PNG pages to the VLM through the `LLMClient` protocol.
- [agent/tracing.py](src/rag_pageindex/agent/tracing.py) — returns Langfuse `CallbackHandler` list for LangChain if `settings.tracing_enabled`.

### PageIndex pipeline ([pageindex/](src/rag_pageindex/pageindex/))
Vendored from upstream commit `a51d97f` (MIT, Vectify AI), refactored into focused modules:

| Module | Owns |
|---|---|
| [llm/](src/rag_pageindex/pageindex/llm/) | `LLMClient` Protocol, `OpenAICompatibleClient` impl (httpx, no openai SDK), retry helper, factory, `TracingLLMClient` |
| [prompts/](src/rag_pageindex/pageindex/prompts/) | One `.j2` Jinja2 template per LLM call + `render()` loader |
| [pdf/reader.py](src/rag_pageindex/pageindex/pdf/reader.py) | PyMuPDF/PyPDF2 wrapper, per-page tokenization |
| [pdf/renderer.py](src/rag_pageindex/pageindex/pdf/renderer.py) | Render PDF pages to PNG bytes (PyMuPDF); wraps as `image_url` content parts |
| [toc/](src/rag_pageindex/pageindex/toc/) | TOC detection, parsing (TOC text→JSON), page-mapping, verification; `*_vlm.py` variants use images |
| [tree/](src/rag_pageindex/pageindex/tree/) | Pydantic tree types, builder, post-processing, summaries |
| [structured_responses.py](src/rag_pageindex/pageindex/structured_responses.py) | Pydantic models for structured LLM outputs (used with `acomplete_structured`) |
| [pipeline.py](src/rag_pageindex/pageindex/pipeline.py) | `page_index()` / `apage_index()` orchestrators |
| [client.py](src/rag_pageindex/pageindex/client.py) | `PageIndexClient`: in-memory workspace for multi-doc indexing + retrieval |
| [retrieve.py](src/rag_pageindex/pageindex/retrieve.py) | Lower-level retrieval helpers: `get_document`, `get_document_structure`, `get_page_content` |
| [observability.py](src/rag_pageindex/pageindex/observability.py) | `@observe` shim — no-op unless `tracing_enabled`; avoids Langfuse auth warnings on import |
| [json_extract.py](src/rag_pageindex/pageindex/json_extract.py) | Tolerant parser for model-emitted JSON (handles ```json fences, trailing commas) |

### Hard rules

1. **Pipeline LLM calls go through `LLMClient` protocol** ([pageindex/llm/protocol.py](src/rag_pageindex/pageindex/llm/protocol.py)) via `OpenAICompatibleClient` (raw `httpx`, not the `openai` SDK). The agent layer separately uses `langchain-openai`'s `ChatOpenAI` — that is the only place LangChain/OpenAI SDK is used. To add a new pipeline provider, implement the `LLMClient` protocol.
2. **No hardcoded prompts in Python.** Every pipeline prompt lives as a `.j2` file under [pageindex/prompts/](src/rag_pageindex/pageindex/prompts/) and is rendered via `prompts.render("name.j2", **ctx)`. The Jinja env uses `StrictUndefined`.
3. **Loguru only.** No `print()` / stdlib `logging`. Use `from loguru import logger`.
4. **In-flight pipeline state uses `TocItem` TypedDicts** ([tree/types.py](src/rag_pageindex/pageindex/tree/types.py)). The *output* uses plain dicts / pydantic models. Don't convert internal dicts to pydantic mid-pipeline.
5. **`@observe` from `observability.py`, not from `langfuse` directly.** The shim avoids noisy auth warnings when tracing is off.

### VLM / vision path

Controlled by `settings.pageindex_vision_mode` (`"off"` or `"fallback"`). When `"fallback"`, pages that fail text-based TOC verification are re-processed using rendered PNG images. Requires a vision-capable model. Key knobs: `pageindex_vision_dpi`, `pageindex_vision_fallback_threshold`, `pageindex_vision_max_images_per_call`.

### Indexed document store

The agent reads pre-built index JSONs from `settings.pageindex_results_dir` (default `examples/results/`). Each JSON must be an `IndexResult`-shaped dict (`doc_name` + `structure`). The source PDF must sit alongside its JSON for `answer_from_pages` to render pages.

### Observability (Langfuse)

Set `TRACING_ENABLED=true` + `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (+ optionally `LANGFUSE_HOST`). The pipeline uses `TracingLLMClient`; the agent binds a `CallbackHandler`. A self-hosted Langfuse stack is available under [deploy/langfuse/](deploy/langfuse/).

### Upstream reference
A read-only clone of upstream PageIndex at the vendored commit lives at `/tmp/pageindex_upstream/` (not in the repo). Re-clone with `git clone --depth 1 https://github.com/VectifyAI/PageIndex.git /tmp/pageindex_upstream` if missing.
