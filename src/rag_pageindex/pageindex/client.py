from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import PyPDF2
from loguru import logger

from rag_pageindex.pageindex.llm.protocol import LLMClient
from rag_pageindex.pageindex.pipeline import page_index
from rag_pageindex.pageindex.retrieve import (
    get_document,
    get_document_structure,
    get_page_content,
)

if TYPE_CHECKING:
    from rag_pageindex.core.config import Settings

META_INDEX = "_meta.json"


class PageIndexClient:
    """Index and retrieve content from PDF documents.

    Flow: index() → get_document() / get_document_structure() /
    get_page_content()
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        workspace: str | Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize a PageIndexClient for document indexing and retrieval.

        Args:
            llm: LLM client implementation to use for indexing.
            workspace: Optional workspace directory for persisting indexed documents.
            settings: Optional Settings instance; uses default if None.
        """
        self._llm = llm
        self._settings = settings
        self.workspace = Path(workspace).expanduser() if workspace else None
        if self.workspace:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.documents: dict[str, Any] = {}
        if self.workspace:
            self._load_workspace()

    def index(self, file_path: str | Path) -> str:
        """Index a PDF document using the PageIndex pipeline.

        Args:
            file_path: Path to the PDF file to index.

        Returns:
            Unique document ID for the indexed document.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
        """
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        doc_id = str(uuid.uuid4())
        logger.info("Indexing PDF: {}", path)

        result = page_index(str(path), llm=self._llm, settings=self._settings)

        pages: list[dict[str, Any]] = []
        with path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages, 1):
                pages.append({"page": i, "content": page.extract_text() or ""})

        self.documents[doc_id] = {
            "id": doc_id,
            "type": "pdf",
            "path": str(path),
            "doc_name": result.get("doc_name", ""),
            "doc_description": result.get("doc_description", ""),
            "page_count": len(pages),
            "structure": result["structure"],
            "pages": pages,
        }
        logger.info("Indexed: {} → {}", path.name, doc_id)

        if self.workspace:
            self._save_doc(doc_id)
        return doc_id

    # ── Workspace persistence ─────────────────────────────────────────────

    @staticmethod
    def _make_meta_entry(doc: dict[str, Any]) -> dict[str, Any]:
        """Extract metadata fields from a document dict.

        Args:
            doc: Full document dict with all indexed content.

        Returns:
            Metadata-only dict suitable for _meta.json index.
        """
        return {
            "type": doc.get("type", ""),
            "doc_name": doc.get("doc_name", ""),
            "doc_description": doc.get("doc_description", ""),
            "path": doc.get("path", ""),
            "page_count": doc.get("page_count"),
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        """Read and parse a JSON file from disk.

        Args:
            path: Path to the JSON file.

        Returns:
            Parsed JSON dict, or None on read/decode error (logged as warning).
        """
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("corrupt {}: {}", path.name, e)
            return None

    def _save_doc(self, doc_id: str) -> None:
        """Save indexed document to workspace and update metadata index.

        Persists the document dict (without full page text) and updates
        the workspace metadata file to include this document.

        Args:
            doc_id: Document ID to save.
        """
        doc = dict(self.documents[doc_id])
        doc.pop("pages", None)
        path = self.workspace / f"{doc_id}.json"  # type: ignore[operator]
        with path.open("w", encoding="utf-8") as f:  # type: ignore[union-attr]
            json.dump(doc, f, ensure_ascii=False, indent=2)
        self._save_meta(doc_id, self._make_meta_entry(doc))
        self.documents[doc_id].pop("structure", None)
        self.documents[doc_id].pop("pages", None)

    def _rebuild_meta(self) -> dict[str, Any]:
        """Reconstruct metadata index by scanning workspace JSON files.

        Returns:
            Metadata dict keyed by document ID.
        """
        meta: dict[str, Any] = {}
        for p in self.workspace.glob("*.json"):  # type: ignore[union-attr]
            if p.name == META_INDEX:
                continue
            doc = self._read_json(p)
            if doc and isinstance(doc, dict):
                meta[p.stem] = self._make_meta_entry(doc)
        return meta

    def _read_meta(self) -> dict[str, Any] | None:
        """Read the workspace metadata index from _meta.json.

        Returns:
            Metadata dict or None if file doesn't exist or is invalid.
        """
        meta = self._read_json(
            self.workspace / META_INDEX  # type: ignore[operator]
        )
        if meta is not None and not isinstance(meta, dict):
            logger.warning("{} is not a JSON object, ignoring", META_INDEX)
            return None
        return meta

    def _save_meta(self, doc_id: str, entry: dict[str, Any]) -> None:
        """Update workspace metadata index with a document entry.

        Reads current metadata, inserts or updates the entry, and writes
        the updated index back to _meta.json.

        Args:
            doc_id: Document ID.
            entry: Metadata dict for this document.
        """
        meta = self._read_meta() or self._rebuild_meta()
        meta[doc_id] = entry
        meta_path = self.workspace / META_INDEX  # type: ignore[operator]
        with meta_path.open("w", encoding="utf-8") as f:  # type: ignore[union-attr]
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _load_workspace(self) -> None:
        """Load all documents from workspace metadata into memory.

        Called during __init__ to populate self.documents with indexed
        documents from the workspace directory.
        """
        meta = self._read_meta()
        if meta is None:
            meta = self._rebuild_meta()
            if meta:
                logger.info("Loaded {} document(s) from workspace (rebuild)", len(meta))
        for doc_id, entry in meta.items():
            doc = dict(entry, id=doc_id)
            if doc.get("path") and not Path(doc["path"]).is_absolute():
                doc["path"] = str(
                    (self.workspace / doc["path"]).resolve()  # type: ignore[operator]
                )
            self.documents[doc_id] = doc

    def _ensure_doc_loaded(self, doc_id: str) -> None:
        """Load full document data (structure, pages) from disk if needed.

        Metadata-only docs loaded from _meta.json are augmented with full
        content by reading the per-document JSON file.

        Args:
            doc_id: Document ID to load.
        """
        doc = self.documents.get(doc_id)
        if not doc or doc.get("structure") is not None:
            return
        full = self._read_json(
            self.workspace / f"{doc_id}.json"  # type: ignore[operator]
        )
        if not full:
            return
        doc["structure"] = full.get("structure", [])
        if full.get("pages"):
            doc["pages"] = full["pages"]

    # ── Public API ────────────────────────────────────────────────────────

    def get_document(self, doc_id: str) -> str:
        """Get document metadata as JSON.

        Args:
            doc_id: Document ID.

        Returns:
            JSON string with document name, description, type, status, page count.
        """
        return get_document(self.documents, doc_id)

    def get_document_structure(self, doc_id: str) -> str:
        """Get document tree structure as JSON (text content stripped).

        Args:
            doc_id: Document ID.

        Returns:
            JSON string with the hierarchical tree structure.
        """
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_document_structure(self.documents, doc_id)

    def get_page_content(self, doc_id: str, pages: str) -> str:
        """Get content of specific pages from a document.

        Args:
            doc_id: Document ID.
            pages: Page specification: single page ('3'), range ('5-7'),
                or comma-separated ('3,8,12').

        Returns:
            JSON string with list of {page, content} dicts.
        """
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_page_content(self.documents, doc_id, pages)
