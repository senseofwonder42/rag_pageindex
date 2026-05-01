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
        self._llm = llm
        self._settings = settings
        self.workspace = Path(workspace).expanduser() if workspace else None
        if self.workspace:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.documents: dict[str, Any] = {}
        if self.workspace:
            self._load_workspace()

    def index(self, file_path: str | Path) -> str:
        """Index a PDF document and return its document_id."""
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
        return {
            "type": doc.get("type", ""),
            "doc_name": doc.get("doc_name", ""),
            "doc_description": doc.get("doc_description", ""),
            "path": doc.get("path", ""),
            "page_count": doc.get("page_count"),
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("corrupt {}: {}", path.name, e)
            return None

    def _save_doc(self, doc_id: str) -> None:
        doc = dict(self.documents[doc_id])
        doc.pop("pages", None)
        path = self.workspace / f"{doc_id}.json"  # type: ignore[operator]
        with path.open("w", encoding="utf-8") as f:  # type: ignore[union-attr]
            json.dump(doc, f, ensure_ascii=False, indent=2)
        self._save_meta(doc_id, self._make_meta_entry(doc))
        self.documents[doc_id].pop("structure", None)
        self.documents[doc_id].pop("pages", None)

    def _rebuild_meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        for p in self.workspace.glob("*.json"):  # type: ignore[union-attr]
            if p.name == META_INDEX:
                continue
            doc = self._read_json(p)
            if doc and isinstance(doc, dict):
                meta[p.stem] = self._make_meta_entry(doc)
        return meta

    def _read_meta(self) -> dict[str, Any] | None:
        meta = self._read_json(
            self.workspace / META_INDEX  # type: ignore[operator]
        )
        if meta is not None and not isinstance(meta, dict):
            logger.warning("{} is not a JSON object, ignoring", META_INDEX)
            return None
        return meta

    def _save_meta(self, doc_id: str, entry: dict[str, Any]) -> None:
        meta = self._read_meta() or self._rebuild_meta()
        meta[doc_id] = entry
        meta_path = self.workspace / META_INDEX  # type: ignore[operator]
        with meta_path.open("w", encoding="utf-8") as f:  # type: ignore[union-attr]
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _load_workspace(self) -> None:
        meta = self._read_meta()
        if meta is None:
            meta = self._rebuild_meta()
            if meta:
                logger.info(
                    "Loaded {} document(s) from workspace (rebuild)", len(meta)
                )
        for doc_id, entry in meta.items():
            doc = dict(entry, id=doc_id)
            if doc.get("path") and not Path(doc["path"]).is_absolute():
                doc["path"] = str(
                    (self.workspace / doc["path"]).resolve()  # type: ignore[operator]
                )
            self.documents[doc_id] = doc

    def _ensure_doc_loaded(self, doc_id: str) -> None:
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
        """Return document metadata JSON."""
        return get_document(self.documents, doc_id)

    def get_document_structure(self, doc_id: str) -> str:
        """Return document tree structure JSON (text fields stripped)."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_document_structure(self.documents, doc_id)

    def get_page_content(self, doc_id: str, pages: str) -> str:
        """Return page content for `pages` (e.g. '5-7', '3,8', '12')."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_page_content(self.documents, doc_id, pages)
