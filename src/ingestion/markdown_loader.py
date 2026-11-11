"""Native Markdown document loader with SHA-256 fingerprinting, page marker parsing, and DLQ handling.

Mengimplementasikan kebutuhan PRD:
- FR-1.1, FR-1.2, FR-1.3: Pembacaan langsung berkas .md, SHA-256 fingerprinting, dan lineage metadata.
- FR-2.1, FR-2.2, FR-2.3, FR-2.4: Parsing native UTF-8, ekstraksi tag halaman ([Hal: X], <!-- page: X -->),
  Quality Gates (>= 50 chars), dan isolasi berkas gagal ke Dead-Letter Queue (data/dlq/).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import Settings, get_settings
from src.ingestion.sanitizer import MarkdownSanitizer, QualityGateError


@dataclass
class MarkdownDocument:
    """Representasi dokumen Markdown tervalidasi beserta metadata lineage & sitasi."""

    document_name: str
    content: str
    doc_hash: str
    source_path: str
    file_size_bytes: int
    line_count: int
    page_numbers: List[int] = field(default_factory=lambda: [1])
    ingested_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarkdownDocumentLoader:
    """Loader native untuk berkas Markdown (.md) di data/processed atau data/raw."""

    def __init__(
        self,
        source_dir: Optional[Path] = None,
        dlq_dir: Optional[Path] = None,
        dlq_error_log: Optional[Path] = None,
        settings_obj: Optional[Settings] = None,
    ) -> None:
        self.settings = settings_obj or get_settings()
        self.source_dir = Path(
            source_dir or self.settings.paths.data_processed_dir
        ).resolve()
        self.dlq_dir = Path(dlq_dir or self.settings.paths.data_dlq_dir).resolve()
        self.dlq_error_log = Path(
            dlq_error_log or self.settings.paths.dlq_error_log
        ).resolve()
        self.sanitizer = MarkdownSanitizer(
            min_file_size_chars=self.settings.ingestion.min_file_size_chars
        )
        self.page_patterns = [
            re.compile(pat) for pat in self.settings.ingestion.page_tag_patterns
        ]

        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.dlq_dir.mkdir(parents=True, exist_ok=True)
        self.dlq_error_log.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def extract_page_numbers(self, content: str) -> List[int]:
        """Mendeteksi nomor halaman dari tag [Hal: X] atau <!-- page: X --> (FR-2.2)."""
        pages_found = set()
        for regex in self.page_patterns:
            for match in regex.finditer(content):
                try:
                    pages_found.add(int(match.group(1)))
                except (ValueError, IndexError):
                    continue
        return sorted(pages_found) if pages_found else [1]

    def _route_to_dlq(self, file_path: Path, doc_hash: str, reason: str) -> None:
        dest = self.dlq_dir / file_path.name
        if file_path.exists() and file_path.resolve() != dest.resolve():
            shutil.copy2(file_path, dest)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "file_name": file_path.name,
            "source_path": str(file_path),
            "dlq_path": str(dest),
            "doc_hash": doc_hash,
            "reason": reason,
        }
        with open(self.dlq_error_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_file(self, file_path: Path) -> Optional[MarkdownDocument]:
        """Membaca dan memvalidasi satu berkas .md sesuai Quality Gates (FR-2.1 - FR-2.4)."""
        doc_hash = self.compute_sha256(file_path)
        try:
            raw_bytes = file_path.read_bytes()
            if b"\x00" in raw_bytes:
                raise QualityGateError(
                    "Berkas .md mengandung null byte (\\x00) / biner korup."
                )
            content = raw_bytes.decode(self.settings.ingestion.encoding)
            is_valid, reason = self.sanitizer.validate_quality_gates(content)
            if not is_valid:
                raise QualityGateError(reason)

            pages = self.extract_page_numbers(content)
            now_iso = datetime.now(timezone.utc).isoformat()
            file_size = file_path.stat().st_size
            line_count = len(content.splitlines())

            metadata = {
                "document_name": file_path.name,
                "doc_hash": doc_hash,
                "source": str(file_path),
                "file_size_bytes": file_size,
                "line_count": line_count,
                "page_number": pages[0],
                "total_pages": len(pages),
                "ingested_at": now_iso,
            }
            return MarkdownDocument(
                document_name=file_path.name,
                content=content,
                doc_hash=doc_hash,
                source_path=str(file_path),
                file_size_bytes=file_size,
                line_count=line_count,
                page_numbers=pages,
                ingested_at=now_iso,
                metadata=metadata,
            )
        except Exception as exc:
            self._route_to_dlq(
                file_path=file_path,
                doc_hash=doc_hash,
                reason=f"{type(exc).__name__}: {exc}",
            )
            return None

    def load_directory(self, target_dir: Optional[Path] = None) -> List[MarkdownDocument]:
        """Memuat seluruh berkas *.md yang valid dari direktori target."""
        scan_dir = Path(target_dir or self.source_dir).resolve()
        if not scan_dir.exists():
            return []
        md_files = sorted(
            [
                p
                for p in scan_dir.iterdir()
                if p.is_file()
                and not p.name.startswith(".")
                and p.suffix.lower() in self.settings.ingestion.supported_extensions
            ],
            key=lambda p: p.name.lower(),
        )
        docs: List[MarkdownDocument] = []
        for f in md_files:
            doc = self.load_file(f)
            if doc is not None:
                docs.append(doc)
        return docs
