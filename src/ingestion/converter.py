"""Multi-format raw document to Markdown (.md) converter with incremental lineage, DLQ, and audit logging.

Mengimplementasikan kebutuhan PRD:
- FR-1.1: Konversi ke format Markdown (*.md) standar untuk pipeline ingestion.
- FR-1.2: SHA-256 fingerprinting & Incremental Idempotency (berkas yang sudah diproses dilewati otomatis).
- FR-1.3: Lineage metadata capture (document_name, ukuran berkas, jumlah baris, jumlah karakter, tanggal proses).
- FR-2.2: Preservasi struktur hirarki (#, ##, ###) dan penanda halaman (<!-- page: X -->, [Hal: X]).
- FR-2.3: Quality Gates (validasi UTF-8 tanpa karakter null/korup & minimal 50 karakter terbaca).
- FR-2.4: Dead-Letter Queue (DLQ) ke data/dlq/ dan pencatatan data/dlq/error.log tanpa menghentikan batch.
- Logging: Pencatatan lengkap riwayat konversi di folder logs/ (logs/conversion.log & logs/conversion_history.jsonl).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import struct
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import Settings, get_settings
from src.ingestion.sanitizer import MarkdownSanitizer, QualityGateError

logger = logging.getLogger(__name__)

# Daftar ekstensi berkas mentah di data/raw yang didukung untuk dikonversi ke .md
CONVERTIBLE_EXTENSIONS = {
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".txt",
    ".pdf",
    ".md",
}


@dataclass
class LineageMetadata:
    """Representasi metadata lineage dokumen hasil konversi (FR-1.2 & FR-1.3)."""

    document_name: str
    source_file: str
    source_path: str
    output_path: str
    source_hash: str
    md_hash: str
    source_size_bytes: int
    md_size_bytes: int
    line_count: int
    char_count: int
    readable_char_count: int
    page_count: int
    converted_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversionRecord:
    """Catatan hasil pemrosesan satu berkas dalam batch konversi."""

    run_id: str
    timestamp: str
    source_file: str
    output_file: Optional[str]
    status: str  # CONVERTED | SKIPPED | DLQ_FAILED
    source_hash: str
    md_hash: Optional[str] = None
    page_count: int = 0
    line_count: int = 0
    char_count: int = 0
    duration_ms: float = 0.0
    reason: str = "OK"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchConversionSummary:
    """Ringkasan hasil eksekusi batch konversi."""

    run_id: str
    started_at: str
    completed_at: str
    input_dir: str
    output_dir: str
    total_scanned: int = 0
    converted_count: int = 0
    skipped_count: int = 0
    dlq_failed_count: int = 0
    total_duration_ms: float = 0.0
    records: List[ConversionRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["records"] = [r.to_dict() for r in self.records]
        return data


def _safe_unpack_sst_table(
    datatab: List[bytes], nstrings: int
) -> Tuple[List[str], Dict[int, List[Tuple[int, int]]]]:
    """Unpacker Shared String Table (SST) BIFF8 yang toleran terhadap EOF dini pada berkas .xls legacy."""
    datainx = 0
    ndatas = len(datatab)
    if ndatas == 0:
        return [], {}
    data = datatab[0]
    datalen = len(data)
    pos = 8
    strings: List[str] = []
    richtext_runs: Dict[int, List[Tuple[int, int]]] = {}
    local_unpack = struct.unpack

    for _unused_i in range(nstrings):
        if datainx >= ndatas or pos + 3 > datalen:
            strings.append("")
            continue
        nchars = local_unpack("<H", data[pos : pos + 2])[0]
        pos += 2
        options = data[pos]
        pos += 1
        rtcount = 0
        phosz = 0
        if options & 0x08:
            if pos + 2 <= datalen:
                rtcount = local_unpack("<H", data[pos : pos + 2])[0]
            pos += 2
        if options & 0x04:
            if pos + 4 <= datalen:
                phosz = local_unpack("<i", data[pos : pos + 4])[0]
            pos += 4
        accstrg = ""
        charsgot = 0
        while True:
            charsneed = nchars - charsgot
            if options & 0x01:
                charsavail = min((datalen - pos) >> 1, charsneed)
                if charsavail < 0:
                    charsavail = 0
                rawstrg = data[pos : pos + 2 * charsavail]
                accstrg += rawstrg.decode("utf_16_le", errors="replace")
                pos += 2 * charsavail
            else:
                charsavail = min(datalen - pos, charsneed)
                if charsavail < 0:
                    charsavail = 0
                rawstrg = data[pos : pos + charsavail]
                accstrg += rawstrg.decode("cp1252", errors="replace")
                pos += charsavail
            charsgot += charsavail
            if charsgot >= nchars:
                break
            datainx += 1
            if datainx >= ndatas:
                break
            data = datatab[datainx]
            datalen = len(data)
            options = data[0]
            pos = 1

        if rtcount:
            runs: List[Tuple[int, int]] = []
            for _ in range(rtcount):
                if pos == datalen:
                    pos = 0
                    datainx += 1
                    if datainx >= ndatas:
                        break
                    data = datatab[datainx]
                    datalen = len(data)
                if pos + 4 <= datalen:
                    runs.append(local_unpack("<HH", data[pos : pos + 4]))
                pos += 4
            richtext_runs[len(strings)] = runs

        pos += phosz
        if pos >= datalen:
            pos = pos - datalen
            datainx += 1
            if datainx < ndatas:
                data = datatab[datainx]
                datalen = len(data)
        strings.append(accstrg)

    return strings, richtext_runs


class RawToMarkdownConverter:
    """Konverter dokumen mentah (data/raw) menjadi berkas Markdown (.md) di data/processed."""

    def __init__(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        dlq_dir: Optional[Path] = None,
        dlq_error_log: Optional[Path] = None,
        lineage_cache_path: Optional[Path] = None,
        conversion_log_path: Optional[Path] = None,
        conversion_history_path: Optional[Path] = None,
        settings_obj: Optional[Settings] = None,
    ) -> None:
        self.settings = settings_obj or get_settings()
        self.input_dir = Path(input_dir or self.settings.paths.data_raw_dir).resolve()
        self.output_dir = Path(
            output_dir or self.settings.paths.data_processed_dir
        ).resolve()
        self.dlq_dir = Path(dlq_dir or self.settings.paths.data_dlq_dir).resolve()
        self.dlq_error_log = Path(
            dlq_error_log or self.settings.paths.dlq_error_log
        ).resolve()
        self.lineage_cache_path = Path(
            lineage_cache_path or self.settings.paths.lineage_cache_path
        ).resolve()
        self.conversion_log_path = Path(
            conversion_log_path or self.settings.paths.conversion_log_path
        ).resolve()
        self.conversion_history_path = Path(
            conversion_history_path or self.settings.paths.conversion_history_path
        ).resolve()

        self.sanitizer = MarkdownSanitizer(
            min_file_size_chars=self.settings.ingestion.min_file_size_chars
        )

        # Pastikan seluruh direktori tersedia
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dlq_dir.mkdir(parents=True, exist_ok=True)
        self.dlq_error_log.parent.mkdir(parents=True, exist_ok=True)
        self.lineage_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.conversion_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.conversion_history_path.parent.mkdir(parents=True, exist_ok=True)

        self.lineage_cache: Dict[str, Dict[str, Any]] = self._load_lineage_cache()

    # ------------------------------------------------------------------
    # Hashing & Lineage Cache Management (FR-1.2 & FR-1.3)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Menghitung hash kriptografi SHA-256 dari sebuah berkas."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def compute_text_sha256(text: str) -> str:
        """Menghitung hash kriptografi SHA-256 dari string teks UTF-8."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_lineage_cache(self) -> Dict[str, Dict[str, Any]]:
        """Memuat cache lineage dari data/processed/lineage_cache.json."""
        if not self.lineage_cache_path.exists():
            return {}
        try:
            with open(self.lineage_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning(f"Gagal membaca lineage cache ({self.lineage_cache_path}): {exc}")
        return {}

    def _save_lineage_cache(self) -> None:
        """Menyimpan cache lineage secara atomik ke data/processed/lineage_cache.json."""
        tmp_path = self.lineage_cache_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.lineage_cache, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.lineage_cache_path)

    def should_process_file(
        self, file_path: Path, source_hash: str, force: bool = False
    ) -> Tuple[bool, str]:
        """Menentukan apakah sebuah berkas di data/raw perlu dikonversi atau dilewati (Incremental Mode).

        Jika berkas sudah pernah dikonversi (tercatat di lineage_cache dengan source_hash yang sama
        dan berkas .md keluaran sudah ada di data/processed), maka berkas dilewati (SKIPPED).
        """
        if force:
            return True, "FORCE_RECONVERT"

        target_md_path = self.output_dir / f"{file_path.stem}.md"
        if not target_md_path.exists():
            return True, "NEW_FILE_NO_OUTPUT"

        cached_entry = self.lineage_cache.get(file_path.name)
        if cached_entry and cached_entry.get("source_hash") == source_hash:
            return False, "SKIPPED_ALREADY_PROCESSED"

        # Jika file .md sudah ada di data/processed namun belum tercatat di lineage_cache,
        # validasi apakah .md tersebut valid; jika valid dan source tidak berubah, catat ke cache dan skip
        if cached_entry is None and target_md_path.exists():
            try:
                existing_md = target_md_path.read_text(encoding="utf-8")
                is_valid, _ = self.sanitizer.validate_quality_gates(existing_md)
                if is_valid:
                    # Backfill lineage cache agar konsisten
                    self._record_lineage(file_path, target_md_path, source_hash, existing_md, page_count=1)
                    return False, "SKIPPED_EXISTING_VALID_MD"
            except Exception:
                pass

        return True, "MODIFIED_SOURCE_HASH"

    def _record_lineage(
        self,
        source_path: Path,
        output_path: Path,
        source_hash: str,
        md_content: str,
        page_count: int,
    ) -> LineageMetadata:
        """Merekam metadata lineage dokumen ke cache dan menyimpannya ke disk."""
        md_hash = self.compute_text_sha256(md_content)
        readable_chars = len(self.sanitizer.extract_readable_chars(md_content))
        now_iso = datetime.now(timezone.utc).isoformat()

        meta = LineageMetadata(
            document_name=output_path.name,
            source_file=source_path.name,
            source_path=str(source_path),
            output_path=str(output_path),
            source_hash=source_hash,
            md_hash=md_hash,
            source_size_bytes=source_path.stat().st_size,
            md_size_bytes=len(md_content.encode("utf-8")),
            line_count=len(md_content.splitlines()),
            char_count=len(md_content),
            readable_char_count=readable_chars,
            page_count=page_count,
            converted_at=now_iso,
        )
        self.lineage_cache[source_path.name] = meta.to_dict()
        self._save_lineage_cache()
        return meta

    # ------------------------------------------------------------------
    # Dead-Letter Queue (DLQ) & Logging (FR-2.4 & Audit Logs)
    # ------------------------------------------------------------------
    def _route_to_dlq(
        self, file_path: Path, source_hash: str, reason: str, tb_text: str = ""
    ) -> None:
        """Memindahkan/menyalin berkas gagal ke data/dlq/ dan mencatat ke data/dlq/error.log (FR-2.4)."""
        try:
            dest_dlq_file = self.dlq_dir / file_path.name
            if file_path.exists() and file_path.resolve() != dest_dlq_file.resolve():
                shutil.copy2(file_path, dest_dlq_file)
        except Exception as copy_exc:
            logger.error(f"Gagal menyalin berkas korup ke DLQ ({file_path.name}): {copy_exc}")

        error_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "file_name": file_path.name,
            "source_path": str(file_path),
            "dlq_path": str(self.dlq_dir / file_path.name),
            "source_hash": source_hash,
            "reason": reason,
            "traceback": tb_text.strip() if tb_text else None,
        }
        with open(self.dlq_error_log, "a", encoding="utf-8") as err_f:
            err_f.write(json.dumps(error_entry, ensure_ascii=False) + "\n")

    def _write_conversion_log_line(self, message: str) -> None:
        """Menulis baris log terformat ke logs/conversion.log."""
        ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")
        log_line = f"[{ts}] {message}\n"
        with open(self.conversion_log_path, "a", encoding="utf-8") as log_f:
            log_f.write(log_line)

    def _write_conversion_history_event(self, event_data: Dict[str, Any]) -> None:
        """Menulis event JSONL terstruktur ke logs/conversion_history.jsonl."""
        with open(self.conversion_history_path, "a", encoding="utf-8") as hist_f:
            hist_f.write(json.dumps(event_data, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Helper Konversi Tabel ke Format Markdown GFM
    # ------------------------------------------------------------------
    @staticmethod
    def _rows_to_markdown_table(rows_data: List[List[str]]) -> str:
        """Mengonversi matriks baris/kolom menjadi tabel GitHub Flavored Markdown (GFM) yang rapi."""
        if not rows_data:
            return ""

        # Bersihkan setiap sel dan cari indeks kolom terakhir yang tidak kosong
        cleaned_matrix: List[List[str]] = []
        max_non_empty_col = -1

        for row in rows_data:
            cleaned_row: List[str] = []
            for c_idx, cell in enumerate(row):
                val = str(cell).strip() if cell is not None else ""
                val = re.sub(r"\s+", " ", val).replace("|", "/")
                cleaned_row.append(val)
                if val != "" and c_idx > max_non_empty_col:
                    max_non_empty_col = c_idx
            if any(cleaned_row):
                cleaned_matrix.append(cleaned_row)

        if not cleaned_matrix or max_non_empty_col < 0:
            return ""

        num_cols = max_non_empty_col + 1
        padded_rows: List[List[str]] = []
        for r in cleaned_matrix:
            trimmed = r[:num_cols]
            padded_rows.append(trimmed + [""] * (num_cols - len(trimmed)))

        header_row = [
            h if h else f"Kolom_{idx + 1}" for idx, h in enumerate(padded_rows[0])
        ]
        header_line = "| " + " | ".join(header_row) + " |"
        sep_line = "| " + " | ".join(["---"] * num_cols) + " |"
        body_lines = ["| " + " | ".join(r) + " |" for r in padded_rows[1:]]

        return "\n".join([header_line, sep_line] + body_lines)

    # ------------------------------------------------------------------
    # Format-Specific Parsers (.doc, .docx, .xlsx, .xls, .txt, .pdf, .md)
    # ------------------------------------------------------------------
    @classmethod
    def _extract_doc(cls, file_path: Path) -> List[Tuple[int, str]]:
        """Mengekstrak teks dan tabel dari berkas MS Word 97-2003 OLE2 (.doc)."""
        import olefile

        if not olefile.isOleFile(str(file_path)):
            raise ValueError(f"Berkas {file_path.name} bukan dokumen OLE2 (.doc) yang valid.")

        ole = olefile.OleFileIO(str(file_path))
        try:
            if not ole.exists("WordDocument"):
                raise ValueError(f"Stream 'WordDocument' tidak ditemukan di {file_path.name}.")
            data = ole.openstream("WordDocument").read()
        finally:
            ole.close()

        raw_text = ""
        if len(data) >= 512:
            w_ident = struct.unpack("<H", data[0:2])[0]
            if w_ident in (0xA5EC, 0xA5DC):
                fc_min = struct.unpack("<I", data[0x0018:0x001C])[0]
                ccp_text = struct.unpack("<I", data[0x004C:0x0050])[0]
                if 0 < fc_min < len(data) and 0 < ccp_text <= len(data) - fc_min:
                    text_slice = data[fc_min : fc_min + ccp_text]
                    raw_text = text_slice.decode("cp1252", errors="ignore")

        if not raw_text.strip():
            # Fallback ekstraksi segmen tekstual dari stream biner
            raw_segments = re.findall(b"[\x20-\x7e\r\n\t]{6,}", data)
            valid_lines = []
            for seg in raw_segments:
                line = seg.decode("cp1252", errors="ignore").strip()
                if any(
                    sig in line
                    for sig in ["Normal.dot", "Times New Roman", "bjbj", "IEND", "JFIF", "MSWordDoc"]
                ):
                    continue
                if re.search(r"[a-zA-Z]{3,}", line):
                    valid_lines.append(line)
            raw_text = "\n".join(valid_lines)

        if not raw_text.strip():
            return []

        # Pisahkan berdasarkan karakter Page Break (\x0c) bila ada
        raw_pages = raw_text.split("\x0c")
        extracted_pages: List[Tuple[int, str]] = []

        for idx, raw_page in enumerate(raw_pages, start=1):
            page_str = cls._convert_doc_page_stream_to_markdown(raw_page)
            if page_str.strip():
                extracted_pages.append((idx, page_str))

        return extracted_pages

    @classmethod
    def _convert_doc_page_stream_to_markdown(cls, raw_page: str) -> str:
        """Merekonstruksi paragraf dan penanda sel tabel Word (\x07) menjadi teks & tabel Markdown."""
        # Normalisasi line break internal Word
        s = raw_page.replace("\r\n", "\r").replace("\n", "\r").replace("\x0b", "\n")
        # Hapus karakter kontrol gambar/objek (\x01, \x08, \x13, \x14, \x15)
        s = re.sub(r"[\x00-\x06\x08\x0e-\x1f]", "", s)

        # Di MS-DOC, akhir baris tabel ditandai oleh \x07\x07 (atau \r\x07\r\x07),
        # sedangkan batas antar sel dalam satu baris tabel ditandai oleh satu \x07.
        paragraphs = s.split("\r")
        output_blocks: List[str] = []
        current_table_rows: List[List[str]] = []
        current_row_cells: List[str] = []

        def flush_table() -> None:
            nonlocal current_table_rows, current_row_cells
            if current_row_cells:
                if any(c.strip() for c in current_row_cells):
                    current_table_rows.append(current_row_cells)
                current_row_cells = []
            if current_table_rows:
                md_tbl = cls._rows_to_markdown_table(current_table_rows)
                if md_tbl:
                    output_blocks.append(md_tbl)
                current_table_rows = []

        for token in paragraphs:
            if "\x07" in token:
                parts = token.split("\x07")
                # Setiap elemen sebelum \x07 terakhir adalah isi sel atau penutup baris
                for p_idx, cell_part in enumerate(parts[:-1]):
                    cell_clean = cell_part.strip()
                    if p_idx > 0 and cell_clean == "" and current_row_cells:
                        # \x07 kedua berturut-turut menandakan akhir baris tabel Word
                        if any(c.strip() for c in current_row_cells):
                            current_table_rows.append(current_row_cells)
                        current_row_cells = []
                    else:
                        current_row_cells.append(cell_clean)
                trailing = parts[-1].strip()
                if trailing:
                    current_row_cells.append(trailing)
            else:
                line_clean = token.strip()
                if not line_clean and current_row_cells:
                    continue
                flush_table()
                if line_clean:
                    output_blocks.append(line_clean)

        flush_table()
        return "\n\n".join(output_blocks)

    @classmethod
    def _extract_docx(cls, file_path: Path) -> List[Tuple[int, str]]:
        """Mengekstrak dokumen MS Word (.docx) secara sekuensial beserta style heading & tabel."""
        import docx
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        doc = docx.Document(str(file_path))
        content_parts: List[str] = []

        for child in doc.element.body:
            if isinstance(child, CT_P):
                para = Paragraph(child, doc)
                p_text = para.text.strip()
                if not p_text:
                    continue
                style_name = (para.style.name if para.style else "").lower()
                if "heading 1" in style_name:
                    content_parts.append(f"## {p_text}")
                elif "heading 2" in style_name or "heading 3" in style_name:
                    content_parts.append(f"### {p_text}")
                else:
                    content_parts.append(p_text)
            elif isinstance(child, CT_Tbl):
                tbl = Table(child, doc)
                rows_data: List[List[str]] = []
                for row in tbl.rows:
                    row_vals = [
                        re.sub(r"\s+", " ", cell.text.strip()) for cell in row.cells
                    ]
                    # Deduplikasi sel merged horizontal yang berulang berurutan
                    deduped: List[str] = []
                    for val in row_vals:
                        if not deduped or val != deduped[-1] or val == "":
                            deduped.append(val)
                    if any(deduped):
                        rows_data.append(deduped)
                md_table = cls._rows_to_markdown_table(rows_data)
                if md_table:
                    content_parts.append(md_table)

        full_text = "\n\n".join(content_parts).strip()
        if not full_text:
            return []
        return [(1, full_text)]

    @classmethod
    def _format_spreadsheet_sheet(
        cls, sheet_name: str, raw_rows: List[List[str]]
    ) -> str:
        """Memformat satu lembar kerja Excel menjadi bagian Markdown (Tabel + Narasi Terstruktur)."""
        if not raw_rows:
            return ""

        md_table = cls._rows_to_markdown_table(raw_rows)

        # Cari baris header terbaik (baris pertama yang memiliki >= 2 kolom terisi, atau baris 0)
        header_idx = 0
        for idx, row in enumerate(raw_rows[:10]):
            non_empty = [c for c in row if c.strip()]
            if len(non_empty) >= 2:
                header_idx = idx
                break

        header_row = raw_rows[header_idx]
        col_names = [
            h.strip() if h and h.strip() else f"Kolom_{i + 1}"
            for i, h in enumerate(header_row)
        ]

        narrative_lines: List[str] = []
        data_rows = raw_rows[header_idx + 1 :] if header_idx + 1 < len(raw_rows) else []
        for r_idx, row_data in enumerate(data_rows, start=1):
            pairs: List[str] = []
            for c_idx, val in enumerate(row_data):
                val_clean = val.strip()
                if val_clean:
                    col_label = (
                        col_names[c_idx]
                        if c_idx < len(col_names)
                        else f"Kolom_{c_idx + 1}"
                    )
                    pairs.append(f"{col_label}: {val_clean}")
            if pairs:
                narrative_lines.append(f"- Baris {r_idx}: " + " | ".join(pairs))

        parts = [
            f"## Lembar Kerja: {sheet_name}",
            f"### Struktur Tabel\n\n{md_table}",
        ]
        if narrative_lines:
            narrative_block = "\n".join(narrative_lines)
            parts.append(f"### Representasi Terstruktur\n\n{narrative_block}")

        return "\n\n".join(parts).strip()

    @classmethod
    def _extract_xlsx(cls, file_path: Path) -> List[Tuple[int, str]]:
        """Mengekstrak seluruh lembar kerja dari berkas MS Excel (.xlsx)."""
        import openpyxl

        wb = openpyxl.load_workbook(str(file_path), data_only=True, read_only=True)
        pages: List[Tuple[int, str]] = []
        try:
            for s_idx, sheet_name in enumerate(wb.sheetnames, start=1):
                sheet = wb[sheet_name]
                raw_rows: List[List[str]] = []
                for row in sheet.iter_rows(values_only=True):
                    if any(val is not None and str(val).strip() != "" for val in row):
                        row_vals = [
                            str(val).strip() if val is not None else "" for val in row
                        ]
                        raw_rows.append(row_vals)
                if not raw_rows:
                    continue
                sheet_md = cls._format_spreadsheet_sheet(sheet_name, raw_rows)
                if sheet_md:
                    pages.append((s_idx, sheet_md))
        finally:
            wb.close()
        return pages

    @classmethod
    def _extract_xls(cls, file_path: Path) -> List[Tuple[int, str]]:
        """Mengekstrak seluruh lembar kerja dari berkas MS Excel 97-2003 (.xls) dengan fallback SST aman."""
        import xlrd
        import xlrd.book

        try:
            wb = xlrd.open_workbook(str(file_path), formatting_info=False)
        except (AssertionError, IndexError):
            # Gunakan safe SST unpacker bila header uniquestrings BIFF8 tidak sinkron
            orig_unpack = xlrd.book.unpack_SST_table
            try:
                xlrd.book.unpack_SST_table = _safe_unpack_sst_table
                wb = xlrd.open_workbook(str(file_path), formatting_info=False)
            finally:
                xlrd.book.unpack_SST_table = orig_unpack

        pages: List[Tuple[int, str]] = []
        for s_idx in range(wb.nsheets):
            sheet = wb.sheet_by_index(s_idx)
            raw_rows: List[List[str]] = []
            for rx in range(sheet.nrows):
                row_vals = sheet.row_values(rx)
                if any(v is not None and str(v).strip() != "" for v in row_vals):
                    cleaned_vals: List[str] = []
                    for v in row_vals:
                        if v is None:
                            cleaned_vals.append("")
                        elif isinstance(v, float) and v.is_integer():
                            cleaned_vals.append(str(int(v)))
                        else:
                            cleaned_vals.append(str(v).strip())
                    raw_rows.append(cleaned_vals)
            if not raw_rows:
                continue
            sheet_md = cls._format_spreadsheet_sheet(sheet.name.strip(), raw_rows)
            if sheet_md:
                pages.append((s_idx + 1, sheet_md))

        return pages

    @classmethod
    def _extract_txt_or_md(cls, file_path: Path) -> List[Tuple[int, str]]:
        """Mengekstrak berkas plain text (.txt) atau Markdown (.md) dengan deteksi encoding."""
        raw_bytes = file_path.read_bytes()
        if b"\x00" in raw_bytes:
            raise QualityGateError(
                f"Berkas teks {file_path.name} mengandung null byte (\\x00) / biner korup."
            )

        content: Optional[str] = None
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                content = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if content is None or not content.strip():
            return []

        return [(1, content)]

    @classmethod
    def _extract_pdf(cls, file_path: Path) -> List[Tuple[int, str]]:
        """Mengekstrak berkas PDF per halaman beserta deteksi tabel otomatis."""
        import pymupdf

        doc = pymupdf.open(str(file_path))
        pages: List[Tuple[int, str]] = []
        try:
            for idx, page in enumerate(doc, start=1):
                page_text = page.get_text("text").strip()
                try:
                    tabs = page.find_tables()
                    if tabs and tabs.tables:
                        md_tables = [
                            t.to_markdown().strip()
                            for t in tabs
                            if t.to_markdown().strip()
                        ]
                        if md_tables:
                            page_text = page_text + "\n\n" + "\n\n".join(md_tables)
                except Exception:
                    pass
                if page_text:
                    pages.append((idx, page_text))
        finally:
            doc.close()
        return pages

    def extract_pages_by_extension(self, file_path: Path) -> List[Tuple[int, str]]:
        """Memilih parser sesuai ekstensi berkas mentah."""
        ext = file_path.suffix.lower()
        if ext == ".doc":
            return self._extract_doc(file_path)
        if ext == ".docx":
            return self._extract_docx(file_path)
        if ext == ".xlsx":
            return self._extract_xlsx(file_path)
        if ext == ".xls":
            return self._extract_xls(file_path)
        if ext in (".txt", ".md"):
            return self._extract_txt_or_md(file_path)
        if ext == ".pdf":
            return self._extract_pdf(file_path)
        raise ValueError(f"Ekstensi berkas '{ext}' tidak didukung untuk konversi.")

    # ------------------------------------------------------------------
    # Orkestrasi Konversi Tunggal & Batch (FR-1 & FR-2)
    # ------------------------------------------------------------------
    def convert_file(
        self, file_path: Path, run_id: str = "manual", force: bool = False
    ) -> ConversionRecord:
        """Mengonversi satu berkas mentah ke .md di data/processed dengan pengecekan incremental."""
        start_t = time.perf_counter()
        now_iso = datetime.now(timezone.utc).isoformat()
        source_hash = self.compute_sha256(file_path)
        target_md_path = self.output_dir / f"{file_path.stem}.md"

        should_run, reason = self.should_process_file(
            file_path, source_hash=source_hash, force=force
        )
        if not should_run:
            cached = self.lineage_cache.get(file_path.name, {})
            duration_ms = round((time.perf_counter() - start_t) * 1000, 2)
            record = ConversionRecord(
                run_id=run_id,
                timestamp=now_iso,
                source_file=file_path.name,
                output_file=target_md_path.name,
                status="SKIPPED",
                source_hash=source_hash,
                md_hash=cached.get("md_hash"),
                page_count=int(cached.get("page_count", 1)),
                line_count=int(cached.get("line_count", 0)),
                char_count=int(cached.get("char_count", 0)),
                duration_ms=duration_ms,
                reason=reason,
            )
            self._write_conversion_log_line(
                f"[SKIPPED] run_id={run_id} | source='{file_path.name}' -> '{target_md_path.name}' | "
                f"reason={reason} | sha256={source_hash[:12]}... | duration={duration_ms}ms"
            )
            self._write_conversion_history_event({"event": "file_conversion", **record.to_dict()})
            return record

        try:
            pages = self.extract_pages_by_extension(file_path)
            if not pages:
                raise QualityGateError(
                    f"Tidak ada konten teks yang dapat diekstrak dari '{file_path.name}'."
                )

            md_content = self.sanitizer.build_markdown_document(
                doc_title=file_path.stem,
                pages_content=pages,
                source_filename=file_path.name,
            )

            is_valid, gate_reason = self.sanitizer.validate_quality_gates(md_content)
            if not is_valid:
                raise QualityGateError(gate_reason)

            # Tulis berkas .md ke data/processed secara atomik dengan encoding UTF-8
            tmp_md = target_md_path.with_suffix(".md.tmp")
            tmp_md.write_text(md_content, encoding="utf-8")
            tmp_md.replace(target_md_path)

            # Catat lineage metadata ke data/processed/lineage_cache.json
            lineage = self._record_lineage(
                source_path=file_path,
                output_path=target_md_path,
                source_hash=source_hash,
                md_content=md_content,
                page_count=len(pages),
            )

            duration_ms = round((time.perf_counter() - start_t) * 1000, 2)
            record = ConversionRecord(
                run_id=run_id,
                timestamp=now_iso,
                source_file=file_path.name,
                output_file=target_md_path.name,
                status="CONVERTED",
                source_hash=source_hash,
                md_hash=lineage.md_hash,
                page_count=lineage.page_count,
                line_count=lineage.line_count,
                char_count=lineage.char_count,
                duration_ms=duration_ms,
                reason=reason,
            )
            self._write_conversion_log_line(
                f"[CONVERTED] run_id={run_id} | source='{file_path.name}' -> '{target_md_path.name}' | "
                f"pages={lineage.page_count} | lines={lineage.line_count} | chars={lineage.char_count} | "
                f"sha256={source_hash[:12]}... | duration={duration_ms}ms"
            )
            self._write_conversion_history_event({"event": "file_conversion", **record.to_dict()})
            return record

        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_t) * 1000, 2)
            tb_str = traceback.format_exc()
            err_msg = f"{type(exc).__name__}: {exc}"
            self._route_to_dlq(
                file_path=file_path,
                source_hash=source_hash,
                reason=err_msg,
                tb_text=tb_str,
            )
            record = ConversionRecord(
                run_id=run_id,
                timestamp=now_iso,
                source_file=file_path.name,
                output_file=None,
                status="DLQ_FAILED",
                source_hash=source_hash,
                md_hash=None,
                page_count=0,
                line_count=0,
                char_count=0,
                duration_ms=duration_ms,
                reason=err_msg,
            )
            self._write_conversion_log_line(
                f"[DLQ_FAILED] run_id={run_id} | source='{file_path.name}' -> data/dlq/{file_path.name} | "
                f"reason={err_msg} | sha256={source_hash[:12]}... | duration={duration_ms}ms"
            )
            self._write_conversion_history_event({"event": "file_conversion", **record.to_dict()})
            return record

    def convert_directory(self, force: bool = False) -> BatchConversionSummary:
        """Memindai seluruh berkas di input_dir (data/raw) dan mengonversi berkas baru/berubah ke output_dir (data/processed)."""
        run_id = uuid.uuid4().hex[:8]
        started_at = datetime.now(timezone.utc).isoformat()
        batch_start_t = time.perf_counter()

        candidate_files = sorted(
            [
                f
                for f in self.input_dir.iterdir()
                if f.is_file()
                and not f.name.startswith(".")
                and f.suffix.lower() in CONVERTIBLE_EXTENSIONS
            ],
            key=lambda p: p.name.lower(),
        )

        self._write_conversion_log_line(
            f"=== BATCH CONVERSION START (run_id={run_id}) | input='{self.input_dir}' | "
            f"output='{self.output_dir}' | files_found={len(candidate_files)} | force={force} ==="
        )

        summary = BatchConversionSummary(
            run_id=run_id,
            started_at=started_at,
            completed_at="",
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            total_scanned=len(candidate_files),
        )

        for file_path in candidate_files:
            rec = self.convert_file(file_path, run_id=run_id, force=force)
            summary.records.append(rec)
            if rec.status == "CONVERTED":
                summary.converted_count += 1
            elif rec.status == "SKIPPED":
                summary.skipped_count += 1
            elif rec.status == "DLQ_FAILED":
                summary.dlq_failed_count += 1

        summary.completed_at = datetime.now(timezone.utc).isoformat()
        summary.total_duration_ms = round(
            (time.perf_counter() - batch_start_t) * 1000, 2
        )

        self._write_conversion_log_line(
            f"=== BATCH CONVERSION END (run_id={run_id}) | total={summary.total_scanned} | "
            f"converted={summary.converted_count} | skipped={summary.skipped_count} | "
            f"dlq_failed={summary.dlq_failed_count} | total_duration={summary.total_duration_ms}ms ==="
        )
        self._write_conversion_history_event(
            {
                "event": "batch_summary",
                "run_id": summary.run_id,
                "started_at": summary.started_at,
                "completed_at": summary.completed_at,
                "input_dir": summary.input_dir,
                "output_dir": summary.output_dir,
                "total_scanned": summary.total_scanned,
                "converted_count": summary.converted_count,
                "skipped_count": summary.skipped_count,
                "dlq_failed_count": summary.dlq_failed_count,
                "total_duration_ms": summary.total_duration_ms,
            }
        )

        return summary
