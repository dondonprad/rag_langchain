"""Markdown text sanitizer, hierarchy formatter, and quality gate validator.

Mengimplementasikan kebutuhan fungsional PRD:
- FR-2.2: Preservasi struktur hirarki Markdown (#, ##, ###) dan penanda halaman ([Hal: X], <!-- page: X -->).
- FR-2.3: Quality Gates (validasi encoding UTF-8 tanpa karakter null/korup dan batas minimum 50 karakter terbaca).
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple


class QualityGateError(ValueError):
    """Exception ketika dokumen gagal melewati Data Quality Gates (FR-2.3)."""


class MarkdownSanitizer:
    """Sanitasi teks, normalisasi UTF-8, pembentukan struktur hierarki Markdown, dan Quality Gates."""

    def __init__(self, min_file_size_chars: int = 50) -> None:
        self.min_file_size_chars = min_file_size_chars

    @staticmethod
    def clean_document_title(raw_filename_stem: str) -> str:
        """Membersihkan nama berkas menjadi judul dokumen utama (# Header 1) yang rapi."""
        title = raw_filename_stem.strip()
        # Rapikan spasi berlebih atau underscore
        title = title.replace("_", " ")
        title = re.sub(r"\s+", " ", title)
        return title.strip()

    def normalize_text(self, text: str) -> str:
        """Normalisasi format teks ke UTF-8 bersih tanpa merusak sintaks tabel/heading Markdown."""
        if not text:
            return ""

        # 1. Perbaikan Mojibake umum (UTF-8 yang salah ter-decode sebagai Latin-1/CP1252)
        mojibake_map = {
            "â€™": "'",
            "â€˜": "'",
            "â€œ": '"',
            "â€\x9d": '"',
            "â€“": "-",
            "â€”": "-",
            "Â ": " ",
        }
        for bad_s, good_s in mojibake_map.items():
            text = text.replace(bad_s, good_s)

        # 2. Perbaikan ligatures tipografi
        ligatures = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
            "ﬆ": "st",
            "ﬅ": "ft",
        }
        for lig, repl in ligatures.items():
            text = text.replace(lig, repl)

        # 3. Normalisasi Unicode NFKC
        text = unicodedata.normalize("NFKC", text)

        # 4. Hapus zero-width chars, BOM, soft-hyphen, dan replacement char
        text = re.sub(r"[\u200B-\u200F\uFEFF\u2060\u00AD\uFFFD]", "", text)

        # 5. Konversi variasi spasi unicode ke spasi standar
        text = re.sub(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]", " ", text)

        # 6. Standarisasi newline
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 7. Konversi tab menjadi spasi
        text = text.replace("\t", " ")

        # 8. Hapus control characters berbahaya & null bytes (kecuali newline \n)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # 9. Hapus karakter sampah padding OLE biner (misal 'ÿÿÿÿ')
        text = re.sub(r"ÿ{2,}", "", text)

        # 10. Standarisasi tanda kutip dan dash
        text = re.sub(r"[\u201C\u201D\u00AB\u00BB\u201E\u201F]", '"', text)
        text = re.sub(r"[\u2018\u2019\u201A\u201B\u02BC\u02BB]", "'", text)
        text = re.sub(r"[\u2013\u2014\u2015\u2212]", "-", text)
        text = text.replace("…", "...")

        # 11. Standarisasi simbol bullet menjadi list Markdown "- "
        text = re.sub(r"(?m)^[\t ]*[•●▪►◦✦]\s*", "- ", text)

        # 12. De-hyphenation kata terpotong di akhir baris
        text = re.sub(r"([A-Za-z]{2,})-\n\s*([a-z]{2,})", r"\1\2", text)

        # 13. Rapikan spasi horizontal berlebih per baris (tetap jaga baris tabel Markdown)
        cleaned_lines: List[str] = []
        for line in text.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                cleaned_lines.append("")
                continue
            if line_stripped.startswith("|") and line_stripped.endswith("|"):
                cleaned_lines.append(line_stripped)
            else:
                norm_line = re.sub(r"[^\S\n]+", " ", line_stripped)
                # Hilangkan nomor halaman berdiri sendiri seperti "- 12 -"
                if re.match(r"^[-\[\(]?\s*\d{1,4}\s*[-\]\)]?$", norm_line):
                    continue
                cleaned_lines.append(norm_line)

        result = "\n".join(cleaned_lines)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    def apply_hierarchy_headers(self, text: str) -> str:
        """Mendeteksi dan mempromosikan struktur bab/sub-bab dokumen menjadi heading Markdown (## dan ###).

        Kompatibel dengan FR-3.1 (MarkdownHeaderTextSplitter: #, ##, ###):
        - BAB I / BAB II + Judul Bab -> ## BAB I - JUDUL BAB
        - [LEMBAR KERJA: Nama] -> ## Lembar Kerja: Nama
        - Bagian Huruf Kapital (A. PENDAHULUAN, B. INTI, C. PENUTUP) -> ## B. INTI
        - Pasal / Sub-bab bernomor (1. Umum., 2. Maksud dan Tujuan.) -> ### 1. Umum
        """
        if not text:
            return ""

        lines = text.splitlines()
        out_lines: List[str] = []
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i].strip()

            if not line:
                out_lines.append("")
                i += 1
                continue

            # Jangan ubah baris tabel Markdown, komentar HTML, atau yang sudah berupa heading
            if (
                line.startswith("#")
                or line.startswith("|")
                or line.startswith("<!--")
                or line.startswith("[Hal:")
            ):
                out_lines.append(line)
                i += 1
                continue

            # 1. Konversi tag [LEMBAR KERJA: ...] menjadi ## Lembar Kerja: ...
            m_sheet = re.match(r"^\[LEMBAR KERJA:\s*(.+?)\]$", line, flags=re.IGNORECASE)
            if m_sheet:
                sheet_title = m_sheet.group(1).strip()
                out_lines.append(f"## Lembar Kerja: {sheet_title}")
                i += 1
                continue

            # 2. Deteksi "BAB I", "BAB II", dst.
            m_bab_only = re.match(
                r"^BAB\s+([IVXLCDM]+|\d+)\s*$", line, flags=re.IGNORECASE
            )
            if m_bab_only:
                bab_num = m_bab_only.group(1).upper()
                # Cari baris non-kosong berikutnya sebagai judul BAB
                j = i + 1
                while j < n and not lines[j].strip():
                    j += 1
                if j < n:
                    next_line = lines[j].strip()
                    # Jika baris berikutnya adalah huruf kapital / judul pendek dan bukan pasal
                    if (
                        len(next_line) <= 100
                        and not next_line.startswith("|")
                        and not re.match(r"^\d+\.", next_line)
                    ):
                        out_lines.append(f"## BAB {bab_num} - {next_line.rstrip('.')}")
                        i = j + 1
                        continue
                out_lines.append(f"## BAB {bab_num}")
                i += 1
                continue

            m_bab_inline = re.match(
                r"^BAB\s+([IVXLCDM]+|\d+)\s*[-:.\s]\s*(.+)$", line, flags=re.IGNORECASE
            )
            if m_bab_inline and len(line) <= 120:
                bab_num = m_bab_inline.group(1).upper()
                bab_title = m_bab_inline.group(2).strip().rstrip(".")
                out_lines.append(f"## BAB {bab_num} - {bab_title}")
                i += 1
                continue

            # 3. Deteksi Bagian Huruf Kapital Utama: "A. PENDAHULUAN", "B. INTI", "C. AKHIR/PENUTUP"
            m_upper_sec = re.match(r"^([A-Z])\.\s+([A-Z][A-Z0-9\s/&-]{2,60})$", line)
            if m_upper_sec:
                sec_letter = m_upper_sec.group(1)
                sec_title = m_upper_sec.group(2).strip().rstrip(".")
                out_lines.append(f"## {sec_letter}. {sec_title}")
                i += 1
                continue

            # 4. Deteksi Sub-Bab / Pasal bernomor:
            # Kasus A: Hanya judul pasal di satu baris, misal "1. Umum." atau "2. Maksud dan Tujuan."
            m_num_heading_only = re.match(
                r"^(\d{1,2})\.\s+([A-Z][A-Za-z0-9\s/()&-]{2,65}?)\.?\s*$", line
            )
            if m_num_heading_only:
                num_str = m_num_heading_only.group(1)
                sec_title = m_num_heading_only.group(2).strip().rstrip(".")
                out_lines.append(f"### {num_str}. {sec_title}")
                i += 1
                continue

            # Kasus B: Judul pasal diikuti langsung kalimat isi pada baris yang sama
            # Misal: "5. Tujuan Pendidikan. Mengembangkan kemampuan Perwira Siswa..."
            m_num_heading_inline = re.match(
                r"^(\d{1,2})\.\s+([A-Z][A-Za-z0-9\s/()&-]{2,55}?)\.\s+([A-Z].+)$",
                line,
            )
            if m_num_heading_inline:
                num_str = m_num_heading_inline.group(1)
                sec_title = m_num_heading_inline.group(2).strip().rstrip(".")
                body_rest = m_num_heading_inline.group(3).strip()
                out_lines.append(f"### {num_str}. {sec_title}")
                out_lines.append("")
                out_lines.append(body_rest)
                i += 1
                continue

            out_lines.append(line)
            i += 1

        formatted = "\n".join(out_lines)
        formatted = re.sub(r"\n{3,}", "\n\n", formatted)
        return formatted.strip()

    def build_markdown_document(
        self,
        doc_title: str,
        pages_content: List[Tuple[int, str]],
        source_filename: str,
    ) -> str:
        """Menyusun dokumen Markdown utuh lengkap dengan Header #, ##, ### dan tag halaman (FR-2.2).

        Setiap halaman/sheet diberi penanda `<!-- page: X -->` dan `[Hal: X]` sesuai
        pola regex pada `config/config.yaml` (`page_tag_patterns`).
        """
        clean_title = self.clean_document_title(doc_title)
        blocks: List[str] = [f"# {clean_title}"]

        has_subheaders = False
        page_blocks: List[str] = []

        for page_num, raw_page_text in pages_content:
            norm_text = self.normalize_text(raw_page_text)
            if not norm_text:
                continue
            structured_text = self.apply_hierarchy_headers(norm_text)
            if re.search(r"(?m)^#{2,3}\s+", structured_text):
                has_subheaders = True

            page_header = f"<!-- page: {page_num} -->\n[Hal: {page_num}]"
            page_blocks.append(f"{page_header}\n\n{structured_text}")

        # Jika dokumen pendek (misal .txt atau struktur singkat) belum punya ## Header 2,
        # tambahkan ## Isi Dokumen agar MarkdownHeaderTextSplitter selalu memiliki section.
        if page_blocks and not has_subheaders:
            first_block = page_blocks[0]
            lines = first_block.splitlines()
            if len(lines) >= 2 and lines[0].startswith("<!-- page:"):
                marker_part = "\n".join(lines[:2])
                rest_part = "\n".join(lines[2:]).strip()
                page_blocks[0] = (
                    f"{marker_part}\n\n## Ringkasan & Isi Dokumen\n\n{rest_part}"
                )

        blocks.extend(page_blocks)
        final_md = "\n\n".join(blocks).strip() + "\n"
        return final_md

    @staticmethod
    def extract_readable_chars(markdown_text: str) -> str:
        """Mengekstrak karakter teks murni (mengabaikan tag HTML, tag [Hal: X], header sintetis, dan simbol dekorasi Markdown)."""
        if not markdown_text:
            return ""
        stripped = re.sub(r"<!--.*?-->", "", markdown_text, flags=re.DOTALL)
        stripped = re.sub(r"\[Hal:\s*\d+\]", "", stripped)
        # Abaikan baris judul utama pertama (# ...) dan subjudul sintetis (## Ringkasan & Isi Dokumen)
        stripped = re.sub(r"\A\s*#\s+[^\n]+\n?", "", stripped)
        stripped = re.sub(r"(?m)^##\s+Ringkasan\s+&\s+Isi\s+Dokumen\s*$", "", stripped)
        stripped = re.sub(r"[#*|\-_>`~]", " ", stripped)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        return stripped

    def validate_quality_gates(self, markdown_text: str) -> Tuple[bool, str]:
        """Memvalidasi integritas encoding UTF-8 dan panjang karakter terbaca minimal (FR-2.3).

        Returns:
            Tuple (is_valid, reason_if_invalid)
        """
        if not markdown_text or not markdown_text.strip():
            return False, "Dokumen kosong (0 karakter)."

        if "\x00" in markdown_text:
            return False, "Dokumen mengandung null byte (\\x00) / indikasi biner korup."

        try:
            markdown_text.encode("utf-8").decode("utf-8")
        except UnicodeError as exc:
            return False, f"Integritas encoding UTF-8 gagal: {exc}"

        readable = self.extract_readable_chars(markdown_text)
        if len(readable) < self.min_file_size_chars:
            return (
                False,
                f"Jumlah karakter terbaca ({len(readable)} chars) kurang dari batas minimum Quality Gate ({self.min_file_size_chars} chars).",
            )

        return True, "OK"
