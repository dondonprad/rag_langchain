"""Unit and integration tests for RawToMarkdownConverter, MarkdownSanitizer, and MarkdownDocumentLoader."""

from __future__ import annotations

import json
import re
from pathlib import Path

from config.settings import get_settings
from src.ingestion.converter import RawToMarkdownConverter
from src.ingestion.markdown_loader import MarkdownDocumentLoader
from src.ingestion.sanitizer import MarkdownSanitizer


def test_sanitizer_hierarchy_and_page_tags():
    settings = get_settings()
    sanitizer = MarkdownSanitizer(min_file_size_chars=50)

    raw_page_1 = (
        "PUSAT ZENI TNI ANGKATAN DARAT\n"
        "BAB I\n"
        "PENDAHULUAN\n"
        "1. Umum.\n"
        "Pendidikan Lanjutan Perwira Kecabangan Zeni TNI AD bertujuan meningkatkan profesionalisme perwira.\n"
        "2. Maksud dan Tujuan. Sebagai pedoman penyelenggaraan operasional pendidikan di Pusdikzi."
    )
    md_doc = sanitizer.build_markdown_document(
        doc_title="0.Badan Renopsdik DIKLAPA",
        pages_content=[(1, raw_page_1)],
        source_filename="0.Badan Renopsdik DIKLAPA.doc",
    )

    assert md_doc.startswith("# 0.Badan Renopsdik DIKLAPA")
    assert "<!-- page: 1 -->" in md_doc
    assert "[Hal: 1]" in md_doc
    assert "## BAB I - PENDAHULUAN" in md_doc
    assert "### 1. Umum" in md_doc
    assert "### 2. Maksud dan Tujuan" in md_doc

    # Pastikan cocok dengan regex page_tag_patterns pada config.yaml
    for pat in settings.ingestion.page_tag_patterns:
        assert re.search(pat, md_doc) is not None

    is_valid, reason = sanitizer.validate_quality_gates(md_doc)
    assert is_valid is True, f"Seharusnya valid, tetapi gagal: {reason}"


def test_quality_gates_reject_short_or_corrupt():
    sanitizer = MarkdownSanitizer(min_file_size_chars=50)

    valid_empty, _ = sanitizer.validate_quality_gates("")
    assert valid_empty is False

    valid_short, reason_short = sanitizer.validate_quality_gates(
        "# Judul\n\n<!-- page: 1 -->\n[Hal: 1]\n\nTeks terlalu pendek."
    )
    assert valid_short is False
    assert "kurang dari batas minimum" in reason_short

    valid_null, reason_null = sanitizer.validate_quality_gates(
        "# Judul\n\nKalimat panjang lebih dari lima puluh karakter tetapi mengandung byte\x00korup di tengah."
    )
    assert valid_null is False
    assert "null byte" in reason_null


def test_incremental_conversion_skips_existing_and_processes_only_new_files(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    dlq_dir = tmp_path / "dlq"
    logs_dir = tmp_path / "logs"
    raw_dir.mkdir()

    file1 = raw_dir / "dokumen_satu.txt"
    file1.write_text(
        "BAB I\nPENDAHULUAN\n1. Umum.\nIni adalah isi dokumen pertama yang memiliki lebih dari lima puluh karakter terbaca untuk pengujian sistem RAG.",
        encoding="utf-8",
    )
    file2 = raw_dir / "dokumen_dua.txt"
    file2.write_text(
        "BAB I\nKETENTUAN POKOK\n1. Dasar.\nIni adalah isi dokumen kedua yang juga memiliki panjang di atas lima puluh karakter terbaca.",
        encoding="utf-8",
    )

    converter = RawToMarkdownConverter(
        input_dir=raw_dir,
        output_dir=proc_dir,
        dlq_dir=dlq_dir,
        dlq_error_log=dlq_dir / "error.log",
        lineage_cache_path=proc_dir / "lineage_cache.json",
        conversion_log_path=logs_dir / "conversion.log",
        conversion_history_path=logs_dir / "conversion_history.jsonl",
    )

    # Eksekusi 1: Kedua file baru harus dikonversi
    summary1 = converter.convert_directory()
    assert summary1.total_scanned == 2
    assert summary1.converted_count == 2
    assert summary1.skipped_count == 0
    assert (proc_dir / "dokumen_satu.md").exists()
    assert (proc_dir / "dokumen_dua.md").exists()

    # Tambahkan 1 file baru di folder raw
    file3 = raw_dir / "dokumen_tiga_baru.txt"
    file3.write_text(
        "BAB II\nPELAKSANAAN\n1. Prosedur.\nDokumen ketiga baru ditambahkan ke dalam folder raw dan harus menjadi satu-satunya berkas yang diproses.",
        encoding="utf-8",
    )

    # Eksekusi 2: dokumen_satu dan dokumen_dua harus SKIPPED, hanya dokumen_tiga_baru yang CONVERTED
    summary2 = converter.convert_directory()
    assert summary2.total_scanned == 3
    assert summary2.converted_count == 1
    assert summary2.skipped_count == 2
    assert summary2.dlq_failed_count == 0

    status_map = {r.source_file: r.status for r in summary2.records}
    assert status_map["dokumen_satu.txt"] == "SKIPPED"
    assert status_map["dokumen_dua.txt"] == "SKIPPED"
    assert status_map["dokumen_tiga_baru.txt"] == "CONVERTED"
    assert (proc_dir / "dokumen_tiga_baru.md").exists()


def test_dlq_handling_and_conversion_logs(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    dlq_dir = tmp_path / "dlq"
    logs_dir = tmp_path / "logs"
    raw_dir.mkdir()

    # 1 berkas korup (< 50 chars) dan 1 berkas .doc palsu/rusak, serta 1 berkas valid
    short_file = raw_dir / "terlalu_pendek.txt"
    short_file.write_text("Pendek sekali.", encoding="utf-8")

    corrupt_doc = raw_dir / "korup_biner.doc"
    corrupt_doc.write_bytes(b"BUKAN_OLE_HEADER_VALID\x00\x01\x02")

    valid_file = raw_dir / "pedoman_valid.txt"
    valid_file.write_text(
        "BAB I\nPENDAHULUAN\n1. Umum.\nDokumen pedoman operasional standar yang valid dan memiliki lebih dari lima puluh karakter terbaca.",
        encoding="utf-8",
    )

    converter = RawToMarkdownConverter(
        input_dir=raw_dir,
        output_dir=proc_dir,
        dlq_dir=dlq_dir,
        dlq_error_log=dlq_dir / "error.log",
        lineage_cache_path=proc_dir / "lineage_cache.json",
        conversion_log_path=logs_dir / "conversion.log",
        conversion_history_path=logs_dir / "conversion_history.jsonl",
    )

    summary = converter.convert_directory()
    assert summary.total_scanned == 3
    assert summary.converted_count == 1
    assert summary.dlq_failed_count == 2

    # Pastikan berkas gagal masuk ke data/dlq/ dan tercatat di error.log
    assert (dlq_dir / "terlalu_pendek.txt").exists()
    assert (dlq_dir / "korup_biner.doc").exists()
    assert (dlq_dir / "error.log").exists()
    error_lines = (dlq_dir / "error.log").read_text(encoding="utf-8").strip().splitlines()
    assert len(error_lines) == 2

    # Pastikan log proses konversi tercatat di folder logs/
    assert (logs_dir / "conversion.log").exists()
    assert (logs_dir / "conversion_history.jsonl").exists()
    log_text = (logs_dir / "conversion.log").read_text(encoding="utf-8")
    assert "[CONVERTED]" in log_text
    assert "[DLQ_FAILED]" in log_text
    assert "BATCH CONVERSION END" in log_text

    history_events = [
        json.loads(line)
        for line in (logs_dir / "conversion_history.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    ]
    assert len(history_events) == 4  # 3 file_conversion + 1 batch_summary
    assert history_events[-1]["event"] == "batch_summary"

    # Uji MarkdownDocumentLoader membaca hasil di processed
    loader = MarkdownDocumentLoader(
        source_dir=proc_dir,
        dlq_dir=dlq_dir,
        dlq_error_log=dlq_dir / "error.log",
    )
    loaded_docs = loader.load_directory()
    assert len(loaded_docs) == 1
    assert loaded_docs[0].document_name == "pedoman_valid.md"
    assert loaded_docs[0].page_numbers == [1]


if __name__ == "__main__":
    import tempfile

    test_sanitizer_hierarchy_and_page_tags()
    test_quality_gates_reject_short_or_corrupt()
    with tempfile.TemporaryDirectory() as d1:
        test_incremental_conversion_skips_existing_and_processes_only_new_files(Path(d1))
    with tempfile.TemporaryDirectory() as d2:
        test_dlq_handling_and_conversion_logs(Path(d2))
    print("✅ All ingestion & converter tests passed successfully!")
