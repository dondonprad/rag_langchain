"""CLI orchestrator untuk mengonversi dokumen mentah di data/raw menjadi berkas Markdown (.md) di data/processed.

Fitur Utama:
- Default Incremental Mode: Berkas yang sudah dikonversi dan memiliki SHA-256 yang sama otomatis dilewati (SKIPPED).
- Hanya memproses berkas baru atau berkas yang isinya berubah di data/raw.
- Mencatat riwayat konversi lengkap ke logs/conversion.log dan logs/conversion_history.jsonl.
- Mengisolasi berkas korup/kosong ke data/dlq/ dan mencatat error di data/dlq/error.log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Pastikan root proyek berada di sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.ingestion.converter import RawToMarkdownConverter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Konversi berkas dokumen mentah (data/raw) menjadi berkas Markdown (.md) "
            "terstruktur di data/processed sesuai ketentuan PRD.md."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=settings.paths.data_raw_dir,
        help=f"Direktori sumber dokumen mentah (default: {settings.paths.data_raw_dir})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.paths.data_processed_dir,
        help=f"Direktori keluaran berkas .md (default: {settings.paths.data_processed_dir})",
    )
    parser.add_argument(
        "--force",
        "--reset",
        dest="force",
        action="store_true",
        help="Paksa konversi ulang seluruh berkas meskipun sudah pernah diproses sebelumnya.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    converter = RawToMarkdownConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    mode_label = "FORCE RE-CONVERT" if args.force else "INCREMENTAL (Hanya Berkas Baru/Berubah)"
    print("=" * 88)
    print(f"🚀 RAG_LANGCHAIN - RAW TO MARKDOWN (.MD) CONVERTER")
    print("=" * 88)
    print(f"📂 Input Directory   : {converter.input_dir}")
    print(f"📁 Output Directory  : {converter.output_dir}")
    print(f"🛡️  DLQ Directory     : {converter.dlq_dir}")
    print(f"📝 Conversion Log    : {converter.conversion_log_path}")
    print(f"📊 Audit History Log : {converter.conversion_history_path}")
    print(f"⚙️  Execution Mode    : {mode_label}")
    print("-" * 88)

    summary = converter.convert_directory(force=args.force)

    for idx, rec in enumerate(summary.records, start=1):
        if rec.status == "CONVERTED":
            badge = "✅ CONVERTED"
            detail = (
                f"-> {rec.output_file} "
                f"({rec.page_count} hal/sheet, {rec.line_count} baris, {rec.char_count} chars, {rec.duration_ms:.1f} ms)"
            )
        elif rec.status == "SKIPPED":
            badge = "⏭️  SKIPPED  "
            detail = f"-> {rec.output_file} (Sudah diproses / Cache Hit, {rec.duration_ms:.1f} ms)"
        else:
            badge = "❌ DLQ_FAIL "
            detail = f"-> Dialihkan ke DLQ ({rec.reason})"

        print(f"{idx:02d}. [{badge}] {rec.source_file} {detail}")

    print("-" * 88)
    print(
        f"📊 RINGKASAN BATCH (run_id={summary.run_id}): "
        f"Total={summary.total_scanned} | "
        f"Converted={summary.converted_count} | "
        f"Skipped={summary.skipped_count} | "
        f"DLQ Failed={summary.dlq_failed_count} | "
        f"Durasi={summary.total_duration_ms:.1f} ms"
    )
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
