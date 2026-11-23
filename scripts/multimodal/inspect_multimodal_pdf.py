
"""Lesson 6A.1 — Inspect a PDF for multimodal processing.

Usage:
    python -m scripts.multimodal.inspect_multimodal_pdf path/to/file.pdf

The script:
1. Extracts text from each PDF page.
2. Reports text length and embedded image counts.
3. Renders the first page as a PNG.
4. Saves a JSON inspection report.

No AI API calls are required.
"""

import argparse
import json
from pathlib import Path

import pymupdf as fitz
from PIL import Image


OUTPUT_DIR = Path("data/multimodal")
RENDERED_DIR = OUTPUT_DIR / "rendered"
REPORTS_DIR = OUTPUT_DIR / "reports"


def inspect_pdf(pdf_path):
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            "Please provide a PDF file."
        )

    RENDERED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "source_file": str(pdf_path),
        "page_count": 0,
        "pages": [],
        "total_characters": 0,
        "classification": None,
        "rendered_first_page": None,
    }

    with fitz.open(pdf_path) as document:
        if document.is_encrypted:
            raise ValueError(
                "Encrypted PDFs are not supported "
                "in this lesson."
            )

        report["page_count"] = len(document)

        for page_index, page in enumerate(document):
            text = page.get_text("text") or ""
            char_count = len(text.strip())

            embedded_images = page.get_images(
                full=True
            )

            page_info = {
                "page_number": page_index + 1,
                "character_count": char_count,
                "embedded_image_count": len(
                    embedded_images
                ),
                "text_preview": text[:300],
            }

            report["pages"].append(page_info)

            report["total_characters"] += (
                char_count
            )

            print(
                f"Page {page_index + 1}: "
                f"{char_count} characters, "
                f"{len(embedded_images)} "
                "embedded images"
            )

        if len(document) > 0:
            first_page = document[0]

            pixmap = first_page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False,
            )

            image_path = (
                RENDERED_DIR
                / f"{pdf_path.stem}_page_1.png"
            )

            pixmap.save(str(image_path))

            with Image.open(image_path) as image:
                print(
                    "Rendered image size:",
                    image.size,
                )

            report["rendered_first_page"] = str(
                image_path
            )

    if report["total_characters"] == 0:
        report["classification"] = (
            "no_extractable_text"
        )
    elif report["total_characters"] < 100:
        report["classification"] = (
            "low_extractable_text"
        )
    else:
        report["classification"] = (
            "text_available"
        )

    report_path = (
        REPORTS_DIR
        / f"{pdf_path.stem}_inspection.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== PDF INSPECTION SUMMARY ===")
    print("Source:", pdf_path)
    print("Pages:", report["page_count"])
    print(
        "Total characters:",
        report["total_characters"],
    )
    print(
        "Classification:",
        report["classification"],
    )
    print(
        "Rendered first page:",
        report["rendered_first_page"],
    )
    print("Report:", report_path)

    return report


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a CV PDF for multimodal RAG."
        )
    )

    parser.add_argument(
        "pdf_path",
        help="Path to a PDF file",
    )

    args = parser.parse_args()

    inspect_pdf(Path(args.pdf_path))


if __name__ == "__main__":
    main()
