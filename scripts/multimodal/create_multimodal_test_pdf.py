
"""Create a synthetic CV PDF for Lesson 6A.1.

Creates:
1. A text-based CV PDF.
2. A scanned-style CV PDF containing only a page image.

No personal data or API calls.
"""

from pathlib import Path

import pymupdf as fitz


OUTPUT_DIR = Path("data/multimodal/input")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    text_pdf = OUTPUT_DIR / "synthetic_text_cv.pdf"
    scan_pdf = OUTPUT_DIR / "synthetic_scanned_cv.pdf"

    lines = [
        "SYNTHETIC CV - CANDIDATE M001",
        "",
        "Role: Software Engineer",
        "Skills: Python, SQL, Docker, GitLab",
        "",
        "Experience:",
        "Built a Python data-processing pipeline.",
        "Used SQL for reporting and analysis.",
        "Containerized an application with Docker.",
        "",
        "Projects:",
        "Recruitment Analytics Dashboard",
        "Developed a dashboard for synthetic data.",
        "",
        "Education:",
        "Bachelor of Computer Science",
    ]

    document = fitz.open()
    page = document.new_page(
        width=595,
        height=842,
    )

    y = 60

    for line in lines:
        page.insert_text(
            (55, y),
            line,
            fontsize=12,
        )
        y += 30

    document.save(text_pdf)
    document.close()

    with fitz.open(text_pdf) as source:
        pixmap = source[0].get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False,
        )

        image_bytes = pixmap.tobytes("png")

    scanned = fitz.open()
    scanned_page = scanned.new_page(
        width=595,
        height=842,
    )

    scanned_page.insert_image(
        scanned_page.rect,
        stream=image_bytes,
    )

    scanned.save(scan_pdf)
    scanned.close()

    print("Created:", text_pdf)
    print("Created:", scan_pdf)


if __name__ == "__main__":
    main()
