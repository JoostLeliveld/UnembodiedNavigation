from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


# Lives in UnembodiedNavigation so it stays version-controlled; it COMPILES the
# paper repo, which carries only what the document compiles from.
ROOT = Path(__file__).resolve().parents[3].parent / "papers" / "Thesis"
BUILD = ROOT / "build"
OUTPUT = ROOT / "output"


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def first_appendix_page(pdf: Path) -> int | None:
    """The 1-based PDF page the appendices start on, located from the rendered text."""
    try:
        text = subprocess.run(
            ["/usr/bin/pdftotext", str(pdf), "-"],
            check=True, capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    for number, page in enumerate(text.split("\f"), start=1):
        if "PPENDIX" in page:
            return number
    return None


def pages(path: Path) -> int:
    result = subprocess.run(
        ["/usr/bin/pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^Pages:\s+(\d+)$", result.stdout, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Could not read page count from {path}")
    return int(match.group(1))


BUILD.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

for _ in range(2):
    run("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "main.tex")
run("bibtex", "build/main")
for _ in range(2):
    run("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "main.tex")

# main.tex carries the AIES cover as page 1. Appendices are permitted but not
# assessed, so they are excluded from the 12-page count.
thesis_pdf = BUILD / "main.pdf"
total_pages = pages(thesis_pdf)
appendix_page = first_appendix_page(thesis_pdf)
body_pages = (appendix_page - 2) if appendix_page else (total_pages - 1)
appendix_pages_actual = (total_pages - appendix_page + 1) if appendix_page else 0
appendix_pages = appendix_pages_actual
if body_pages > 12:
    raise RuntimeError(f"Assessed body is {body_pages} pages; maximum is 12")

combined = OUTPUT / "thesis_submission_draft.pdf"
shutil.copy2(thesis_pdf, combined)

status = "target reached" if body_pages == 12 else f"{12 - body_pages} body pages remain"
print(f"Built {combined}")
print(f"Cover: 1 page; body: {body_pages}/12 pages; {status}")
if appendix_pages:
    print(f"Appendices: {appendix_pages} pages, not assessed and not counted")
