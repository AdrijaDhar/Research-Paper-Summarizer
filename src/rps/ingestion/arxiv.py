"""Fetch a paper's PDF from arXiv by its ID."""

from pathlib import Path

import requests

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"


def fetch_pdf(arxiv_id: str, out_dir: Path) -> Path:
    """Download the PDF for `arxiv_id` (e.g. "2501.17887") into `out_dir`.

    Returns the local path to the saved PDF.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{arxiv_id}.pdf"
    if target.exists():
        return target

    response = requests.get(ARXIV_PDF_URL.format(arxiv_id=arxiv_id), timeout=60)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target
