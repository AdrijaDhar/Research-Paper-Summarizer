"""End-to-end Phase 1 smoke test: fetch a paper from arXiv, parse it with
Docling/Granite-Docling, extract its references with GROBID, caption its
figures with a local VLM, and parse it a second time with MinerU for
side-by-side quality comparison.

Usage:
    python scripts/parse_paper.py 2501.17887
    python scripts/parse_paper.py 2501.17887 --skip-captions  # figures only, no VLM load
    python scripts/parse_paper.py 2501.17887 --skip-mineru
"""

import argparse
import json
from pathlib import Path

from rps.ingestion.arxiv import fetch_pdf
from rps.parsing.caption_figures import caption_figures
from rps.parsing.docling_parser import parse_pdf
from rps.parsing.figures import extract_figures
from rps.parsing.grobid_client import extract_references
from rps.parsing.mineru_parser import parse_pdf_mineru


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arxiv_id")
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--skip-captions", action="store_true")
    parser.add_argument("--skip-mineru", action="store_true")
    args = parser.parse_args()

    pdf_path = fetch_pdf(args.arxiv_id, Path("data/raw/pdfs"))
    print(f"fetched PDF -> {pdf_path}")

    paper_dir = args.out / args.arxiv_id
    doc_dict = parse_pdf(pdf_path, paper_dir)
    print(f"docling parsed {len(doc_dict.get('texts', []))} text elements, "
          f"{len(doc_dict.get('tables', []))} tables -> {paper_dir}")

    references = extract_references(pdf_path)
    (paper_dir / "references.json").write_text(json.dumps(references, indent=2))
    print(f"grobid extracted {len(references)} references -> {paper_dir / 'references.json'}")

    figures = extract_figures(pdf_path, paper_dir)
    print(f"docling extracted {len(figures)} figure images -> {paper_dir / 'figures'}")

    if figures and not args.skip_captions:
        figures = caption_figures(figures)
        print("captioned all figures with mlx-community/Qwen2.5-VL-7B-Instruct-4bit")

    (paper_dir / "figures.json").write_text(json.dumps(figures, indent=2))

    if not args.skip_mineru:
        mineru_path = parse_pdf_mineru(pdf_path, paper_dir)
        print(f"mineru parsed (tier=flash) -> {mineru_path}")
        print("compare against the docling .md in the same directory to see "
              "where each parser does better — that comparison is what will "
              "drive the hard-page routing heuristic in a later entry")


if __name__ == "__main__":
    main()
