"""Run the full Phase 0-2 pipeline (fetch, parse, build graph) over a small
multi-paper corpus.

SAFETY NOTE — read before running: each paper runs as its own fresh
subprocess (`parse_paper.py`, then `build_graph.py`), not inside one shared
long-running process. An earlier version of this script loaded Granite-Docling,
the standard Docling pipeline, and Qwen2.5-VL-7B all at once and held them
resident across all 7 papers in a single Python process — on top of MinerU
running as a concurrent subprocess and GROBID's Docker container, all
competing for the same unified memory on Apple Silicon, with nothing released
between papers. That crashed the machine on the first real run (see
lessons.md Entry 16). Running one subprocess per paper is slower (each paper
reloads its own models from scratch) but guarantees the OS fully reclaims
memory when each subprocess exits — a far more important property than
saving reload time, and it exactly replicates the single-paper pattern
(`parse_paper.py`) that's already run successfully several times this session.

Usage:
    python scripts/process_corpus.py
    python scripts/process_corpus.py --ids 1706.03762 1810.04805
    python scripts/process_corpus.py --skip-captions --skip-mineru
"""

import argparse
import subprocess
import sys

DEFAULT_CORPUS = [
    "1706.03762",  # Attention Is All You Need
    "1810.04805",  # BERT
    "1910.13461",  # BART
    "1912.08777",  # PEGASUS
    "2106.09685",  # LoRA
    "2305.18290",  # Direct Preference Optimization (DPO)
    "2005.11401",  # Retrieval-Augmented Generation (RAG)
]


def process_paper(arxiv_id: str, skip_captions: bool, skip_mineru: bool) -> None:
    parse_cmd = [sys.executable, "scripts/parse_paper.py", arxiv_id]
    if skip_captions:
        parse_cmd.append("--skip-captions")
    if skip_mineru:
        parse_cmd.append("--skip-mineru")
    subprocess.run(parse_cmd, check=True)
    subprocess.run([sys.executable, "scripts/build_graph.py", arxiv_id], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", default=DEFAULT_CORPUS)
    parser.add_argument("--skip-captions", action="store_true")
    parser.add_argument("--skip-mineru", action="store_true")
    args = parser.parse_args()

    succeeded, failed = [], []
    for arxiv_id in args.ids:
        print(f"\n=== {arxiv_id} ===")
        try:
            process_paper(arxiv_id, args.skip_captions, args.skip_mineru)
            succeeded.append(arxiv_id)
        except subprocess.CalledProcessError as exc:
            print(f"FAILED: {exc}")
            failed.append(arxiv_id)

    print("\n=== corpus summary ===")
    print(f"succeeded ({len(succeeded)}): {succeeded}")
    if failed:
        print(f"failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
