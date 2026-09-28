"""Build and save the heterogeneous knowledge graph for an already-parsed paper.

Usage:
    python scripts/build_graph.py 2501.17887
"""

import argparse
import json
from pathlib import Path

from rps.graph.build_kg import build_graph, save_graph, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arxiv_id")
    parser.add_argument("--dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    paper_dir = args.dir / args.arxiv_id
    g = build_graph(paper_dir, args.arxiv_id)

    out_path = paper_dir / "graph.json"
    save_graph(g, out_path)

    print(f"saved graph -> {out_path}")
    print(json.dumps(summarize(g), indent=2))


if __name__ == "__main__":
    main()
