"""Download the benchmark datasets this project trains and evaluates against.

Usage:
    python scripts/fetch_datasets.py --which arxiv pubmed scitldr multixscience
    python scripts/fetch_datasets.py --which all
"""

import argparse
from pathlib import Path

from datasets import load_dataset

# arxiv/pubmed: no loading script, README YAML maps the "section" name to a
# data directory, so a plain `name=` load works.
#
# scitldr/multixscience: main branch ships a legacy Python loading script that
# `datasets` refuses to execute; `refs/convert/parquet` is the Hub's script-free
# auto-converted mirror, but that branch has no README YAML mapping a config
# name to a directory, so `name=` doesn't resolve there — `data_dir=` selects
# the subdirectory directly instead. If that generic-loader path doesn't work
# either, `_load_via_glob` is a fallback that points straight at the known
# `<data_dir>/<split>/*.parquet` file layout on that branch.
DATASETS = {
    "arxiv": dict(path="ccdv/arxiv-summarization", name="section"),
    "pubmed": dict(path="ccdv/pubmed-summarization", name="section"),
    "scitldr": dict(path="allenai/scitldr", data_dir="AIC", revision="refs/convert/parquet"),
    "multixscience": dict(
        path="bigbio/multi_xscience",
        data_dir="multi_xscience_source",
        revision="refs/convert/parquet",
    ),
}


def _load_via_glob(spec: dict):
    base = f"hf://datasets/{spec['path']}@{spec['revision']}/{spec['data_dir']}"
    data_files = {split: f"{base}/{split}/*.parquet" for split in ("train", "validation", "test")}
    return load_dataset("parquet", data_files=data_files)


def fetch(name: str, out_dir: Path) -> None:
    spec = DATASETS[name]
    try:
        ds = load_dataset(**spec)
    except Exception as exc:
        if "revision" not in spec or "data_dir" not in spec:
            raise
        print(f"{name}: data_dir load failed ({exc}); retrying with explicit parquet glob")
        ds = _load_via_glob(spec)

    target = out_dir / name
    ds.save_to_disk(str(target))
    print(f"{name}: saved {spec['path']} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--which",
        nargs="+",
        choices=[*DATASETS.keys(), "all"],
        default=["all"],
    )
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    names = list(DATASETS.keys()) if "all" in args.which else args.which
    args.out.mkdir(parents=True, exist_ok=True)

    failed = []
    for name in names:
        try:
            fetch(name, args.out)
        except Exception as exc:  # one bad dataset shouldn't block the rest
            print(f"{name}: FAILED — {exc}")
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} dataset(s) failed: {', '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
