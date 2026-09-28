"""High-fidelity fallback parser for pages Docling/Granite-Docling handles
poorly (dense formulas/tables), via the local `mineru` CLI.

Shells out to the CLI rather than importing a Python API — `mineru --help`
and `mineru parse --help`, run directly in this repo's environment, showed no
Python API surface, only the CLI. (An earlier attempt to look this up via web
search returned a fabricated-looking API — `DoclibClient`, a separate
`mineru-kit` binary — that doesn't match the real, locally-confirmed CLI; see
lessons.md Entry 7/8. This wrapper is built only against the confirmed
`mineru`/`mineru parse` output.)
"""

import subprocess
from pathlib import Path


def parse_pdf_mineru(pdf_path: Path, out_dir: Path, *, pages: str = "all", tier: str = "flash") -> Path:
    """Parse `pdf_path` with MinerU, writing Markdown to `out_dir`.

    `tier="flash"`: "standard"/"advanced" require a local parse-server that's
    disabled by default (confirmed by the actual CLI error — "Local
    parse-server is disabled. Use --tier flash or --remote." — not a guess).
    "flash" runs fully locally without that server. Revisit once we know what
    enabling the local parse-server actually buys in quality, and whether
    it's worth the extra setup for what's already a fallback/comparison tool.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{pdf_path.stem}.mineru.md"

    subprocess.run(
        [
            "mineru", "parse", str(pdf_path),
            "--pages", pages,
            "--tier", tier,
            "--output", str(output_path),
            "--wait", "600",
            "--force",
        ],
        check=True,
    )
    return output_path
