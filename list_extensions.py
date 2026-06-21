#!/usr/bin/env python3
"""Walk a directory recursively and collect all unique file extensions."""

import argparse
from pathlib import Path


def collect_extensions(root: Path) -> set[str]:
    """Return the set of unique file extensions under ``root`` (recursive).

    Extensions are lowercased and include the leading dot (e.g. ``.py``).
    Files without an extension are reported as ``""``.
    """
    return {path.suffix.lower() for path in root.rglob("*") if path.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory to walk")
    args = parser.parse_args()

    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")

    for ext in sorted(collect_extensions(args.directory)):
        print(ext or "(no extension)")


if __name__ == "__main__":
    main()
