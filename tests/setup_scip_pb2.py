"""Fetch SCIP proto + generate Python bindings (SCIP plan, Phase A.4).

Run once before using the SCIP-backed extractor. This script:
  1. Downloads scip.proto from github.com/sourcegraph/scip @ v0.5.2.
  2. Saves it to ``docgen/scip/scip.proto``.
  3. Invokes ``protoc`` to generate ``docgen/scip/scip_pb2.py``.

Requirements:
  - ``protoc`` on PATH (install via ``brew install protobuf`` on macOS).
  - Network access to GitHub for the initial proto download.
  - The generated ``scip_pb2.py`` is checked in so collaborators don't
    need protoc to run; this script only needs to be re-run when the
    pinned version changes.

Usage:
    uv run python tests/setup_scip_pb2.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

SCIP_VERSION = 'v0.5.2'
PROTO_URL = (
    f'https://raw.githubusercontent.com/sourcegraph/scip/{SCIP_VERSION}/scip.proto'
)
TARGET_DIR = Path(__file__).resolve().parent.parent / 'docgen' / 'scip'


def main() -> int:
    if shutil.which('protoc') is None:
        print(
            'ERROR: protoc not found on PATH. Install it (e.g. '
            '`brew install protobuf` on macOS) and re-run.',
            file=sys.stderr,
        )
        return 1

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    proto_path = TARGET_DIR / 'scip.proto'

    print(f'Downloading {PROTO_URL}')
    with urllib.request.urlopen(PROTO_URL) as resp:
        proto_path.write_bytes(resp.read())
    print(f'Wrote {proto_path}')

    print('Running protoc → scip_pb2.py')
    cmd = [
        'protoc',
        f'--proto_path={TARGET_DIR}',
        f'--python_out={TARGET_DIR}',
        str(proto_path),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f'ERROR: protoc failed (exit {result.returncode})', file=sys.stderr)
        return result.returncode

    pb2_path = TARGET_DIR / 'scip_pb2.py'
    if not pb2_path.exists():
        print(f'ERROR: expected output not found at {pb2_path}', file=sys.stderr)
        return 2

    print(f'Generated {pb2_path}')
    print('\nDone. The SCIP-backed extractor can now import docgen.scip.scip_pb2.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
