"""Keep one native model server alive only while Ariadne owns its stdin pipe."""

from __future__ import annotations

import select
import signal
import subprocess
import sys
import threading
from contextlib import suppress


def supervise(command: list[str]) -> int:
    """An application exit closes stdin; terminate and reap only our own child."""
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda _signum, _frame: stopped.set())
    with subprocess.Popen(command, stdin=subprocess.DEVNULL) as child:  # noqa: S603 — pinned argv from the runtime manager, never a shell.
        try:
            while child.poll() is None and not stopped.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if readable and not sys.stdin.buffer.read(1):
                    break
        finally:
            if child.poll() is None:
                with suppress(ProcessLookupError):
                    child.terminate()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        child.kill()
                    child.wait()
    return max(0, child.returncode)


if __name__ == '__main__':
    if not sys.argv[2:] or sys.argv[1] != '--':
        raise SystemExit('Expected a runtime command after --')
    raise SystemExit(supervise(sys.argv[2:]))
