"""Fresh synthetic inputs frozen before the first live infrastructure run.

Every source has the same relative filename to expose accidental source mixing.
Questions and expected implementation invariants are not generation instructions.
"""

# ruff: noqa: Q001 - source literals follow the repository formatter.

CASES = (
    (
        'ledger',
        'explanation',
        """from decimal import Decimal

def total_due(lines, discount=Decimal("0")):
    subtotal = sum((Decimal(str(price)) * quantity for price, quantity in lines), Decimal("0"))
    return max(Decimal("0"), subtotal - discount)
""",
        'How is the invoice total calculated?',
    ),
    (
        'schedule',
        'qa',
        """def insert_window(windows, candidate):
    start, end = candidate
    if end <= start:
        raise ValueError("empty interval")
    if any(start < old_end and old_start < end for old_start, old_end in windows):
        return False
    windows.append(candidate)
    windows.sort()
    return True
""",
        'How are overlapping booking intervals handled?',
    ),
    (
        'labels',
        'gotcha',
        """import unicodedata

def unique_labels(values):
    seen = set()
    result = []
    for value in values:
        label = unicodedata.normalize("NFC", value).strip()
        key = label.casefold()
        if label and key not in seen:
            seen.add(key)
            result.append(label)
    return result
""",
        'How are Unicode labels normalized and deduplicated?',
    ),
    (
        'archive',
        'architecture',
        """from pathlib import PurePosixPath

def archive_path(root, member):
    name = PurePosixPath(member)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("unsafe member")
    return PurePosixPath(root) / name
""",
        'How are archive member paths validated?',
    ),
    (
        'routing',
        'diagram',
        """def route_event(event, handlers):
    kind = event.get("kind", "unknown")
    handler = handlers.get(kind, handlers.get("*"))
    if handler is None:
        return {"status": "unhandled", "kind": kind}
    return {"status": "handled", "result": handler(event)}
""",
        'How does event routing choose its handler?',
    ),
    (
        'text',
        'explanation',
        """def split_records(text, separator="|"):
    if not separator:
        raise ValueError("separator must not be empty")
    return [part.strip() for part in text.split(separator) if part.strip()]
""",
        'How are delimited text records split and cleaned?',
    ),
)
