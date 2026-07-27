"""Guardrail schema + structural code-signal evaluation (slice e1).

The §6 model: a guardrail is repo-agnostic prose (recommendation,
rationale, citation) plus a machine-checkable code-signal. Structural
signals are a CLOSED set of predicates over the catalog, every result
carrying fired + evidence + a confidence label. NL-judged signals exist
in the schema but evaluate to an honest ``unevaluated`` until the LLM
path lands (e2). Two probe-verified catalog caveats shape the DSL
(IMPLEMENT.md, 2026-07-08): scans must filter subtypes (local-var
over-capture) and multi-line signatures are stored truncated (hence
``body_contains``). Design: designs/spool-environment-plugin.md §6 · §9.
"""
from dataclasses import dataclass, field
from pathlib import Path

STRUCTURAL_CONFIDENCE = 'verified (structural)'
NL_UNEVALUATED = 'unevaluated (nl)'
# CRIT-8: a runtime read fault (target file stale/moved since indexing) is
# isolated at the signal — non-fired, labeled, reason in evidence — never
# propagated. Distinct from a SCHEMA fault (unknown kind / missing key),
# which stays loud (author error).
SIGNAL_ERROR_CONFIDENCE = 'signal-error (structural)'

_GUARDRAIL_REQUIRED_FIELDS = (
    'name', 'kind', 'recommendation', 'rationale', 'citation',
    'signal_type', 'signal',
)


class GuardrailError(Exception):
    """A guardrail schema/signal violation — always loud, never masked."""


@dataclass(frozen=True)
class SignalResult:
    fired: bool
    evidence: tuple = ()
    confidence: str = STRUCTURAL_CONFIDENCE


class CatalogView:
    """The signal DSL's read surface over a Library's catalog.

    Element metadata shape (as the catalog stores it): ``qualified_name``,
    ``signature``, ``subtype``, ``file``, ``location.line_start/line_end``.
    """

    def __init__(self, library, *, source_name: str):
        self._source_name = source_name
        self._elements = []
        for meta in library.list_documents_lite():
            if meta.source_name != source_name:
                continue
            if not meta.metadata.get('qualified_name'):
                continue
            element = dict(meta.metadata)
            # Live-catalog shape (validated 2026-07-08): the file path is
            # the document's source_files entry, not a metadata key.
            if not element.get('file') and meta.source_files:
                element['file'] = meta.source_files[0]
            self._elements.append(element)

    def lookup(self, pattern: str) -> list[dict]:
        """Exact qualified-name match, else suffix match."""
        exact = [
            e for e in self._elements
            if e.get('qualified_name') == pattern
        ]
        if exact:
            return exact
        return [
            e for e in self._elements
            if str(e.get('qualified_name', '')).endswith('.' + pattern)
        ]

    def scan(self, prefix: str, subtypes) -> list[dict]:
        allowed = set(subtypes)
        return [
            e for e in self._elements
            if str(e.get('qualified_name', '')).startswith(prefix)
            and e.get('subtype') in allowed
        ]

    @staticmethod
    def body(element: dict) -> str:
        """Current source text at the element's cataloged location."""
        location = element.get('location') or {}
        start = int(location.get('line_start', 0))
        end = int(location.get('line_end', 0))
        try:
            lines = Path(element['file']).read_text(
                encoding='utf-8',
            ).splitlines()
        except (KeyError, OSError, UnicodeDecodeError) as exc:
            raise GuardrailError(
                f"cannot read body for "
                f"{element.get('qualified_name')!r}: {exc}",
            ) from exc
        return '\n'.join(lines[max(start - 1, 0):end])


def _cite(element: dict) -> str:
    location = element.get('location') or {}
    return (
        f"{element.get('qualified_name')} "
        f"({element.get('file')}:{location.get('line_start')})"
    )


def _eval_symbol_exists(signal, view):
    matches = view.lookup(signal['pattern'])
    return SignalResult(
        fired=bool(matches),
        evidence=tuple(_cite(e) for e in matches),
    )


def _eval_signature_contains(signal, view):
    matches = [
        e for e in view.lookup(signal['symbol'])
        if signal['needle'] in str(e.get('signature', ''))
    ]
    return SignalResult(
        fired=bool(matches),
        evidence=tuple(_cite(e) for e in matches),
    )


def _eval_body_contains(signal, view):
    # CRIT-8: read each candidate's body defensively — a stale/missing file
    # degrades that element to a recorded note, never crashing the signal
    # (and thus never the whole capability profile). KeyError on the signal
    # dict itself (schema fault) is NOT caught here — it surfaces loudly via
    # evaluate_signal.
    needle = signal['needle']
    symbol = signal['symbol']
    fired_evidence = []
    error_notes = []
    for e in view.lookup(symbol):
        try:
            body = view.body(e)
        except GuardrailError as exc:
            error_notes.append(str(exc))
            continue
        if needle in body:
            fired_evidence.append(_cite(e))
    if fired_evidence:
        return SignalResult(fired=True, evidence=tuple(fired_evidence))
    if error_notes:
        return SignalResult(
            fired=False, evidence=tuple(error_notes),
            confidence=SIGNAL_ERROR_CONFIDENCE,
        )
    return SignalResult(fired=False)


def _eval_signature_scan(signal, view):
    matches = [
        e for e in view.scan(signal['prefix'], signal['subtypes'])
        if signal['needle'] in str(e.get('signature', ''))
    ]
    return SignalResult(
        fired=bool(matches),
        evidence=tuple(_cite(e) for e in matches),
    )


def _eval_all_of(signal, view):
    results = [evaluate_signal(s, view) for s in signal['signals']]
    return SignalResult(
        fired=all(r.fired for r in results),
        evidence=tuple(e for r in results for e in r.evidence),
    )


def _eval_any_of(signal, view):
    results = [evaluate_signal(s, view) for s in signal['signals']]
    return SignalResult(
        fired=any(r.fired for r in results),
        evidence=tuple(e for r in results if r.fired for e in r.evidence),
    )


_SIGNAL_KINDS = {
    'symbol_exists': _eval_symbol_exists,
    'signature_contains': _eval_signature_contains,
    'body_contains': _eval_body_contains,
    'signature_scan': _eval_signature_scan,
    'all_of': _eval_all_of,
    'any_of': _eval_any_of,
}


def evaluate_signal(signal: dict, view: CatalogView) -> SignalResult:
    """Evaluate one structural signal against the catalog — fired +
    evidence, always labeled."""
    kind = signal.get('kind')
    evaluator = _SIGNAL_KINDS.get(kind)
    if evaluator is None:
        raise GuardrailError(
            f'unknown signal kind {kind!r}; structural kinds: '
            f'{sorted(_SIGNAL_KINDS)}',
        )
    try:
        return evaluator(signal, view)
    except KeyError as exc:
        raise GuardrailError(
            f'signal {kind!r} missing required key: {exc}',
        ) from exc


@dataclass(frozen=True)
class Guardrail:
    name: str
    kind: str                       # 'antipattern' | 'method'
    recommendation: str
    rationale: str
    citation: str
    signal_type: str                # 'structural' | 'nl'
    signal: dict = field(default_factory=dict)
    requires: tuple = ()
    provides: tuple = ()
    spans: tuple = ()               # patterns-allowed granularity (§16)


@dataclass(frozen=True)
class GuardrailResult:
    guardrail: Guardrail
    result: SignalResult


def load_guardrails(path) -> list[Guardrail]:
    """Load a guardrail catalog YAML; loud on any missing required field."""
    from spools import load_yaml_mapping
    data = load_yaml_mapping(path, GuardrailError)
    entries = data.get('guardrails') or []
    guardrails = []
    for raw in entries:
        missing = [f for f in _GUARDRAIL_REQUIRED_FIELDS if not raw.get(f)]
        if missing:
            raise GuardrailError(
                f"guardrail {raw.get('name', '<unnamed>')!r} missing "
                f"required field(s): {', '.join(missing)}",
            )
        guardrails.append(Guardrail(
            name=raw['name'],
            kind=raw['kind'],
            recommendation=raw['recommendation'],
            rationale=raw['rationale'],
            citation=raw['citation'],
            signal_type=raw['signal_type'],
            signal=dict(raw['signal']),
            requires=tuple(raw.get('requires') or ()),
            provides=tuple(raw.get('provides') or ()),
            spans=tuple(raw.get('spans') or ()),
        ))
    return guardrails


def evaluate_guardrail(guardrail: Guardrail, view: CatalogView) -> GuardrailResult:
    """Evaluate one guardrail's signal; NL signals stay honestly unevaluated."""
    if guardrail.signal_type == 'nl':
        return GuardrailResult(
            guardrail=guardrail,
            result=SignalResult(fired=False, confidence=NL_UNEVALUATED),
        )
    return GuardrailResult(
        guardrail=guardrail,
        result=evaluate_signal(guardrail.signal, view),
    )


_SIGNAL_SYMBOL_FIELDS = ('pattern', 'symbol', 'prefix')


def _signal_symbol_refs(signal: dict) -> list:
    """Every non-empty symbol/namespace a structural signal keys on, recursing
    through ``all_of``/``any_of``. An empty ``prefix`` (scan-all) names no
    namespace, so it contributes nothing."""
    refs = []
    for field_name in _SIGNAL_SYMBOL_FIELDS:
        value = signal.get(field_name)
        if value:
            refs.append(str(value))
    for sub in signal.get('signals', ()) or ():
        refs.extend(_signal_symbol_refs(sub))
    return refs


def validate_spool_guardrails(guardrails, *, allowed_prefixes) -> None:
    """Reject any SHIPPED spool guardrail keyed on non-environment symbols.

    A spool encodes ENVIRONMENT knowledge, so a structural signal may name
    only the environment's own API packages (``allowed_prefixes`` — e.g.
    ``pyspark``/``mlflow``/``delta`` for databricks). A signal naming a
    caller's own symbols is consumer knowledge and must never ship in a
    pack. Prose (``nl``) guardrails name no symbols and always pass. Loud on
    violation — an author error, caught at build, never masked.
    """
    allowed = tuple(allowed_prefixes)
    for guardrail in guardrails:
        if guardrail.signal_type != 'structural':
            continue
        for ref in _signal_symbol_refs(guardrail.signal):
            if not any(ref == p or ref.startswith(p + '.') for p in allowed):
                raise GuardrailError(
                    f'spool guardrail {guardrail.name!r}: structural signal '
                    f'references {ref!r}, outside the environment API allowlist '
                    f'{sorted(allowed)} — a spool guardrail must key on the '
                    f'environment (or be prose `nl`); consumer symbols must not '
                    f'ship in a spool.',
                )
