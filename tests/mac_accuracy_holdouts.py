"""Reserved confirmation fixtures, independent of the first prompt-recipe trial.

Only these checked-in source literals are executed. Never execute generated code
or transmit the observations/required facts to a generator. These cases were
reserved before observing the first trial's output; that historical label does
not make them untouched holdouts for later tuning after their use.
"""

from __future__ import annotations

import hashlib

SOURCES = {
    'late_bound_callbacks': '''def make_checks(values):
    checks = []
    for value in values:
        checks.append(lambda candidate: candidate == value)
    return checks
''',
    'eager_default': '''def ensure_entry(entries, key, factory):
    return entries.setdefault(key, factory())
''',
}

REQUIRED_FACTS = {
    'late_bound_callbacks': (
        'make_checks returns a list of callbacks; empty input produces an empty list.',
        'Creating the callbacks does not execute their comparisons; a comparison runs when a callback is called.',
        'The callbacks share the late-bound loop variable rather than capturing a separate value for each iteration.',
        'For input [1, 2], both returned callbacks compare their argument with 2: both reject 1 and accept 2.',
    ),
    'eager_default': (
        'factory() is evaluated before setdefault executes, including when the key already exists.',
        'For an ordinary dict and a factory that does not mutate it, an existing value, including None, is returned without being replaced.',
        'If the key is absent when setdefault executes, the factory result is stored and that same object is returned.',
        'A factory exception propagates before setdefault executes, even for an existing key.',
        'With a non-mutating raising factory the dict is unchanged; side effects a factory performs before raising are not rolled back.',
    ),
}


def _function(case: str, name: str):
    namespace = {}
    # Trusted source fixtures only, never generated documentation or model code.
    exec(compile(SOURCES[case], case + '.py', 'exec'), namespace)  # noqa: S102
    return namespace[name]


def _observe_callbacks() -> list[dict]:
    function = _function('late_bound_callbacks', 'make_checks')
    empty = {'scenario': 'empty_input', 'callback_count': len(function([]))}
    values = [1, 2]
    callbacks = function(values)
    multiple = {
        'scenario': 'two_values',
        'callback_count': len(callbacks),
        'input_after_creation': list(values),
        'matches_one': [callback(1) for callback in callbacks],
        'matches_two': [callback(2) for callback in callbacks],
    }
    comparisons = []

    class Candidate:
        __hash__ = None

        def __eq__(self, other):
            comparisons.append(other)
            return True

    candidate = Candidate()
    callbacks = function([1, 2])
    timing = {'scenario': 'comparison_timing', 'comparisons_at_creation': list(comparisons)}
    callbacks[0](candidate)
    timing['comparisons_after_first_callback'] = list(comparisons)
    callbacks[1](candidate)
    timing['comparisons_after_second_callback'] = list(comparisons)
    return [empty, multiple, timing]


def _observe_default() -> list[dict]:
    function = _function('eager_default', 'ensure_entry')
    observations = []
    for label, entries in (
        ('existing_value', {'key': 'kept'}),
        ('existing_none', {'key': None}),
        ('missing_value', {}),
    ):
        calls, created = [], {'created': True}

        def factory(calls=calls, created=created):
            calls.append('factory')
            return created

        returned = function(entries, 'key', factory)
        observed = {'scenario': label, 'factory_calls': len(calls), 'returned': returned}
        if label == 'missing_value':
            observed.update(
                returned_is_created=returned is created, stored_is_created=entries['key'] is created
            )
        observations.append({**observed, 'state': entries})

    for label, entries, mutate in (
        ('factory_error_existing', {'key': 'kept'}, False),
        ('factory_error_missing', {}, False),
        ('factory_side_effect_then_error', {}, True),
    ):
        calls = []

        def failing_factory(calls=calls, mutate=mutate, entries=entries):
            calls.append('factory')
            if mutate:
                entries['side_effect'] = True
            raise RuntimeError('factory failed')

        observed = {'scenario': label}
        try:
            function(entries, 'key', failing_factory)
        except RuntimeError as error:
            observed['exception'] = type(error).__name__
        observations.append({**observed, 'factory_calls': len(calls), 'state': entries})
    return observations


def frozen_cases() -> dict:
    """Return fresh, JSON-serializable source/oracle snapshots for later freezing."""
    return {
        name: {
            'split': 'reserved_confirmation',
            'reservation_status': 'Untouched by candidate tuning before first trial',
            'source': SOURCES[name],
            'source_sha256': hashlib.sha256(SOURCES[name].encode()).hexdigest(),
            'required_facts': list(REQUIRED_FACTS[name]),
            'required_behavior': ' '.join(REQUIRED_FACTS[name]),
            'oracle': observe(),
        }
        for name, observe in (
            ('late_bound_callbacks', _observe_callbacks),
            ('eager_default', _observe_default),
        )
    }
