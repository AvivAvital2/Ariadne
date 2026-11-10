"""Independent expectations for reserved source-only confirmation fixtures."""

import hashlib
import json

from tests.mac_accuracy_holdouts import frozen_cases


def test_late_bound_callbacks_observations_match_independent_expectations():
    case = frozen_cases()['late_bound_callbacks']
    assert case['oracle'] == [
        {'scenario': 'empty_input', 'callback_count': 0},
        {
            'scenario': 'two_values',
            'callback_count': 2,
            'input_after_creation': [1, 2],
            'matches_one': [False, False],
            'matches_two': [True, True],
        },
        {
            'scenario': 'comparison_timing',
            'comparisons_at_creation': [],
            'comparisons_after_first_callback': [2],
            'comparisons_after_second_callback': [2, 2],
        },
    ]


def test_eager_default_observations_match_independent_expectations():
    case = frozen_cases()['eager_default']
    assert case['oracle'] == [
        {
            'scenario': 'existing_value',
            'factory_calls': 1,
            'returned': 'kept',
            'state': {'key': 'kept'},
        },
        {
            'scenario': 'existing_none',
            'factory_calls': 1,
            'returned': None,
            'state': {'key': None},
        },
        {
            'scenario': 'missing_value',
            'factory_calls': 1,
            'returned': {'created': True},
            'returned_is_created': True,
            'stored_is_created': True,
            'state': {'key': {'created': True}},
        },
        {
            'scenario': 'factory_error_existing',
            'factory_calls': 1,
            'exception': 'RuntimeError',
            'state': {'key': 'kept'},
        },
        {
            'scenario': 'factory_error_missing',
            'factory_calls': 1,
            'exception': 'RuntimeError',
            'state': {},
        },
        {
            'scenario': 'factory_side_effect_then_error',
            'factory_calls': 1,
            'exception': 'RuntimeError',
            'state': {'side_effect': True},
        },
    ]


def test_confirmation_cases_are_json_serializable_and_independent_snapshots():
    cases = frozen_cases()
    assert tuple(cases) == ('late_bound_callbacks', 'eager_default')
    assert json.loads(json.dumps(cases)) == cases
    for case in cases.values():
        assert case['split'] == 'reserved_confirmation'
        assert case['reservation_status'] == 'Untouched by candidate tuning before first trial'
        assert case['source_sha256'] == hashlib.sha256(case['source'].encode()).hexdigest()
        assert case['required_behavior'] == ' '.join(case['required_facts'])
        assert len(case['required_facts']) >= 3
    original = frozen_cases()
    cases['late_bound_callbacks']['oracle'][1]['matches_one'][0] = True
    cases['eager_default']['oracle'][2]['state']['key']['created'] = False
    cases['eager_default']['required_facts'].clear()
    assert frozen_cases() == original
