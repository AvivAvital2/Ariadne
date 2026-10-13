"""Reviewed synthetic fixtures and execution oracles for source verification.

Only these checked-in source literals are executed. Model-generated code is
never executed. Oracle results and document labels are not sent to the reviewer.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass

from tests.model_smoke_cases import CASES


@dataclass(frozen=True)
class Case:
    name: str
    split: str
    source: str
    function: str
    probes: tuple
    correct: str
    incorrect: str
    question: str
    ordered: bool = False


# Each probe is (arguments, expected return, expected exception, first argument
# after the call). These expectations were specified independently of any LLM.
CASES_BY_NAME = {
    c.name: c
    for c in (
        Case(
            'cache',
            'development',
            CASES[2][2],
            'cache_put',
            (
                (({}, 'a', 1, -1), None, 'KeyError', {}),
                (({}, 'a', 1, 0), None, None, {}),
                (({}, 'a', 1, 1), None, None, {'a': 1}),
            ),
            'A zero capacity leaves the cache empty and returns None. A negative capacity empties the cache and then raises KeyError on another popitem call. The exception terminates the call; it does not loop forever. Only writes change recency; a plain read does not.',
            'Zero and negative capacity both leave the cache empty and return None without an exception. Only writes change recency; a plain read does not.',
            'In cache_put, what happens with capacity=-1, and what state is the cache left in?',
            ordered=True,
        ),
        Case(
            'guarded_cache',
            'development',
            CASES[2][2].replace(
                'while len(cache) > capacity:', 'while cache and len(cache) > capacity:'
            ),
            'cache_put',
            (
                (({}, 'a', 1, -1), None, None, {}),
                (({}, 'a', 1, 0), None, None, {}),
                (({}, 'a', 1, 1), None, None, {'a': 1}),
            ),
            'Zero and negative capacity both empty the cache and return None. The cache truthiness guard prevents popping an empty OrderedDict. Positive capacity retains up to that many entries. The function changes the caller-owned cache in place.',
            'Negative capacity empties the cache and then raises KeyError on another popitem call. Positive capacity retains up to that many entries. The function changes the caller-owned cache in place.',
            'In cache_put with the cache truthiness guard, does capacity=-1 raise an exception?',
            ordered=True,
        ),
        Case(
            'queue',
            'development',
            'def take(items, count):\n'
            '    result = []\n'
            '    while len(result) != count:\n'
            '        result.append(items.pop(0))\n'
            '    return result\n',
            'take',
            (
                (([1, 2], -1), None, 'IndexError', []),
                (([1, 2], 0), [], None, [1, 2]),
                (([1, 2], 1), [1], None, [2]),
                (([1], 2), None, 'IndexError', []),
            ),
            'take removes items from the front of the caller-owned list. Zero count returns an empty list without changing the input. A negative count or a count larger than the list exhausts the input and raises IndexError; it does not return a partial final batch.',
            'take removes items from the front of the caller-owned list. Zero count returns an empty list without changing the input. A count larger than the list returns the available items as a partial final batch without an exception.',
            'Does take([1], 2) return a partial batch, and what happens to the input list?',
        ),
        Case(
            'token_expiry',
            'heldout',
            'def consume_token(tokens, token, now):\n'
            '    user, deadline = tokens.pop(token)\n'
            '    if now >= deadline:\n'
            '        raise ValueError("expired")\n'
            '    return user\n',
            'consume_token',
            (
                (({'t': ('u', 10)}, 't', 10), None, 'ValueError', {}),
                (({'t': ('u', 10)}, 't', 9), 'u', None, {}),
                (({}, 't', 9), None, 'KeyError', {}),
            ),
            'consume_token removes the token before checking its deadline. A token is expired when now equals the deadline, and ValueError is raised after removal. Earlier calls return the user and consume the token. A missing token raises KeyError.',
            'consume_token checks the deadline before removing the token. An expired token raises ValueError and remains available in the mapping. Earlier calls return the user and consume the token. A missing token raises KeyError.',
            'In consume_token, what happens when now equals the deadline, and is the token retained?',
        ),
        Case(
            'job_claim',
            'heldout',
            'def claim_job(jobs, key):\n'
            '    job = jobs.pop(key)\n'
            '    if not job["ready"]:\n'
            '        raise RuntimeError("not ready")\n'
            '    return job["payload"]\n',
            'claim_job',
            (
                (({'j': {'ready': False, 'payload': 7}}, 'j'), None, 'RuntimeError', {}),
                (({'j': {'ready': True, 'payload': 7}}, 'j'), 7, None, {}),
                (({}, 'j'), None, 'KeyError', {}),
            ),
            'claim_job removes the job before checking readiness. A not-ready job raises RuntimeError and is no longer stored. A ready job returns its payload after removal. An absent key raises KeyError before the readiness check.',
            'claim_job checks readiness before removing a job. A not-ready job raises RuntimeError and remains stored. A ready job returns its payload after removal. An absent key raises KeyError before the readiness check.',
            'Does claim_job preserve a not-ready job after raising RuntimeError?',
        ),
        Case(
            'optional_remove',
            'heldout',
            'def remove_optional(mapping, key):\n    return mapping.pop(key, None)\n',
            'remove_optional',
            (
                (({}, 'x'), None, None, {}),
                (({'x': 3}, 'x'), 3, None, {}),
                (({'x': None}, 'x'), None, None, {}),
            ),
            'remove_optional returns and removes a present value. An absent key returns None without an exception because pop receives a default. A present value of None also returns None and is removed, so the return value alone cannot distinguish those cases.',
            'remove_optional returns and removes a present value. An absent key raises KeyError because there is no matching entry. A present value of None returns None and is removed.',
            'Does remove_optional({}, "x") raise an exception or return a value?',
        ),
        Case(
            'index_lookup',
            'heldout',
            'def at_index(items, index):\n    return items[index]\n',
            'at_index',
            (
                (([10, 20], -1), 20, None, [10, 20]),
                (([10, 20], 0), 10, None, [10, 20]),
                (([10, 20], -3), None, 'IndexError', [10, 20]),
            ),
            'at_index uses Python list indexing without mutating the list. Index -1 returns the final item of a nonempty list. An index outside the valid positive or negative range raises IndexError. There is no separate non-negative-index check.',
            'at_index uses Python list indexing without mutating the list. Every negative index raises IndexError, including -1 on a nonempty list. Positive indices select the corresponding item.',
            'What does at_index([10, 20], -1) return, and does it modify the list?',
        ),
        Case(
            'range_pop',
            'heldout',
            'def pop_many(items, count):\n'
            '    result = []\n'
            '    for _ in range(count):\n'
            '        result.append(items.pop())\n'
            '    return result\n',
            'pop_many',
            (
                (([1, 2], -1), [], None, [1, 2]),
                (([1, 2], 0), [], None, [1, 2]),
                (([1, 2], 1), [2], None, [1]),
                (([1], 2), None, 'IndexError', []),
            ),
            'pop_many removes items from the end, returning them in pop order. Zero and negative counts produce an empty range, return an empty list, and preserve the input. A positive count exceeding the available items drains the list and raises IndexError.',
            'pop_many removes items from the end, returning them in pop order. A negative count keeps popping until the input is empty and then raises IndexError. Zero count returns an empty list and preserves the input.',
            'What does pop_many([1, 2], -1) return, and does it modify the input list?',
        ),
    )
}


def observe_case(case: Case) -> list[dict]:
    namespace = {}
    exec(compile(case.source, case.name + '.py', 'exec'), namespace)  # noqa: S102 - checked-in fixtures, never model output
    observations = []
    for original_args, result, error, state in case.probes:
        args = list(copy.deepcopy(original_args))
        if case.ordered:
            args[0] = OrderedDict(args[0])
        returned, raised = None, None
        try:
            returned = namespace[case.function](*args)
        except (KeyError, IndexError, ValueError, RuntimeError) as exc:
            raised = type(exc).__name__
        actual = {'return': returned, 'exception': raised, 'state_after': args[0]}
        expected = {'return': result, 'exception': error, 'state_after': state}
        if actual != expected:
            raise AssertionError(f'Execution oracle mismatch in {case.name}: {actual!r}')
        observations.append({'arguments': original_args, **actual})
    return observations


def control_document(case: Case, *, correct: bool) -> str:
    return (
        f'## Question\nHow does {case.function} behave?\n\n## Answer\n'
        + (case.correct if correct else case.incorrect)
        + '\n'
    )
