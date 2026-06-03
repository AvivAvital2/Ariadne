from __future__ import annotations

from slack_bridge.pool import SessionPool


class _FakeRunner:
    def __init__(self, name, *, fail_close=False):
        self.name = name
        self.closed = 0
        self.fail_close = fail_close

    async def ask(self, text):
        return f'{self.name}:{text}'

    async def aclose(self):
        self.closed += 1
        if self.fail_close:
            import asyncio
            raise asyncio.CancelledError


def _make_pool(max_size=2, idle_ttl=100.0, clock=None, fail_close=False):
    created = []

    def factory(thread_ts, seed=None):  # noqa: ARG001 — matches the pool's factory contract
        r = _FakeRunner(thread_ts, fail_close=fail_close)
        created.append(r)
        return r

    pool = SessionPool(
        runner_factory=factory,
        max_size=max_size,
        idle_ttl=idle_ttl,
        clock=clock or (lambda: 0.0),
    )
    return pool, created


async def test_pool_reuse_lru_and_size_bound():
    pool, created = _make_pool(max_size=2)

    s1 = await pool.get_or_create('t1')
    assert await pool.get_or_create('t1') is s1   # warm reuse, no new runner
    assert len(created) == 1

    await pool.get_or_create('t2')                # {t1, t2}
    await pool.get_or_create('t1')                # touch t1 → t2 becomes LRU
    await pool.get_or_create('t3')                # over cap → evict LRU (t2)

    assert 't1' in pool and 't3' in pool
    assert 't2' not in pool
    assert created[1].name == 't2' and created[1].closed == 1   # evicted runner torn down once


async def test_pool_evicts_idle_but_skips_in_flight():
    now = {'t': 0.0}
    pool, created = _make_pool(max_size=10, idle_ttl=100.0, clock=lambda: now['t'])

    await pool.get_or_create('t1')   # last_activity = 0
    now['t'] = 50.0
    s2 = await pool.get_or_create('t2')   # last_activity = 50
    now['t'] = 200.0                       # both now idle past the 100s TTL

    await s2.lock.acquire()                # t2 has an in-flight turn
    try:
        await pool.evict_idle()
    finally:
        s2.lock.release()

    assert 't1' not in pool                # idle + unlocked → evicted
    assert 't2' in pool                    # idle but locked → kept
    assert created[0].closed == 1
    assert created[1].closed == 0


async def test_pool_teardown_tolerates_cancellation():
    # SDK bug #890: client/MCP teardown can raise CancelledError/BaseExceptionGroup.
    pool, created = _make_pool(max_size=1, fail_close=True)

    await pool.get_or_create('t1')
    # Creating t2 evicts t1, whose aclose() raises — it must be swallowed so the
    # new session is still created and the pool keeps working.
    await pool.get_or_create('t2')

    assert 't1' not in pool and 't2' in pool
    assert created[0].closed == 1   # teardown was attempted once


async def test_get_or_create_skips_in_flight_lru():
    # The LRU-eviction path in get_or_create must skip a session whose turn
    # is in flight (lock held) — the same guard evict_idle has. Otherwise a
    # new arrival at capacity tears down a live session mid-`ask`, reaping its
    # MCP subprocess underneath the running turn.
    pool, created = _make_pool(max_size=2)

    s1 = await pool.get_or_create('t1')   # created[0]; least-recently-used
    await pool.get_or_create('t2')        # created[1]

    await s1.lock.acquire()               # t1 (the LRU) has an in-flight turn
    try:
        await pool.get_or_create('t3')    # over cap → must evict t2, NOT the locked t1
    finally:
        s1.lock.release()

    assert 't1' in pool                   # locked LRU preserved (not torn down mid-turn)
    assert 't3' in pool                   # new session still created
    assert created[0].closed == 0         # t1's runner was never aclosed
    assert 't2' not in pool and created[1].closed == 1   # unlocked LRU evicted instead
