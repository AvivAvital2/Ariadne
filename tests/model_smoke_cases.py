"""Synthetic source and retrieval questions shared by all live model profiles."""

CASES = (
    (
        'average',
        'Arithmetic mean',
        'def mean(values):\n'
        '    if not values:\n'
        '        raise ValueError("empty input")\n'
        '    return sum(values) / len(values)\n',
        'How can I calculate the arithmetic average of a list of numbers?',
    ),
    (
        'password-reset',
        'Single-use password reset tokens',
        'def consume_reset_token(tokens, token, now):\n'
        '    user_id, expires_at = tokens.pop(token)\n'
        '    if now >= expires_at:\n'
        '        raise ValueError("expired reset token")\n'
        '    return user_id\n',
        'How does account recovery reject expired links and prevent token reuse?',
    ),
    (
        'cache',
        'Least recently used cache eviction',
        'from collections import OrderedDict\n'
        'def cache_put(cache, key, value, capacity):\n'
        '    cache[key] = value\n'
        '    cache.move_to_end(key)\n'
        '    while len(cache) > capacity:\n'
        '        cache.popitem(last=False)\n',
        'Which entry is evicted when a bounded LRU cache exceeds its capacity?',
    ),
)
