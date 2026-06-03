"""Phase 1 evolutionary-TDD walk for ``Config.scope_closure``.

This file grows one demand at a time. Each cycle adds a new behavioral
demand to ``TestScopeClosure`` (or to a fixture shared by it); the test
file *is* the spec for the directional-closure rule:

    closure(source):
        forward closure if source.depends_on is non-empty
        reverse closure (consumers) if it's a leaf

Fixture nomenclature: ``shared`` is a leaf library; ``product`` depends
on ``shared``; ``extension`` depends on ``product``. The directional
rule is what we're testing — the names just denote dep direction.
"""
from __future__ import annotations

from pathlib import Path


def _write_config(tmp_path: Path, body: str) -> Path:
    p = tmp_path / 'ariadne.yaml'
    p.write_text(body, encoding='utf-8')
    return p


class TestScopeClosure:
    # ---- T1 -----------------------------------------------------------
    # A single source with no dependencies returns just itself. The
    # smallest possible demand: scope_closure exists, returns the
    # source's own name in a frozenset.
    def test_t1_single_source_returns_just_self(
        self, tmp_path: Path,
    ) -> None:
        from config import Config

        cfg_path = _write_config(tmp_path, '''\
sources:
  shared:
    path: /tmp/shared
''')
        cfg = Config(cfg_path)
        assert cfg.scope_closure('shared') == frozenset({'shared'})

    # ---- T2 -----------------------------------------------------------
    # A source with one forward dependency: closure includes the source
    # plus its declared dep. The walk follows ``depends_on`` exactly
    # once (no recursion needed for this case).
    def test_t2_single_forward_dep(self, tmp_path: Path) -> None:
        from config import Config

        cfg_path = _write_config(tmp_path, '''\
sources:
  shared:
    path: /tmp/shared
  product:
    path: /tmp/product
    depends_on: [shared]
''')
        cfg = Config(cfg_path)
        assert cfg.scope_closure('product') == frozenset(
            {'product', 'shared'},
        )
        # T1's demand still holds for shared-in-isolation (no consumers
        # in this fixture's reverse direction beyond product). With
        # product as shared's consumer, shared is a leaf and its closure
        # flips to reverse → T4 will exercise that explicitly; here it
        # produces {shared, product}.
        assert cfg.scope_closure('shared') == frozenset(
            {'shared', 'product'},
        )

    # ---- T3 -----------------------------------------------------------
    # Transitive forward closure: extension.depends_on = [product] only;
    # product.depends_on = [shared]. shared is reachable from extension
    # only via product — closure MUST recurse.
    # (The first draft of this test declared extension.depends_on =
    # [product, shared] explicitly, which let the one-hop implementation
    # satisfy it by accident. The corrected fixture forces actual
    # transitive walking.)
    def test_t3_transitive_forward_closure(self, tmp_path: Path) -> None:
        from config import Config

        cfg_path = _write_config(tmp_path, '''\
sources:
  shared:
    path: /tmp/shared
  product:
    path: /tmp/product
    depends_on: [shared]
  extension:
    path: /tmp/extension
    depends_on: [product]
''')
        cfg = Config(cfg_path)
        assert cfg.scope_closure('extension') == frozenset(
            {'extension', 'product', 'shared'},
        )
        # T2 still holds.
        assert cfg.scope_closure('product') == frozenset(
            {'product', 'shared'},
        )
        # shared is a leaf; reverse closure includes both consumers
        # (product directly, extension via product). T4 elevates this
        # exact assertion to a named demand.
        assert cfg.scope_closure('shared') == frozenset(
            {'shared', 'product', 'extension'},
        )

    # ---- T4 -----------------------------------------------------------
    # The directional flip: shared has no depends_on (it's a leaf), but
    # product and extension both declare shared transitively.
    # scope_closure('shared') must flip to the REVERSE direction and
    # include shared's transitive consumers. This is the load-bearing
    # demand that lets a leaf shared-library see its consumer-side
    # context.
    def test_t4_leaf_flips_to_reverse_closure(self, tmp_path: Path) -> None:
        from config import Config

        cfg_path = _write_config(tmp_path, '''\
sources:
  shared:
    path: /tmp/shared
  product:
    path: /tmp/product
    depends_on: [shared]
  extension:
    path: /tmp/extension
    depends_on: [product]
''')
        cfg = Config(cfg_path)
        # shared is a leaf → reverse closure: shared and everyone that
        # transitively consumes shared (product directly; extension via
        # product).
        assert cfg.scope_closure('shared') == frozenset(
            {'shared', 'product', 'extension'},
        )
        # Non-leaves still use forward closure.
        assert cfg.scope_closure('product') == frozenset(
            {'product', 'shared'},
        )
        assert cfg.scope_closure('extension') == frozenset(
            {'extension', 'product', 'shared'},
        )

    # ---- T5 -----------------------------------------------------------
    # Diamond fixture: top depends_on [left, right]; both left and right
    # depend_on [base]. top's closure must include each of left, right,
    # base exactly once (the dedup is what stops the walk from
    # revisiting base twice via two paths). The reverse direction has
    # the same property: base's reverse closure reaches left, right,
    # top exactly once each.
    def test_t5_diamond_dedup(self, tmp_path: Path) -> None:
        from config import Config

        cfg_path = _write_config(tmp_path, '''\
sources:
  base:
    path: /tmp/base
  left:
    path: /tmp/left
    depends_on: [base]
  right:
    path: /tmp/right
    depends_on: [base]
  top:
    path: /tmp/top
    depends_on: [left, right]
''')
        cfg = Config(cfg_path)
        # Forward: top → {top, left, right, base} (base reached twice,
        # included once).
        assert cfg.scope_closure('top') == frozenset(
            {'top', 'left', 'right', 'base'},
        )
        # Reverse from leaf base: {base, left, right, top} (top reached
        # twice, included once).
        assert cfg.scope_closure('base') == frozenset(
            {'base', 'left', 'right', 'top'},
        )
        # Mid-graph nodes use forward only — left doesn't see right
        # (sibling).
        assert cfg.scope_closure('left') == frozenset({'left', 'base'})
        assert cfg.scope_closure('right') == frozenset({'right', 'base'})

    # ---- T6 -----------------------------------------------------------
    # Unknown / typo'd source raises a clear error: the caller asked for
    # something that isn't configured, and silently returning {source}
    # would mask the typo (and serve a misleading closure downstream).
    # The error names the offending source and lists what IS configured.
    def test_t6_unknown_source_raises(self, tmp_path: Path) -> None:
        import pytest
        from config import Config

        cfg_path = _write_config(tmp_path, '''\
sources:
  shared:
    path: /tmp/shared
  product:
    path: /tmp/product
    depends_on: [shared]
''')
        cfg = Config(cfg_path)
        with pytest.raises(KeyError) as exc:
            cfg.scope_closure('sharednot-a-source')
        message = str(exc.value)
        assert 'sharednot-a-source' in message
        # The error tells the user which sources ARE configured, so they
        # can spot the typo.
        assert 'shared' in message and 'product' in message

    # ---- T7 -----------------------------------------------------------
    # Cycle in depends_on (misconfiguration: alpha → beta → alpha) is
    # rejected with a clear error rather than infinite-looping or
    # silently returning a closure with whatever order BFS happened to
    # visit. The visited-set in T3's BFS already prevents infinite
    # looping, but the user should be told their config is invalid, not
    # silently get a result.
    def test_t7_cycle_in_depends_on_raises(self, tmp_path: Path) -> None:
        import pytest
        from config import Config

        cfg_path = _write_config(tmp_path, '''\
sources:
  alpha:
    path: /tmp/alpha
    depends_on: [beta]
  beta:
    path: /tmp/beta
    depends_on: [alpha]
''')
        cfg = Config(cfg_path)
        with pytest.raises(ValueError) as exc:
            cfg.scope_closure('alpha')
        message = str(exc.value)
        # The error should name both nodes in the cycle so the user
        # can find them in the config.
        assert 'cycle' in message.lower()
        assert 'alpha' in message and 'beta' in message
