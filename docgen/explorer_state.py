"""Tier 3 of the dry-run explorer: the headless interactive state model.

The terminal UI isn't unit-testable in this sandbox, so all the logic lives
here in a fully-tested :class:`ExplorerState`; the termios view
(``run_explorer``) and the ``cmd_dry_run`` wiring are thin and smoke-tested
only. ``ExplorerState`` composes Tier 1's :class:`~docgen.scan_tree.ScanNode`
tree with Tier 2's ``{rel_path: NodeCost}`` cost map and supports:
navigate in/up, toggle excludes at any depth, an instant live total, and the
``exclude_dirs`` / ``exclude`` rule set to persist on confirm.

Exclusion is an **override map** (``rel_path -> excluded/kept``) with
nearest-ancestor-wins resolution: excluding a directory excludes its whole
subtree, but you can then *keep* (carve out) a child, and keeping a directory
clears stale child selections. ``excluded_rules`` then projects the effective
state onto the config model below — a fully-excluded directory collapses to one
``exclude_dirs`` entry; a directory with a carve-out lists its excluded files
(and fully-excluded subdirs) individually instead.

Config-model semantics (resolved against the real walk in
``docgen.staleness.find_catalog_files``):

- A **directory** is pruned by *name* at every depth — the walk drops a dir
  whose basename is in ``exclude_dirs``. So excluding a directory emits its
  **basename** into ``exclude_dirs`` (the only mechanism that prunes the whole
  subtree; ``Path.match`` globs don't recurse). Caveat: this also excludes any
  same-named directory elsewhere — fine for the noise/vendored dirs this
  targets, where names are distinctive.
- A **file** is matched per-path via ``Path.match``, so excluding a file emits
  its rel-path into ``exclude`` (a glob that matches that file).
"""
from __future__ import annotations

from attrs import frozen


@frozen
class Row:
    """One selectable (or auto-excluded) entry at the current level."""

    name: str
    rel_path: str
    is_dir: bool
    tokens: int          # content tokens (0 for an auto-excluded entry)
    docs: int            # generate docs (0 when the node has no cost)
    est_cost: float      # generate $ (token-value; 0 when no cost)
    excluded: bool       # user-toggled (self or an ancestor)
    auto_excluded: bool  # policy-pruned (shown dimmed, not selectable)


@frozen
class Totals:
    """Live kept-vs-excluded split, re-summed on every toggle."""

    kept_docs: int
    kept_cost: float
    excluded_docs: int
    excluded_cost: float


class ExplorerState:
    """Headless navigation + exclusion state over a scanned, costed tree."""

    def __init__(self, tree, costs, auto_excluded=()):
        self._tree = tree
        self._costs = costs
        self._auto = tuple(auto_excluded)
        # rel_path -> explicit override (True = excluded, False = kept). A path
        # with no entry inherits its nearest ancestor's state (default: kept).
        # This nearest-ancestor-wins model lets you exclude a dir and then carve
        # out a kept child, and makes keeping a dir clear stale child selections.
        self._overrides: dict[str, bool] = {}
        self._stack = [tree]  # nav stack; top is the current directory
        self._is_dir: dict[str, bool] = {}
        self._file_leaves: list[str] = []  # rel_paths of file nodes, for totals
        self._index(tree)

    def _index(self, node) -> None:
        self._is_dir[node.rel_path] = node.is_dir
        if not node.is_dir:
            self._file_leaves.append(node.rel_path)
        for child in node.children:
            self._index(child)

    # -- navigation ---------------------------------------------------------

    def current_rows(self) -> list[Row]:
        """Rows for the current directory: child dirs then files, each ranked
        by cost descending, with Tier 2 cost overlaid. At the root, the
        policy-pruned (auto-excluded) names are appended as dimmed,
        non-selectable rows for reassurance."""
        cwd = self._stack[-1]
        rows = [self._row(c) for c in cwd.children]
        rows.sort(key=lambda r: (not r.is_dir, -r.est_cost, r.name))
        if cwd is self._tree:
            rows.extend(
                Row(
                    name=n, rel_path=n, is_dir=True, tokens=0, docs=0,
                    est_cost=0.0, excluded=True, auto_excluded=True,
                )
                for n in self._auto
            )
        return rows

    def _row(self, node) -> Row:
        cost = self._costs.get(node.rel_path)
        return Row(
            name=node.name,
            rel_path=node.rel_path,
            is_dir=node.is_dir,
            tokens=node.content_tokens,
            docs=cost.docs if cost else 0,
            est_cost=cost.total if cost else 0.0,
            excluded=self._is_excluded(node.rel_path),
            auto_excluded=False,
        )

    def breadcrumb(self) -> list[str]:
        return [n.name for n in self._stack]

    def enter(self, name) -> None:
        """Drill into a child directory by name (no-op if there's no such
        directory child)."""
        for child in self._stack[-1].children:
            if child.is_dir and child.name == name:
                self._stack.append(child)
                return

    def up(self) -> None:
        """Pop to the parent directory (no-op at the root)."""
        if len(self._stack) > 1:
            self._stack.pop()

    # -- exclusion ----------------------------------------------------------

    def toggle(self, rel_path) -> None:
        """Flip the kept/excluded state of ``rel_path``.

        Excluding a directory excludes its whole subtree; the *minimal*
        override is recorded (nearest-ancestor-wins), so: keeping a directory
        clears any now-redundant child selections under it, and you can keep
        (carve out) a single item inside an excluded directory. Auto-excluded
        (policy) entries are not selectable — toggling one is a no-op.
        """
        if rel_path in self._auto:
            return
        new_state = not self._is_excluded(rel_path)
        # This node now dictates its subtree — drop descendant overrides so
        # toggling it back restores a clean subtree (no stale residue).
        prefix = rel_path + '/'
        self._overrides = {
            k: v for k, v in self._overrides.items() if not k.startswith(prefix)
        }
        # Record an override only when it differs from the inherited state;
        # otherwise inheriting from an ancestor is enough (keeps the set minimal).
        if new_state == self._inherited_excluded(rel_path):
            self._overrides.pop(rel_path, None)
        else:
            self._overrides[rel_path] = new_state

    def _is_excluded(self, rel) -> bool:
        """Effective state: the nearest ancestor-or-self override wins; a path
        with no override defaults to kept (not excluded)."""
        if rel in self._overrides:
            return self._overrides[rel]
        return self._inherited_excluded(rel)

    def _inherited_excluded(self, rel) -> bool:
        """Effective state from STRICT ancestors only (ignoring ``rel``'s own
        override) — what ``rel`` inherits if it has no override of its own."""
        parts = rel.split('/')
        for i in range(len(parts) - 1, 0, -1):
            ancestor = '/'.join(parts[:i])
            if ancestor in self._overrides:
                return self._overrides[ancestor]
        return False

    # -- accessors for alternate renderers (e.g. the optional Textual view) --

    @property
    def root(self):
        """The underlying ``ScanNode`` tree root."""
        return self._tree

    def cost_of(self, rel_path):
        """The ``NodeCost`` for ``rel_path``, or ``None`` (no generate cost)."""
        return self._costs.get(rel_path)

    def set_costs(self, costs) -> None:
        """Swap the cost map — e.g. after the doc-type selection changes, which
        re-prices every node. Navigation and exclusion overrides are keyed on
        rel_path (independent of cost), so they survive the swap."""
        self._costs = costs

    def is_excluded(self, rel_path) -> bool:
        """Effective state: True if ``rel_path``'s nearest ancestor-or-self
        override is 'excluded' (carve-outs respected)."""
        return self._is_excluded(rel_path)

    def excluded_rules(self) -> dict:
        """The ``exclude_dirs`` / ``exclude`` rules to persist.

        A file excluded on its own → an ``exclude`` glob (its rel-path). When
        *every* file under a directory is excluded the rules collapse to a
        single ``exclude_dirs`` entry (the dir basename) rather than
        enumerating files — the only mechanism that recursively prunes the
        subtree."""
        _full, dir_rels, file_rels = self._collect(self._tree)
        return {
            'exclude_dirs': sorted({rel.rsplit('/', 1)[-1] for rel in dir_rels}),
            'exclude': sorted(file_rels),
        }

    def _collect(self, node) -> tuple[bool, set, set]:
        """Bottom-up rule collection. Returns (fully_excluded, dir_rels,
        file_rels) for ``node``'s subtree; a directory whose entire subtree is
        excluded collapses to its own rel-path and discards descendant rules."""
        if not node.is_dir:
            ex = self._is_excluded(node.rel_path)
            return (ex, set(), {node.rel_path} if ex else set())
        results = [self._collect(c) for c in node.children]
        full = bool(node.children) and all(r[0] for r in results)
        if full and node is not self._tree:
            return (True, {node.rel_path}, set())
        dir_rels: set = set()
        file_rels: set = set()
        for _f, dirs, files in results:
            dir_rels |= dirs
            file_rels |= files
        return (False, dir_rels, file_rels)

    # -- live total ---------------------------------------------------------

    def live_total(self) -> Totals:
        """Instant kept/excluded split, re-summed on every toggle. Sums the
        cost of every *file* whose effective state is excluded, so carve-outs
        (a kept file inside an excluded dir) are reflected exactly."""
        root = self._costs.get(self._tree.rel_path)
        root_total = root.total if root else 0.0
        root_docs = root.docs if root else 0
        exc_cost = 0.0
        exc_docs = 0
        for rel in self._file_leaves:
            if self._is_excluded(rel):
                cost = self._costs.get(rel)
                if cost is not None:
                    exc_cost += cost.total
                    exc_docs += cost.docs
        return Totals(
            kept_docs=root_docs - exc_docs,
            kept_cost=root_total - exc_cost,
            excluded_docs=exc_docs,
            excluded_cost=exc_cost,
        )


def apply_excludes(state: ExplorerState, cfg, source_name: str) -> bool:
    """Persist ``state``'s exclude rules to ``cfg`` for ``source_name``.

    Unions with the source's existing ``exclude_dirs`` / ``exclude``: the
    config writer (:meth:`Config.set_source_config`) *replaces* each field, so
    the full merged set is computed here rather than appended. ``cfg`` is a
    ``Config`` (or anything exposing ``get_source_config`` /
    ``set_source_config``). A no-op (returns ``True``) when nothing is
    excluded, so confirming an untouched explorer never rewrites the config.
    """
    rules = state.excluded_rules()
    if not rules['exclude_dirs'] and not rules['exclude']:
        return True
    sc = cfg.get_source_config(source_name)
    existing_dirs = list(sc.exclude_dirs) if sc else []
    existing_globs = list(sc.exclude) if sc else []
    return cfg.set_source_config(
        source_name,
        exclude_dirs=sorted(set(existing_dirs) | set(rules['exclude_dirs'])),
        exclude=sorted(set(existing_globs) | set(rules['exclude'])),
    )
