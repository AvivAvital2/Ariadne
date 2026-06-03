# Contributing to Ariadne

Thanks for considering a contribution. This guide covers what you need to
know before opening a pull request.

## License

Ariadne is licensed under the [Apache License 2.0](LICENSE). Unless you
state otherwise, any contribution you intentionally submit for inclusion
in the project is provided under the same Apache License 2.0, per Section 5
of the license — no separate contributor agreement or signing step is
required.

## Reporting bugs and requesting features

Use GitHub Issues for both. Before opening one, please search existing
issues to avoid duplicates. For bug reports, include:

- What you ran (command, version, OS)
- What you expected
- What actually happened (include stack traces verbatim)
- A minimal reproduction if possible

For security issues, **do not open a public issue** — see
[SECURITY.md](SECURITY.md).

## Development setup

Prerequisites:

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- An `OPENAI_API_KEY` (required for embeddings) and optionally
  `ANTHROPIC_API_KEY` (for generation) in a local `.env` file —
  `.env` is gitignored.

```bash
git clone https://github.com/AvivAvital2/ariadne.git
cd ariadne
uv sync
```

To run the CLI against a local source, copy the example config:

```bash
cp ariadne.yaml.example ariadne.yaml
# Edit ariadne.yaml to point at the source you want to index
```

`ariadne.yaml` is gitignored — your local paths stay local.

## Running tests

```bash
uv run pytest                              # full suite
uv run pytest tests/test_<file>.py         # one file
uv run pytest -x --tb=short                # stop on first failure
uv run pytest --cov --cov-report=term-missing   # with coverage
```

A pre-commit hook runs `ty` (type checker) on changed Python files.
Install it once:

```bash
uvx pre-commit install
```

Or run it manually before pushing:

```bash
uvx ty check
```

## Code style

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and
formatting. Configuration lives in `pyproject.toml` under `[tool.ruff]`.

```bash
uvx ruff check .            # lint
uvx ruff format .           # format
```

Line length is 100. Target version is Python 3.12. CI (when added) will
run both checks.

## Submitting a pull request

1. Fork the repo and create a topic branch (`feat/...`, `fix/...`,
   `docs/...`).
2. Keep PRs focused — one logical change per PR is much easier to review
   than a sweep.
3. Add or update tests when you change behavior. Tests live in `tests/`.
4. Update `CHANGELOG.md` under the `[Unreleased]` section if your change
   is user-visible (new feature, bug fix, breaking change).
5. Update relevant documentation (README, CLAUDE.md, docs under
   `docs/`) when behavior or commands change.
6. Open the PR against `main`. Describe what changed and why; link any
   related issue.

## What the maintainer is looking for

- **Bug fixes** with a test that demonstrates the bug
- **Documentation improvements** — corrections, clarifications, missing
  command references
- **New features** — please open an issue to discuss before doing
  significant work, so the design can be agreed on first
- **Test coverage** for under-tested code paths

Large architectural changes are best discussed in an issue or
[Discussion](https://github.com/AvivAvital2/ariadne/discussions) before
implementation.

## Questions

Open a GitHub Discussion or email the maintainer at ariadne.switch027@passinbox.com.
