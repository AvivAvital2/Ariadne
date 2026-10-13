# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the project is at `0.x`, breaking changes may occur in any release.

## [Unreleased]

### Added

- Added local inference for `Qwen/Qwen3.8-27B` and `Qwen/Qwen3-Embedding-8B`, with independent endpoints, exact token preflight, validated 4096 → normalized 3072 embeddings and no cloud fallback.
- Added shared inference configuration, immutable model-pair binding, operation-scoped writer locking, atomic document/chunk/section replacement, export/import provenance and revision-aware retrieval caches. Model changes require a fresh library and staleness database.
- Added resumable local jobs, bounded persisted retries, fixed workflow phases, offline job inspection and previews without inference or corpus writes.
- Added cloud/local Compose presets and a pinned vLLM serving candidate. The public Docker image retains its browser/MCP startup and SCIP toolchains. External image publication and actual GPU qualification remain pending.
- Added hosted verification for GPT-6 Astra, Claude Opus 5.5 and Qwen, recorded-response regressions, crash-recovery tests and a frozen 50-question retrieval benchmark. See [supported models](docs/supported-models.md) and the [qualification report](docs/local-model-qualification.md) for evidence and limits.
- Fixed usage-statistics tuple access and source filtering that discarded relevance order. Updated the shared OpenAI token-limit helper for GPT-6.
- Fixed MCP section fallback returning a Markdown title without its body when no section meets the similarity threshold. The fallback now skips heading-only sections, and older search-cache entries are bypassed.

- The Slack bridge now attaches answers as Markdown files when users explicitly
  request file output, with lossless bounded thread chunks as an upload fallback.

## [0.1.0] - 2026-05-26

- Initial public release under the Apache License 2.0.
