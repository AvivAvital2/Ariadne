# Local models

Ariadne implements local inference for the fixed pair `Qwen/Qwen3.8-27B` and `Qwen/Qwen3-Embedding-8B`. The software is tested against two independent HTTP fixtures. The pinned [serving candidate](../deployments/local-v1/README.md) still needs [real-model qualification](local-model-qualification.md); it is not a production support claim.

For the cloud OpenAI/Claude pairs, credentials and switching configurations with `ARIADNE_CONFIG`, see [supported model configurations](supported-models.md). CLI and MCP must select the same configuration; restart MCP when changing it.

Development and automated tests do not require GPUs, model weights, or running Qwen servers. Real-model qualification is deferred until a deployment is available. Actual local generation, embedding and semantic retrieval require the configured model endpoints.

For paid verification without local servers, use the [shared hosted workflow runner](live-model-verification.md). Change `--profile` to run the same scenarios with hosted Qwen, Claude or OpenAI; each profile uses a fresh library and an explicit spending cap. The separate blind implementation run passed 41 checks across six fresh Qwen cases; earlier factual failures remain recorded separately.

The [Docker application image](docker.md) includes cloud/local presets and has passed local registry push/pull, installed-package CLI/MCP checks with networking disabled, and state persistence across container recreation. External publication and real GPU qualification remain separate.

The browser uses the same configured pair for cost previews, onboarding, search and answers. Add the [web Compose override](docker.md#browser-with-a-configured-model-pair) to either preset. The [Qwen browser journey](../web/e2e/README.md) exercises the combined image with deterministic HTTP fixtures, including clew retrieval and application restart.

Start from [ariadne.local.yaml.example](../ariadne.local.yaml.example). Set independent `/v1` endpoints, actual artifact and tokenizer revisions, BF16 weight representation, and fresh library/sidecar paths. Credential settings name environment variables; `null` omits authorization. Local requests ignore `OPENAI_BASE_URL`, cloud keys and proxy environment variables. HTTP redirects are refused. Token counting uses each local server's `/tokenize`, including the rendered non-thinking chat template. Generation reserves its complete requested output budget plus 64 tokens. Inputs that do not fit fail explicitly.

```sh
ariadne config
ariadne doctor --models
ariadne onboard --source myproject
ariadne onboard --source myproject --approve
ariadne generate --source myproject --batch
ariadne search 'How is authentication configured?' --source myproject
```

Local previews do deterministic discovery and character estimates without inference, corpus writes, binding, catalog embeddings, or spool creation. Estimates say **$0 provider token charges; hardware/runtime cost unestimated**. Missing structural artifacts make the estimate incomplete. Hosted preview behavior is unchanged.

`config`, inspection/export and batch list/status work with inference servers offline. `doctor --models` explicitly probes both advertised model IDs and tokenization routes; it does not generate text or bind a missing library.

A library reserves both model identities before its first corpus mutation. Its staleness database belongs to the same UUID/profile. Changing endpoints, credential values, timeouts or concurrency limits does not change the model identity. Changing artifact/tokenizer pins, weight format or embedding recipe requires a new pair. `--force`, import and rebuild cannot migrate models. With local `--db`, an unspecified sidecar is derived beside that database; use `--staleness-db` before the subcommand to choose it explicitly.

Populated unbound libraries, including structural-only libraries, cannot be adopted into local mode. Existing populated hosted libraries keep their legacy behavior without a historical identity guarantee. Newly created hosted libraries also reserve their configured pair. Bare handles may inspect or export bound databases; semantic consumers and writers need an `InferenceContext`.

The embedding recipe is `qwen3-embedding-8b-3072-v1`. Only search queries receive the fixed retrieval instruction. Documents retain the title plus the first 2000 content characters; chunks retain current chunk text and sections retain their heading or `Introduction`. The server returns 4096 floats, response indices restore ordering, and Ariadne stores the normalized first 3072 components. Raw local embedding wrappers require a query/document purpose. Invalid vectors never enter a prepared document transaction.

Live operations prepare document, chunk and section vectors before atomically replacing their text and vectors. Metadata-only changes preserve vectors. Same-model rebuild refreshes all three levels and removes obsolete children. Corpus revisions invalidate retrieval caches across processes. Audience responses use the same model pair and scope/branch/revision-aware caches. Generated cross-reference decoration excludes authored text and catalogs.

## Clews

After structural indexing, build and embed clew routes with the same configuration as CLI and MCP:

```sh
export ARIADNE_CONFIG=/absolute/path/to/ariadne.local.yaml
uv run python evaluation/answer-path/write_clews.py --source myproject --yes
# Optional: add retrieval questions for theme-associated routes.
uv run python evaluation/answer-path/question_clews.py --source myproject --yes
uv run python evaluation/answer-path/embed_clews.py --source myproject --yes
```

All three tools honor configured library paths and accept `--db` and `--staleness-db` overrides. `--dry-run` inspects without inference or corpus writes. Writes use the library's model binding, writer lock and revision tracking. Embedding commits complete batches and skips completed rows on rerun. Changing a clew's question or steps invalidates its vector. Clew text uses the document embedding recipe; retrieval questions receive the query instruction. Existing OpenAI clew vectors cannot be reused in a Qwen library: build the routes in the fresh, Qwen-bound library and embed them there.

The [hosted clew scenario](live-model-verification.md#clew-workflow) checks retrieval, route selection, source evidence and synthesis with the configured model pair. Local HTTP tests cover both deterministic and model-driven selection without a GPU.

## Durable local runs

`--batch` runs a single foreground worker with SQLite persistence; it has no cloud batch discount or 24-hour SLA and never calls `/batches` or uploads batch files. Auto mode uses `execution.auto_batch_threshold`. Local onboarding and theme builds have durable phases; catalog synchronization uses embedding-only jobs. Catalog descriptions support `--batch`.

```sh
ariadne batch list
ariadne batch status local:RUN_UUID
ariadne batch resume local:RUN_UUID
ariadne batch cancel local:RUN_UUID
ariadne batch clear local:RUN_UUID
```

List/status work offline. Ctrl-C and process termination leave work resumable. Saved chat responses and parsed results survive an interrupted embedding/application stage. Retry attempts are persisted before work and remain bounded after restarts. An applied document and its job marker commit together, so replay preserves timestamps. Status distinguishes saved responses from applied documents. Explicit cancellation is terminal; clearing accepts terminal runs only and preserves applied documents.

One nonblocking POSIX lock owns corpus writes across CLI and MCP operations. Independent writers fail as busy. Cancellation uses a short control-only transaction, allowing another process to stop an active run. SQLite remains WAL/`synchronous=normal`: process-crash recovery is tested; power-loss durability has not been qualified. Keep both database files, their SQLite WAL files as appropriate for a consistent backup, and the main database's spool when backing up a library. Backups preserve UUIDs.

Later phases are planned after their inputs exist: structural catalog → catalog descriptions → per-file generation → reverse augmentation → semantic graph/themes → generated-prose cross-references → staleness reconciliation. Source and target changes make queued work obsolete. A file becomes current only when all requested, language-supported document types committed for the same captured hash. Sidecar keys include source name, generation scope and relative path.

Bound Markdown exports carry versioned model-pair and generated-document provenance using safe nested YAML. Import preflights all candidates before applying any and re-embeds matching content locally; another library UUID is allowed for the same profile. Import never copies sidecar ownership or spools. Unidentified generated exports are rejected. Use authored ingestion (`add`, `finding`, MCP contributions) for human-authored material.
