# Ariadne v0.5

## New Features

### Local and hosted Qwen inference

- Added local inference with `Qwen/Qwen3.8-27B` for documentation and synthesis, and `Qwen/Qwen3-Embedding-8B` for semantic retrieval. Chat and embeddings use independently configured endpoints without requiring cloud credentials.
- Added hosted Qwen through the production DeepInfra provider. The same Qwen pair uses `DEEPINFRA_API_KEY` for both generation and embeddings, without an OpenAI key or local GPU. Hosted Qwen supports live execution; cloud batch discounts do not apply.
- Unified model configuration across CLI generation, MCP answers and retrieval. Select an OpenAI, Claude or Qwen configuration through `ARIADNE_CONFIG`, keeping each model pair in its own library.
- Added server-based token counting and context-budget checks for local generation. Qwen embeddings use separate query and document recipes, with validated native 4096-dimensional vectors reduced and normalized to 3072 dimensions. Local and hosted Qwen retain separate embedding identities.
- Added `ariadne doctor --models` to check configured local model identities and tokenization endpoints. Configuration inspection and local job status remain available while model servers are offline.

### Model selection and cost comparison during onboarding

- Moved model selection into the browser's **Preview** step. Compare OpenAI, Claude, hosted Qwen and configured local Qwen against the current project and scope before choosing **Start build**.
- Added generation and embedding cost estimates for the selected pair. Previews make no inference calls, and cloud estimates work without credentials. Local inference shows zero provider token charges, with hardware and runtime costs unestimated.
- Saved the selected model pair only when a build starts, with a separate project configuration, library and staleness database. Search, answers and clews then use that pair; the console's project selector reopens saved projects after a restart.

### Qwen support throughout the clew workflow

- Completed configured-model support for the existing clew tools: route construction, optional retrieval-question generation and embedding now use the same configuration as CLI and MCP.
- Applied library model binding, writer locking and revision tracking to clew writes. Embedding commits complete batches and skips completed routes when resumed.
- Extended verification through semantic retrieval, route selection, compiler traversal and answer synthesis, checking that selected routes contribute source evidence to the model's answer context.

### Resumable local workflows and consistent storage

- Added durable local jobs for generation, onboarding, catalog descriptions and themes, with persisted progress, bounded retries, cancellation and explicit resume commands.
- Preserved completed model responses across interrupted embedding or storage stages, allowing recovery without repeating completed generation.
- Bound new libraries and their staleness databases to their configured model pair. Incompatible models and embedding recipes are rejected before they can mix stored vectors.
- Made document, chunk and section replacement atomic, with shared writer ownership and revision-based cache invalidation.
- Added local previews that perform discovery and estimation without inference calls or corpus writes.

### Docker and browser deployment

- Added cloud and local Docker Compose presets, bundled configuration examples for OpenAI, local Qwen and hosted Qwen, and a pinned vLLM serving configuration for the local Qwen pair.
- Added direct CLI command execution through the Docker entrypoint, alongside browser and MCP startup. One application container offers the configured model pairs; a source-directory volume exposes host projects to onboarding, while project configuration and libraries persist under `/data`.

## Bugfixes

- Fixed MCP section fallback returning only a Markdown title when no section met the similarity threshold. Retrieval now selects a section containing body text and bypasses obsolete cached responses.
- Fixed stale document embeddings after title changes. Clew question and embedded route-text changes now invalidate their vectors, and retrieval caches refresh after corpus updates.
- Preserved relevance ordering during source filtering and applied source restrictions to chunk retrieval. The Ask console now sends its selected source with both requests.
- Corrected usage-statistics row handling and GPT-6 completion-token parameter selection.
- Prevented SQLite slice transfers from bypassing model-pair checks on bound libraries.
- Fixed Docker onboarding failures caused by an omitted URL-literal extractor and a missing Go manifest language mapping.
- Kept the selected provider, endpoints, credentials and writer ownership throughout nested onboarding phases. Cost estimation no longer creates an empty project library, and Preview waits for pending scope edits before estimating or building.
- Widened the model menu and wrapped long model names to avoid unnecessary horizontal scrolling.

## Other Changes

- Added shared hosted verification scenarios for Qwen, GPT-6 Astra and Claude Opus 5.5, alongside reusable offline fixtures, recovery tests and clew workflow coverage.
- Added a frozen 50-question benchmark comparing Qwen and OpenAI embedding retrieval, plus Docker and browser coverage for cost previews, onboarding, retrieval, clews and restart persistence.
- Updated model configuration, Docker, onboarding and import/export documentation. Markdown exports now carry model provenance that imports validate.
- Switching model pairs requires fresh library and staleness databases; existing vectors are not migrated between incompatible models. Local endpoints must be configured before local Qwen can be selected in Preview.
- Actual local GPU/vLLM deployment qualification remains pending; hosted-model and fixture-based application tests do not establish local GPU capacity or throughput.
