# Supported model configurations

Ariadne supports native OpenAI and Anthropic generation with OpenAI embeddings, and a fixed local Qwen generation/embedding pair. CLI generation, MCP synthesis and retrieval use the same configured pair. The MCP client, such as Claude Code, can use its own model independently of Ariadne's backend.

## Model pairs and verification

| Configuration | Documentation and synthesis | Embeddings | Verification recorded on 2026-10-11 |
| --- | --- | --- | --- |
| OpenAI cloud | `gpt-6-astra` | `text-embedding-3-large`, 3072 dimensions | 24/24 live workflow checks passed |
| Anthropic cloud | `claude-opus-5-5` | `text-embedding-3-large`, 3072 dimensions | 24/24 live workflow checks passed, including a fresh rerun after the section-fallback fix |
| Local Qwen | `Qwen/Qwen3.8-27B` | `Qwen/Qwen3-Embedding-8B`, native 4096 reduced and normalized to 3072 | Application HTTP, CLI/MCP, recovery and container tests passed; pinned GPU/vLLM deployment remains unqualified |

The named cloud models are the pairs exercised in the live reports, not an exhaustive vendor model catalog. Existing cloud configuration remains compatible; the built-in CLI generation default is still `gpt-5.5`. Set `llm.model` explicitly to select the verified models above.

The separate hosted Qwen test profile uses DeepInfra and `DEEPINFRA_API_KEY`. Its six-case implementation run passed 41 checks; an earlier model factual error remains documented. The runner supplies a hosted embedding adapter and does not establish a supported local deployment or a general DeepInfra configuration preset.

## Configure the cloud suite

Save this as a configuration file, for example `ariadne.openai.yaml`, and set your source path:

```yaml
default_source: myproject
sources:
  myproject: /absolute/path/to/myproject
defaults:
  db_path: ./state/openai.db
  staleness_db_path: ./state/openai_staleness.db
llm:
  provider: openai
  model: gpt-6-astra
  api_key_env: OPENAI_API_KEY
embeddings:
  provider: openai
  model: text-embedding-3-large
  dimensions: 3072
  api_key_env: OPENAI_API_KEY
```

For Claude, copy the configuration to `ariadne.claude.yaml`, choose separate database paths, and replace its `llm` block with:

```yaml
llm:
  provider: anthropic
  model: claude-opus-5-5
  api_key_env: ANTHROPIC_API_KEY
```

OpenAI generation needs `OPENAI_API_KEY`. Claude generation needs both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`, since its embeddings still use OpenAI. Set credentials in the environment or an ignored `.env`; YAML stores their variable names only. Legacy cloud `defaults.model` / `defaults.provider` and top-level aliases remain accepted, but explicitly supplied aliases must agree with canonical settings.

## Configure the local suite

Start with [ariadne.local.yaml.example](../ariadne.local.yaml.example) and the [local setup guide](local-models.md). Set `provider: local` for both `llm` and `embeddings`, independent `/v1` URLs, immutable model/tokenizer revisions and fresh database paths. Both endpoints must expose the matching `/tokenize` route. The [pinned serving candidate](../deployments/local-v1/README.md) and Docker local preset contain concrete artifact pins.

Local mode does not require cloud credentials. Endpoint authentication is optional through `api_key_env`; `null` sends no authorization header. The initial local implementation requires the exact Qwen pair, BF16 artifacts and `qwen3-embedding-8b-3072-v1` recipe. Mixed local/cloud pairs, other local models, other serving engines and alternate quantizations are outside this release's contract.

## Switch configurations

Select the configuration with `ARIADNE_CONFIG`; no application-code change is needed:

```sh
ARIADNE_CONFIG=/absolute/path/to/ariadne.openai.yaml uv run ariadne config
ARIADNE_CONFIG=/absolute/path/to/ariadne.claude.yaml uv run ariadne generate --source myproject
ARIADNE_CONFIG=/absolute/path/to/ariadne.local.yaml uv run ariadne doctor --models
```

Use the same environment setting when starting `ariadne mcp`, and restart the MCP server after switching. Paths under `defaults` resolve relative to the configuration file. [Docker usage](docker.md) describes the bundled configurations and Compose commands.

Clew route generation and embedding use the same configured model pair and storage controls; see the [clew workflow](local-models.md#clews).

Each new library binds to its generation model, embedding model and recipe. A different pair needs a fresh main database and staleness database generated from the original sources. Keep previous configurations to reopen their existing libraries. Import, `--force` and rebuild do not migrate between pairs. Existing populated unbound cloud libraries retain their legacy lifecycle; they cannot be adopted into local mode.

Cloud batching selects the configured OpenAI or Anthropic batch API. Local batching runs a durable foreground worker, resumed through `ariadne batch resume local:RUN_UUID`; it has no cloud batch discount. Local previews make no inference calls or corpus writes. Local provider token charges are zero; hardware and runtime costs remain unestimated.

The browser wizard selects the configured model for its initial preview. In local mode it offers that Qwen pair, uses concurrency 1 and the durable onboarding pipeline, and shows zero provider charges with hardware/runtime cost unestimated. The [web Compose override](docker.md#browser-with-a-configured-model-pair) enables the browser for either deployment preset.

## What the evidence establishes

The hosted tests exercise generation, validation, storage, document/chunk/section retrieval, MCP answering, replacement, replay and rollback. The separate [50-question benchmark](retrieval-verification.md) measures embeddings and passage delivery. Claude and GPT share the OpenAI embedding backend, so its retrieval score is not a score for either model's answer quality.

The application image passed clean registry pull, installed CLI/MCP, persistence and both Compose configurations on linux/amd64 using model HTTP fixtures. It has not been published to an external registry. Actual local weights, vLLM tokenization, GPU performance and ARM64 execution remain unverified; see the [qualification report](local-model-qualification.md).
