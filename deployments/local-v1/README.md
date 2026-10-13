# Local serving candidate v1

This configuration has **not been qualified on GPUs**. The [manifest](manifest.yaml) pins real upstream artifacts and a vLLM image; it is a candidate for the qualification procedure in [the report](../../docs/local-model-qualification.md). No quality, capacity, or latency claim follows from these metadata checks.

The candidate uses BF16 weights and two NVIDIA devices. Select suitable devices and record their model, memory, driver and CUDA versions before production binding. Adjust device IDs in the Compose file to your host. A different weight artifact, quantization or embedding recipe needs a separate library and sidecar.

Provision artifacts and the image while network access is available, from this directory:

```sh
hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --local-dir artifacts/chat
hf download Qwen/Qwen3-Embedding-8B --revision 1d8ad4ca9b3dd8059ad90a75d4983776a23d44af --local-dir artifacts/embeddings
docker pull vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
```

Retain the downloaded snapshot metadata and file checksums. `artifacts/` is excluded from git. Offline startup reads these directories, disables optional telemetry and puts the servers on an internal Docker network:

```sh
docker compose up -d
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8001/v1/models
```

Ports are loopback-only. On a remote host, use an authenticated TLS reverse proxy and configure its explicit URLs and credential environment-variable names in Ariadne. Keep `/tokenize` on each endpoint's root alongside `/v1`. Ariadne accepts URL prefixes, ignores proxy environment variables and refuses redirects. A `/models` alias cannot attest which weights are loaded; verify the operator's snapshot and image manifest independently.

The optional `client` Compose profile runs the [Ariadne application image](../../docs/docker.md) on the same internal network. Its [configuration](../app/local.yaml) already contains these artifact pins and server names. After building the application image, run `docker compose --profile client run --rm ariadne doctor --models`, then `docker compose --profile client run --rm ariadne generate --source project --batch --yes`. Set `ARIADNE_SOURCE_DIR` to an absolute source path; the default is this checkout. State persists in the separate `ariadne-state` volume.

For a host-installed application, copy `ariadne.local.yaml.example` to its `ariadne.yaml` and replace both model and tokenizer revision placeholders with the corresponding manifest revisions. Keep `weight_format: bf16`, recipe `qwen3-embedding-8b-3072-v1`, and the context/input limits aligned with these servers. Non-thinking options are sent on every chat and chat-tokenization request. No tool, streaming, multimodal or reasoning-response interface is enabled by Ariadne.

The embedding service returns all 4096 components with last-token pooling. Ariadne retains the first 3072, normalizes in float64, and stores float32. Do not configure server-side dimensional reduction. The pinned vLLM pooler uses `seq_pooling_type` and `use_activation`; the former `normalize` parameter has been removed in this version. See the [pinned pooler source](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/pooler.py) and [tokenization protocol](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/entrypoints/serve/tokenize/protocol.py).

Run `ariadne doctor --models` to inspect binding and probe both model/tokenization endpoints without generating content. Run `ariadne onboard --source myproject` for a preview with no inference or corpus writes. `ariadne onboard --source myproject --approve` initializes the fresh pair and executes its durable run. Preserve the printed `local:` ID. Use `ariadne batch status ID` or `ariadne batch resume ID` after interruption.

Before enabling production use, complete the pending quality, retrieval, resource, offline-egress and interruption measurements in the qualification report. The moving [upstream Qwen recipe](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-27B.yaml) is background information; deployment uses the immutable pins above.
