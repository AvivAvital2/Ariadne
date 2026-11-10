# Connect model servers you manage

Ariadne can use supported model servers on this machine or a remote host. You
install the models, run the servers, configure authentication, and provide enough
memory and compute. Ariadne checks the serving contract and uses the saved pair
for generation, embedding, Search and Ask. This path does not require an M5 or
Ariadne's managed Mac launcher. It makes no promise about a particular Mac's
memory use or speed.

## Connect through the UI

1. Open onboarding and continue to **Preview** or **Generate**.
2. Open **Connect servers you manage** beside the model controls.
3. Choose **Qwen 27B + Qwen 8B embeddings** or **Gemma 4 12B + Qwen 8B embeddings**,
   then enter the generation and embedding server URLs, each ending in `/v1`. URL
   prefixes are supported, for example `https://models.example/chat/v1`.
4. If authentication is required, enter the names of credential environment
   variables already set in the Ariadne server process. Do not paste API keys.
5. Optionally set **Request timeout in minutes** to a whole number from 10 to 240.
   CPU-only servers may need longer. The value applies to requests to both servers
   and the overall connection check. Leave it blank to keep the existing defaults,
   including the two-minute overall connection check.
6. Click **Test and save**. This explicitly makes a small completion and document/
   query embedding requests and may incur charges on your server. Ordinary Preview
   does not send inference requests.
7. After success, the saved profile is selected for estimates and the next build.
   Generate, reopen, Search, Ask and resumable build controls remain in the UI.

The form connects either **Qwen3.8-27B Q4 thinking** or **Gemma 4 12B Q4
non-thinking** generation with **Qwen3-Embedding-8B Q4**. Omitting the model choice
in an API request retains the original Qwen profile. Existing YAML-configured
BF16 and legacy pairs remain supported through their existing profiles.

The connection check verifies advertised model aliases, template/tokenization
APIs, server context capacity, a complete final chat response, and actual document
and query vectors. It rejects truncated/reasoning-only chat output and malformed
embeddings. Raw 4096-component embeddings are reduced and normalized to
3072-component float32 vectors using the existing recipe. The check uses temporary
databases and does not write to a project library.

**A passing connection check is compatibility evidence. It does not verify answer
accuracy, attest the loaded weight files, or qualify hardware performance.** The
configured aliases must correspond to the supported artifacts; the operator must
verify their hashes and runtime configuration.

## Required server contract

| Setting | Generation | Embeddings |
| --- | --- | --- |
| Model alias advertised by `/v1/models` | `Qwen/Qwen3.8-27B` | `Qwen/Qwen3-Embedding-8B` |
| GGUF artifact | `Qwen3.8-27B-Q4_K_M.gguf` | `Qwen3-Embedding-8B-Q4_K_M.gguf` |
| Minimum advertised context for this form's fixed profile | 32768 tokens | 8192 tokens |
| Completion/vector contract | Thinking enabled, `xhigh`; separate final content; up to 8192 total completion tokens for document jobs | Return all 4096 components, without server-side dimensional reduction |

For the Gemma choice, the generation alias is `google/gemma-4-12b-it`, the artifact
is `gemma-4-12b-it-Q4_K_M.gguf`, and the minimum generation context is **16384**.
Run generation with `--jinja --chat-template-kwargs '{"enable_thinking":false}'
--reasoning-format auto`. The profile keeps reasoning disabled and accepts only
complete final content, with up to 8192 output tokens. The embedding server,
8192-token embedding context, input recipe, dimensions, and normalization are
identical to the Qwen Q4 pair. `model_profile: gemma4-12b-qwen8b-mac-q4-v1` selects
this pair in the `/api/external-models` request.

Both servers must implement `/v1/models`, `/tokenize` and `/props`. The generation
server must also implement `/apply-template` and `/v1/chat/completions`; the
embedding server must implement `/v1/embeddings`. An OpenAI-compatible completion
route alone is insufficient. Native template and tokenization calls must reflect
the input actually consumed by inference. TLS verification stays enabled,
redirects are refused, and ambient proxy variables are not used.

Exact artifact revisions, SHA-256 hashes and selected generation settings are in
[local_models.py](../local_models.py) and the
[qualification record](mac-local-models-qualification.md#selected-27b-thinking-profile).
The existing runtime candidate is llama.cpp `b11669`. The UI does not download,
start, stop or attest user-managed servers. The optional managed Mac controls
continue to have their own platform and qualification requirements.

For an Intel CPU-only run, start both `llama-server` processes with
`--n-gpu-layers 0 --threads 8 --timeout 14400` and choose **240** minutes in the
form. The server timeout must allow the same wait as the client; increasing only
the form value does not extend a shorter server deadline. The client setting
configures HTTP operation timeouts and the overall connection-check deadline;
it is not a deadline for an entire multi-request generation workflow. It does
not change separate remote MCP transport or optional tool-specific deadlines.
These settings permit a longer check and do not establish that the models fit
or run fast enough on the machine.

## Saved profiles and existing knowledge

Successfully checked profiles are stored in `web-projects/model-profiles.json`
beside the root configuration's project directory. The registry saves the fixed
model settings, endpoint URLs and credential variable names; credential values
are not copied into it.
The root source configuration and `depends_on` declarations are untouched.

The profile identifier includes the endpoint configuration and supported model
settings. Repeating an identical connection is idempotent. Changing an endpoint,
credential reference or explicit request timeout creates a distinct profile and
a separate project when used for a build. Existing projects retain their own endpoints and bound model pair;
selecting a new profile does not migrate or rebind their libraries. Model artifact
and embedding-space identities keep their existing definitions.

A failed check saves no profile. A failed registry replacement preserves its prior
contents. Credentials missing from the Ariadne process fail before inference;
changing a shell environment after Ariadne starts does not update that process.
If the server advertises a smaller context, adjust its configuration to the stated
contract or use an appropriately configured existing YAML profile.

## Remote validation and acceptance

Remote native validation should use the actual local adapter against the supported
serving APIs and operator-verified artifacts. The server's physical location does
not change the adapter. Validate the full generation, embedding, persistence,
reopen, Search/Ask and interruption/resume paths, including dependency sources and
an unrelated-source control. Judge actual delivered evidence and citations, not
just configured scope or a successful HTTP status.

Quality checks use frozen source facts, all completed outputs, repeated cases,
and reserved confirmation cases. Record unsupported claims and omitted required
facts separately. Serving compatibility does not excuse inaccurate generated
documentation. The current candidate has recorded factual failures and remains
unqualified for production-quality output.

The [2026-10-25 Intel CPU smoke](../evaluation/intel-local-2026-10-25/README.md)
completed native generation, embeddings, persistence, reopen, Search and Ask
under a project saved through the same API as this UI. It used one known
five-line fixture and an authored dependency control. The generated FAQ covers
all five required facts, but its extra guidance still contains precision and
example defects. Ask correctly answers the one equality/removal question with
help from the authored dependency; the original answer does not satisfy its
conflicting location-citation instruction.

Separate October 26 corrections fix Search omission counts for repeated headings
and align documentation-Ask prompts with document-title citations. Native Search
passed in 3.08 seconds with its scope and selected content unchanged. The final
Python suite passed 5151 tests, with 5 skipped and 1 existing expected failure;
the earlier timeout browser selection passed 9 cases using intercepted replies.
The report retains these results apart from the original run. The post-correction
native Ask completed in **16.8 minutes** and correctly answered the same question
with supported document-title citations and authored dependency assistance.
The linked report also records HTTP/static and real project-MCP metadata readiness;
those checks do not constitute a native browser Generate/onboarding run.

On an Intel Core i9 with 64 GiB RAM, the workflow took **104.97 minutes**:
**84.6 minutes** for QA generation and **17.4 minutes** for Ask. The highest
sampled combined server RSS was **36.4 GiB**. These are observations from one
CPU run, not full native browser onboarding, a broad accuracy result, a
throughput benchmark or evidence of M5 capacity. Process RSS can count shared
pages twice and does not measure total system memory.

Hosted DeepInfra checks can provide hosted model-quality and application evidence.
Their request mapping does not exercise the native template/tokenization contract
or establish equivalence to pinned Q4 weights. Keep those results separate from
native validation. M5 benchmarking is outside this user-managed-server acceptance
boundary; no Mac-specific capacity or latency claim follows from remote testing.
