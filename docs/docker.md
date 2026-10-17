# Docker application image

The application image installs Ariadne and its locked runtime dependencies, including the CLI, browser UI, and stdio/HTTP MCP servers. With no command arguments it starts the browser on port 8765 and MCP on port 8000; explicit arguments run the CLI. The root `docker compose up --build` starts both services. It runs as UID 1000, stores state under `/data`, and reads source under `/workspace`. It includes cloud and local configurations with container paths; the local configuration already contains the serving manifest's artifact pins.

The [supported-models guide](supported-models.md) covers the OpenAI, Claude and local Qwen pairs, their credentials and verification status. The application image serves any of these configurations; changing the model pair requires separate database paths.

On 2026-10-11 the earlier CLI image was built and tested on **linux/amd64**. A subsequent local distribution test pushed it to an isolated registry, pulled it by digest into an empty Docker daemon, and verified the installed package, both Compose configurations and state persistence across container recreation. The CLI/MCP model fixtures run with external networking disabled. External publication is separate from this verification; build the image from this checkout:

```sh
docker build -t ariadne:local-model-support .
docker run --rm --network none ariadne:local-model-support --help
```

The public image uses Python 3.12 and the locked runtime dependencies. It includes the SCIP indexers and build toolchains; `WITH_JVM=0` omits the JVM tools. Host credentials, operator databases, tests and model weights are excluded from the image. Indexing requires writable source locations. The [Docker browser journey](../web/e2e/README.md) builds this combined image and checks Qwen configuration, cost estimation, onboarding, search, answers, clews and restart persistence with HTTP model fixtures. Browser checks use `WITH_JVM=0` and exercise Python and Go indexing.

## Browser with a configured model pair

Add `deployments/app/compose.web.yaml` to either CLI preset to start the browser
and MCP. It copies the selected configuration into `/data/ariadne.yaml` on first
startup, keeping the wizard's edits in the state volume, and mounts source
writable for indexing. Existing configuration is preserved on restart.

```sh
# Cloud: OpenAI, or Claude after editing deployments/app/cloud.yaml.
docker compose -f deployments/app/compose.yaml -f deployments/app/compose.web.yaml up --build -d

# Local: start the provisioned Qwen servers first, then start the application.
docker compose -f deployments/local-v1/compose.yaml -f deployments/app/compose.web.yaml --profile client up --build -d ariadne
```

Open `http://127.0.0.1:8765`; MCP is at `http://127.0.0.1:18000/mcp`
(`ARIADNE_MCP_PORT` can change that port). The separate web network makes these
ports reachable while the local model servers stay on their inference network.
Use a separate Compose project and state volume for each model pair. For
existing external model servers, use a configuration with their reachable URLs
as the mounted template. The wizard uses that configuration's model, shows zero
provider token charges for local inference, and leaves hardware/runtime cost
unestimated. Local onboarding uses the same resumable pipeline as the CLI.

## Cloud suite

From the repository root, export `OPENAI_API_KEY` and set `ARIADNE_SOURCE_DIR` to an absolute path to the repository you want documented. The Compose file defaults to this checkout when that path is omitted.

```sh
export ARIADNE_SOURCE_DIR=/absolute/path/to/your/project
docker compose -f deployments/app/compose.yaml run --rm ariadne config
docker compose -f deployments/app/compose.yaml run --rm ariadne generate --source project --dry-run
docker compose -f deployments/app/compose.yaml run --rm ariadne generate --source project --no-batch --yes
docker compose -f deployments/app/compose.yaml run --rm ariadne search 'How does authentication work?' --source project
```

The cloud preset uses GPT-6 Astra with OpenAI `text-embedding-3-large` at 3072 dimensions. For Claude, set `llm.provider: anthropic`, `llm.model: claude-opus-5-5`, and `llm.api_key_env: ANTHROPIC_API_KEY` in `deployments/app/cloud.yaml`, export that key too, and choose fresh database paths. The mounted YAML is read-only; edit configuration on the host.

For a stdio MCP client, launch:

```sh
docker compose -f deployments/app/compose.yaml run --rm -T ariadne mcp
```

Run Compose from this checkout or give the client an absolute Compose file path. Source is mounted read-only. The named `ariadne-state` volume persists the library, sidecar and generated exports across container removal. Do not remove that volume to switch models.

The bundled cloud preset also supports running the image directly, without mounting a configuration file:

```sh
docker run --rm -e OPENAI_API_KEY \
  -e ARIADNE_CONFIG=/opt/ariadne/examples/cloud.yaml \
  --mount type=bind,src=/absolute/path/to/your/project,dst=/workspace,readonly \
  --mount type=volume,src=ariadne-cloud-state,dst=/data \
  ariadne:local-model-support generate --source project --no-batch --yes
```

## Local suite

First provision and start the two GPU model servers using the [pinned serving recipe](../deployments/local-v1/README.md). This application image does not contain weights or start vLLM. The serving candidate still requires real GPU qualification.

The local Compose client shares the servers' internal network. Its bundled configuration uses `chat:8000` and `embeddings:8000`, the pinned BF16 model identities, and separate local database paths. It requires no cloud credentials or revision-placeholder edits.

```sh
docker compose -f deployments/local-v1/compose.yaml --profile client run --rm ariadne config
docker compose -f deployments/local-v1/compose.yaml --profile client run --rm ariadne doctor --models
docker compose -f deployments/local-v1/compose.yaml --profile client run --rm ariadne generate --source project --batch --yes
docker compose -f deployments/local-v1/compose.yaml --profile client run --rm -T ariadne mcp
```

Use the same `ARIADNE_SOURCE_DIR` setting as the cloud suite. Switching Compose files selects a separate state volume and model pair; existing documents remain in their original library. Generated documents cannot be moved between incompatible model pairs through import or rebuild.

## Published images

Both Compose files accept `ARIADNE_IMAGE` for a registry tag or digest. Once an image is published, set that variable to the published reference and run `docker compose -f deployments/app/compose.yaml pull ariadne` (or use the local Compose path). Local registry push/pull and execution are verified. External registry hosting/authentication, ARM64 execution and real vLLM serving are outside that result.

## Reproduce local distribution verification

```sh
python3 tests/container_distribution.py --output /path/to/new-distribution-check
```

The runner starts a temporary registry and two independent Docker daemons on an internal network. The producer loads the existing application image and pushes it; the consumer starts with an empty image store and receives the image only through a registry pull. It compares filesystem layer hashes and runtime settings, checks the installed CLI/MCP workflows, verifies persisted databases and completed-batch replay after container removal, and runs both Compose configurations without an application checkout or build files. The recorded full run passed every stage in one invocation after rebuilding with the section-fallback fix, including all cleanup commands.

The test requires Docker access and uses privileged Docker-in-Docker containers with private state and Unix sockets; no host Docker socket or credentials are mounted into them. Pinned test infrastructure images are downloaded if absent. Application tests use synthetic loopback model responses and make no paid API calls. Test containers, networks and temporary image archives are removed, while output logs and the original application image remain. Use a fresh output directory for every run. `--compose-only` rechecks distribution and Compose without repeating runtime checks.

## Reproduce the installed-image checks

```sh
docker run --rm --network none \
  --mount type=bind,src="$(pwd)/tests",dst=/checks,readonly \
  --entrypoint python ariadne:local-model-support /checks/container_smoke.py
```

Only the test directory is mounted. The script requires no test dependencies or package installation inside the image. It checks CLI generation, local durable batches, search, rebuild and real MCP sessions against two loopback HTTP fixtures, with cloud credentials absent and present. The output records every CLI result; a failing assertion exits nonzero.
