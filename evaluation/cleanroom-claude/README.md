# Clean-room bare-LLM arm — a fresh Claude Code in a hermetic container

The honest bare-LLM side of the Ariadne-vs-LLM head-to-head: a **new Claude Code
instance** answers the 25 questions with **read-only tools over the raw source
only** — never Ariadne's synthesized docs, the spool pack, MCP, or any eval
artifact. Isolation is *by construction*, not by trust.

## What the container can see — and nothing else

At runtime exactly two volumes are mounted:

| Mount | Mode | Contents |
|---|---|---|
| `/corpus` | **read-only** | a **cleaned** raw checkout (spark, delta, databricks-sdk-py) — `.ariadne/`, `*.scip`, `target/`, `.git/` stripped |
| `/work` | rw | `questions.json` (25 × `{id, question}`, **no hinges**) + `answers.json` |

There is **no** host home, **no** Ariadne repo, **no** `.ariadne`/pack/docs,
**no** `~/.claude` from the host. A container has no host filesystem beyond its
explicit mounts.

## Leak surface, closed vector by vector

- **Host filesystem** — only `/corpus:ro` + `/work` are mounted; nothing else exists inside.
- **Ariadne docs in the spool dir** — `run.sh` builds `/corpus` with `rsync --exclude='.ariadne/' --exclude='*.scip' …` and then **fails closed** if any `.ariadne` / `*.scip` / `pack.db` / `manifest.yaml` survived.
- **MCP** — the image bakes an empty MCP config (`mcpServers:{}`, `enableAllProjectMcpServers:false`); no `.mcp.json` is mounted; `WebFetch`/`Bash` are disallowed, so no Ariadne endpoint is reachable even by hand.
- **Serialized files / residual artifacts (`top25.json`, `synth_*.jsonl`, scores, maps)** — none are in the build context (this dir has none) and none are mounted. The `.dockerignore` whitelists only `entrypoint.py` into the image.
- **The questions themselves** — stripped to `{id, question}`; `hinges_on`, `family`, `repo_a/b` are dropped, so the intended answers never enter the container. The model sees each question only as a CLI prompt (it never reads the questions file).
- **Comments / hints** — the Dockerfile, entrypoint, and system prompt carry no answer hints.

## Tool restrictions (secondary — the container is the hard boundary)

`entrypoint.py` runs Claude Code headless per question with
`--allowedTools "Read,Grep,Glob"` and `--disallowedTools
"Bash,Write,Edit,MultiEdit,NotebookEdit,WebFetch,WebSearch,Task"`, model
`claude-opus-4-8`, cwd `/corpus`. Even if a flag drifted, `/corpus` is read-only,
`/work` holds only the questions + its own answers, and there is no Ariadne data
anywhere in the container to reach.

## Run

```bash
ANTHROPIC_API_KEY=sk-... ./run.sh          # corpus defaults to ../../spool-corpus
# or: CORPUS_SRC=/path/to/raw/checkout ANTHROPIC_API_KEY=sk-... ./run.sh
```

Output → `answers-cleanroom-claude.json` (`[{id, question, answer}]`), graded
next with the **same** 0–10 rubric/hinges as the Ariadne arm, on the same 25.

## Residual (call it out honestly)

The container has default egress so Claude Code can reach `api.anthropic.com`.
With `Bash`/`WebFetch`/`WebSearch` disallowed the model can't initiate network
calls, and there is no Ariadne server to reach — but if you want egress locked to
the model API only, run under a network with an allowlist firewall.
