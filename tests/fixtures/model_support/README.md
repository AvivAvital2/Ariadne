# Model support regression inputs

These small inputs are required by the offline tests. Generated run logs, database dumps, container evidence, and unrelated provider responses are excluded.

- `contract-v1.json` and `exports-v1/`: frozen model identity, wire protocol, crash recovery, and import provenance examples.
- `retrieval-corpus-v1.json`: frozen retrieval documents, questions, and relevance labels.
- `qwen-responses-v1.json`: the 12 unedited provider response objects used by the six Qwen generation/answer replay cases, selected from the 2026-10-11 blind run by exact content equality. Only the surrounding lookup keys are new.
- `reviewer-drafts-v1.json`: the 11 unchanged drafts used by source-review regression tests, keyed by their original run paths.

The recorded routing diagram uses Mermaid, while the public diagram contract requires DOT. Its replay checks rejection and preservation of the failed durable response; the other five cases check successful generation, retrieval, and restart. Synthetic diagram fixtures use DOT.
