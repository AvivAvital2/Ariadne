---
id: generated
type: explanation
title: Quoted "arithmetic"
created_at: '2026-10-10T00:00:00Z'
updated_at: '2026-10-10T00:00:00Z'
source_name: alpha
source_files:
- core.py
metadata:
  generated: true
  nested:
    items:
    - one
    - 'two: value'
    multiline: 'one

      two'
  model_provenance:
    version: 1
    profile_hash: f8a8b4da641b0c0b1480ccdeb73b2dc4ae41e7d6e53d208d78d09c2833337d9c
    kind: generated
ariadne_export:
  version: 1
  models:
    version: 1
    library_uuid: 00000000-0000-4000-8000-000000000001
    profile_hash: f8a8b4da641b0c0b1480ccdeb73b2dc4ae41e7d6e53d208d78d09c2833337d9c
    generation:
      version: 1
      provider: local
      artifact:
        repository: Qwen/Qwen3.8-27B
        revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        weight_format: bf16
      tokenizer_revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    embedding:
      version: 1
      provider: local
      model: Qwen/Qwen3-Embedding-8B
      dimensions: 3072
      artifact:
        repository: Qwen/Qwen3-Embedding-8B
        revision: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
        weight_format: bf16
      recipe:
        name: qwen3-embedding-8b-3072-v1
        tokenizer_revision: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
        pooling: last-token
        native_dimensions: 4096
        stored_dimensions: 3072
        reduction: prefix
        normalization: l2-float64-then-float32
        query_instruction: Given a question about a codebase, retrieve relevant code
          documentation that answers the question.
        document_input: title-newline2-content2000-v1
        chunk_input: ariadne-chunks-v1
        section_input: heading-or-introduction-v1
        add_special_tokens: true
  body_prefix_length: 0
---
# Arithmetic

Adds two integers.
