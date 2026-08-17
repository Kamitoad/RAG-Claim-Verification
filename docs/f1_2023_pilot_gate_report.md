# F1 2023 local pilot gate report

- Run: `20260817T092731.765244Z-70c65235`
- Executed: 2026-08-17
- Scope: six balanced claims across three conditions (18 planned cases)

This is an engineering and methodology gate, not a research result and not evidence about the
complete 2023 Formula One season.

## Operational result

- All 18 planned cases were persisted successfully.
- There were no retrieval, provider, parsing, or pipeline errors.
- All 18 model outputs were valid on the first attempt; no repair call was required.
- Every decisive RAG verdict cited a retrieved document, and every citation matched the gold
  document.
- Offline re-evaluation reproduced the stored metrics without a retriever or model call.

The operational gate therefore passed.

## Descriptive classification and retrieval

| Condition | Accuracy | Macro-F1 | Gold evidence at rank 1 | First-pass valid |
|---|---:|---:|---:|---:|
| Baseline | 0.3333 | 0.1667 | n/a | 6/6 |
| Clean RAG | 1.0000 | 1.0000 | 4/4 eligible | 6/6 |
| Noisy RAG | 1.0000 | 1.0000 | 4/4 eligible | 6/6 |

The sample is deliberately too small for significance or generalization.

## Findings that block the full pilot

The baseline predicted `NOT_ENOUGH_EVIDENCE` for all six claims. Its two correct cases were the
two NEE claims; it did not exercise useful parametric Formula One knowledge in this gate. This
may be a capacity/behavior limitation of the 4B model or excessive caution induced by the
baseline instructions. It must be diagnosed before treating the baseline as meaningful.

The Noisy condition retrieved no Noise document in any of its six cases. Its ordered evidence
lists were identical to Clean: the correct race document at rank 1, followed by the other two
race documents. Consequently, the perfect Noisy score does not show robustness to noise; the
current gate did not expose the verifier to noise at all.

Latency is not comparable between conditions in this run. Conditions executed sequentially,
the model was warm, LightRAG cached keyword calls, and Clean/Noisy verifier prompts contained
identical evidence. The recorded times remain provenance, not a performance conclusion.

## Index quality observations

Noisy-v5 was derived from the exact Clean-v4 snapshot and ingested only four additional
documents. Its final state contains seven processed chunks, 72 entity vectors/nodes, and 24
relationship vectors/edges. The 4B extraction model nevertheless showed mixed graph quality:

- completion delimiters were missing for all four Noise extraction responses;
- one malformed relation was rejected because it had four instead of five required fields;
- some relation wording was materialized as additional entities in the sprint graph;
- Abu Dhabi qualifying produced 20 entities but no relations.

The index is technically valid and provenance-complete, but that is not a semantic quality
guarantee.

## Next decision

Do not run the 54-case pilot unchanged. First run a small, explicitly versioned diagnostic to
determine whether the same indices retrieve Noise chunks under a chunk-oriented LightRAG query
mode (`naive` or `mix`) and whether a minimally revised baseline instruction lets the local model
use parametric knowledge without inventing citations. Any adopted query-mode or prompt change
must be fixed before a new gate and the full pilot.
