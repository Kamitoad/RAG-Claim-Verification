# F1 2023 local pilot gate report

- Corrected run: `20260817T134240.218846Z-e5205a24`
- Superseded run: `20260817T092731.765244Z-70c65235`
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
- The Noisy condition loaded its own seven chunks after the Clean condition closed.

The operational gate therefore passed.

## Descriptive classification and retrieval

| Condition | Accuracy | Macro-F1 | Gold evidence at rank 1 | First-pass valid |
|---|---:|---:|---:|---:|
| Baseline | 0.3333 | 0.1667 | n/a | 6/6 |
| Clean RAG | 1.0000 | 1.0000 | 4/4 eligible | 6/6 |
| Noisy RAG | 1.0000 | 1.0000 | 4/4 eligible | 6/6 |

The sample is deliberately too small for significance or generalization.

## Retrieval-noise result

Every Noisy case contained at least one Noise document. Across the six cases, the five returned
chunks contained 13 Noise-document occurrences. The correct race document nevertheless remained
at rank 1 for all four claims with annotated gold evidence, and every verdict remained correct.

This is the intended small methodology signal: the verifier was actually exposed to related
qualifying/sprint material without losing the decisive race evidence. It is still only six cases
and does not establish general robustness.

Retrieval-only comparison of `hybrid`, `naive`, and `mix` on the same Noisy index produced the
same Noise exposure counts and nearly identical document ordering. There is therefore no current
evidence-based reason to change the fixed `hybrid` mode.

## Superseded first run and root cause

The first gate run incorrectly returned the three Clean chunks for the Noisy condition. LightRAG
1.5.4 keeps local JSON/KV data in process-global shared state under the default empty workspace.
Closing an instance with `finalize_storages()` did not clear this state, so opening the Noisy index
after Clean in the same process reused Clean KV data even though the Noisy vector and graph files
were distinct.

The adapter now releases LightRAG's shared state after each complete storage finalization. A
sequence diagnostic then loaded 3 Clean chunks followed by 7 Noisy chunks in the same process.
The original run remains preserved as an observed technical failure but must not be interpreted as
a clean-versus-noisy result.

## Remaining blocker for the full pilot

The baseline predicted `NOT_ENOUGH_EVIDENCE` for all six claims. Its two correct cases were the
two NEE claims; it did not exercise useful parametric Formula One knowledge in this gate. This
may be a capacity/behavior limitation of the 4B model or excessive caution induced by the
baseline instructions. It must be diagnosed before treating the baseline as meaningful.

Latency is not comparable between conditions in this run. Conditions executed sequentially,
the model was warm, LightRAG cached keyword calls, and Noisy prompts contained more evidence than
Clean prompts. The recorded times remain provenance, not a performance conclusion.

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

Keep `hybrid` fixed for the pilot. The offline
[`f1_2023_baseline_diagnostic.md`](f1_2023_baseline_diagnostic.md) review found that all six raw
reasons used the expected absence of external evidence as sufficient justification for NEE and
never attempted a model-knowledge assessment. It proposes one minimal versioned clarification,
subject to user approval, followed by six baseline cases. Any adopted prompt change requires a new
six-claim three-condition gate before the full pilot.
