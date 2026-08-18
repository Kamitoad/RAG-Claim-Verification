# F1 2023 final local pilot report

- Run: `20260818T161551.959752Z-248fa8fc`
- Executed: 2026-08-18
- Scope: 18 balanced claims across three conditions (54 planned cases)
- Git commit: `e0bc15ce71bda7dd952414dca18a436150e0c731`
- Prompt: `verification-v3-baseline-knowledge`
- Model: local `ragcv-qwen3-4b-pilot:v1`

This report describes one small, controlled three-event pilot. It does not establish statistical
significance, causal effects, general robustness, or performance across the complete 2023 Formula
One season.

## Reproducibility and operational result

The run was executed from a clean worktree and completed in 2,194,021 ms (about 36 minutes and
34 seconds). All 54 planned cases were persisted, and offline evaluation reproduced the stored
metrics without another retriever or model call.

- 54/54 predictions completed successfully.
- All 54 outputs were valid on the first attempt.
- There were no retrieval, provider, parsing, or pipeline errors.
- No repair calls were required.
- The raw predictions hash is
  `612767345ed3c444f0f6a00a33188e99776d089af02559a3fea1331d734396cf`.
- The claims hash is
  `feafae8ad3f80294d3206f08493d8d788c09e9337c3fe97f30c3c3bd2aa132b4`.
- The resolved benchmark config hash is
  `5a3af208f3b02fdf3e070836ec229978bb52fbdc6efaf305340208c16e7d71b2`.
- The combined prompt hash is
  `1263b7f7a4a2e922cef33247f4b3143fad2231209688af8cf73940f6144fddde`.

The generated run remains ignored and is not committed. Its raw observations are in
`runs/20260818T161551.959752Z-248fa8fc/predictions.jsonl`; derived metrics can be regenerated with
the CLI `evaluate` command.

## Classification result

| Condition | N | Accuracy | Macro-F1 | Correct | First-pass valid | Repairs | Technical errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM baseline | 18 | 0.3333 | 0.1667 | 6/18 | 18/18 | 0 | 0 |
| Clean RAG | 18 | 1.0000 | 1.0000 | 18/18 | 18/18 | 0 | 0 |
| Noisy RAG | 18 | 1.0000 | 1.0000 | 18/18 | 18/18 | 0 | 0 |

The baseline predicted `NOT_ENOUGH_EVIDENCE` for all 18 claims. It therefore classified only the
six gold NEE claims correctly and produced F1 0 for both `SUPPORTED` and `REFUTED`. The v3
clarification did not cause this local 4B model to use parametric Formula One knowledge. This is a
documented weakness of the selected baseline, not a technical failure and not a reason for further
post-result prompt tuning.

Clean and Noisy RAG produced the correct balanced label distribution: six predictions per label in
each condition. There were no Clean-to-Noisy label changes.

## Retrieval and grounding

Clean RAG retrieved the annotated race document at rank 1 for all 12 claims eligible for gold
evidence. Evidence Recall@1, @3, and @5 and MRR were all 1.0.

Noisy RAG retrieved the annotated race document at rank 1 for 11/12 eligible claims and at rank 2
for the remaining claim. Its retrieval values were:

| Metric | Noisy RAG |
|---|---:|
| Evidence Recall@1 | 0.9167 |
| Evidence Recall@3 | 1.0000 |
| Evidence Recall@5 | 1.0000 |
| MRR | 0.9583 |

All 24 decisive RAG predictions (`SUPPORTED` or `REFUTED`) cited evidence. Every one cited its
annotated gold race document, and none cited a non-gold document. The 12 RAG NEE predictions were
uncited, as permitted by the contract, and explained that the retrieved material lacked the
claimed pit-stop or tyre information.

## Measured noise exposure

The Noisy condition returned 90 evidence occurrences: five chunks for each of 18 claims. Of these,
39 (43.3%) came from the four qualifying/sprint Noise documents. Every Noisy case contained at
least one Noise document.

| Retrieval rank | Noise-document occurrences |
|---|---:|
| 1 | 2 |
| 2 | 17 |
| 3 | 9 |
| 4 | 2 |
| 5 | 9 |

The most informative displacement occurred for `r12_refuted_grid`: the Belgian sprint document
ranked first and the annotated race document ranked second. The verifier still returned `REFUTED`
and cited only the race document. A Belgian tyre NEE case also had the sprint document at rank 1
and correctly remained NEE without a citation.

The Noise challenge therefore genuinely affected retrieval ordering but did not change any label
or citation in these 18 cases. This is a useful result for this fixed pilot, but it is not evidence
that the architecture is generally robust to irrelevant or conflicting documents.

## Latency as provenance

Mean end-to-end case time was about 7.6 seconds for the baseline, 51.3 seconds for Clean RAG, and
63.0 seconds for Noisy RAG. These values must not be treated as a controlled performance
comparison. Conditions ran sequentially, model warm state and keyword caches differed, and RAG
prompts contained substantially more text than baseline prompts.

## Interpretation and limitations

For this exact local setup, retrieval supplied the evidence that the baseline did not use. The
result demonstrates that the implemented pipeline can retrieve mapped source documents, resist the
defined true-but-irrelevant Noise set, produce all three labels, and obey the citation contract.

The perfect RAG classification scores must be interpreted conservatively:

- only 18 deliberately balanced claims from three races were tested;
- the corpus contains compact, structured podium classifications that state decisive facts
  directly;
- refuted claims use controlled single-field mutations;
- NEE claims target fields deliberately absent from all documents;
- the Noise set is small and truth-preserving, not adversarial or contradictory;
- only one local 4B model profile and one run were evaluated;
- temperature 0 and a seed do not guarantee complete model determinism.

The defensible conclusion is therefore limited: the controlled pilot completed successfully, RAG
outperformed the weak all-NEE baseline on these claims, and added related session documents reduced
rank-1 gold retrieval once without harming the recorded verdicts or citations. No further prompt
tuning or model run should be performed on this test set after observing these final results.
