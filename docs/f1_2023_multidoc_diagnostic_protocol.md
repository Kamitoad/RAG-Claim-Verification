# F1 2023 multi-document diagnostic protocol

## Purpose

This diagnostic tests one narrow question left open by the final pilot: can the unchanged local
Clean-RAG configuration verify a single aggregate proposition whose decisive evidence is spread
across all three selected race documents?

It is an exploratory feasibility check, not an extension of the final 54-case pilot. Its results
must not be pooled with the final pilot metrics or used to tune that completed experiment.

## Fixed setup

- Corpus content: the unchanged three-document Clean pilot corpus.
- Retriever: LightRAG 1.5.4, `hybrid`, `top_k=5`.
- Verification model: `ragcv-qwen3-4b-pilot:v1`.
- Prompt: `verification-v3-baseline-knowledge`.
- Model temperature: `0.0`.
- Requested seed: `17`.
- Conditions: one Clean-RAG condition only; no baseline and no Noisy comparison.
- Index isolation: an exact copy of the validated Clean-v4 index is queried from
  `indices/f1_2023_multidoc_diagnostic_v1` so diagnostic keyword-cache writes cannot alter the
  frozen pilot index.

No prompt, model, retrieval parameter, corpus document, or existing gold annotation may be changed
after observing the diagnostic outputs.

## Predeclared claims

The diagnostic contains exactly three aggregate propositions:

1. `multidoc_supported_three_wins`: Max Verstappen won all three selected races (`SUPPORTED`).
2. `multidoc_refuted_two_wins`: Max Verstappen won exactly two selected races (`REFUTED`).
3. `multidoc_nee_pit_stops`: Max Verstappen made exactly six pit stops across the selected races
   (`NOT_ENOUGH_EVIDENCE`).

The first two claims require the Bahrain, Belgium, and Abu Dhabi race documents jointly. Their
`gold_document_ids` therefore contain all three stable document IDs. The NEE label is relative to
the closed pilot corpus: none of the three documents contains pit-stop counts, and it has no gold
document IDs under the existing benchmark convention.

## Evaluation questions

The stored run and its raw evidence will be inspected in this order:

1. Did all three cases complete without retrieval, provider, parse, or pipeline errors?
2. For each decisive claim, what fraction of the three required documents appeared at ranks 1, 3,
   and 5?
3. If all required documents were available, was the aggregate verdict correct?
4. Did the decisive output cite all three jointly required documents?
5. Did the absent pit-stop aggregation remain NEE without an invented citation?

The existing evaluator already measures fractional Evidence Recall@k for multiple gold document
IDs. Hit Rate and MRR are secondary here because they become positive after only one relevant
document is found. The existing citation contract requires at least one valid evidence ID, not all
jointly required IDs, so all-document citation completeness will be reported by direct comparison
of `cited_document_ids` with the predeclared gold set rather than by changing the generic metric or
prediction schema.

## Interpretation and stop rules

- If fewer than all three required documents are retrieved within `top_k=5`, record a retrieval
  coverage limitation and do not raise `top_k` on these observed claims.
- If all three documents are retrieved but a verdict is wrong, record an aggregation/reasoning
  limitation and do not tune the prompt or model on these claims.
- If a verdict is correct but not all required documents are cited, record that the current
  citation contract is insufficient for joint-evidence claims.
- If the NEE claim receives a decisive label or citation unsupported by the documents, record an
  insufficient-evidence handling failure.
- If all cases pass, conclude only that three-document aggregation was feasible in this small,
  structured diagnostic. Do not generalize to twelve races, long documents, or arbitrary
  multi-hop reasoning.

The diagnostic is executed once. Any broader or revised experiment requires a new predeclared
claim set and protocol.
