# F1 2023 long-document diagnostic protocol

## Purpose

This diagnostic tests one narrow question that the short-document pilot could not exercise: can the
unchanged local Clean-RAG configuration retrieve and use decisive evidence from the opening, an
overlapping chunk boundary, and the ending of one document that exceeds the configured LightRAG
chunk size?

It is an exploratory architecture check, not an extension of the final 54-case pilot or the
three-claim multi-document diagnostic. Its results must not be pooled with either experiment and
must not be used to tune those completed experiments.

## Fixed setup

- Corpus: one tracked, hand-written synthetic F1 2023 diagnostic dossier.
- Retriever: LightRAG 1.5.4, `hybrid`, `top_k=5`.
- Chunking: fixed windows of 1,800 tokens with 150 tokens of overlap.
- Tokenizer: the LightRAG default `gpt-4o-mini` tiktoken tokenizer.
- Embeddings: FastEmbed `jinaai/jina-embeddings-v2-small-en`, dimension 512.
- Verification model: local `ragcv-qwen3-4b-pilot:v1`.
- Prompt: `verification-v3-baseline-knowledge`.
- Model temperature: `0.0`.
- Requested seed: `17`.
- Conditions: one Clean-RAG condition only; no baseline and no Noisy comparison.
- Index isolation: a fresh `indices/f1_2023_long_document_diagnostic_v1` directory.

No prompt, model, retrieval parameter, fixture passage, claim, label, or expected evidence phrase
may be changed after the definition commit or after observing retrieval output.

## Synthetic fixture and pre-run structural calibration

The fixture is not a news article or an independent research source. Neutral archive commentary
creates distance between three explicitly marked controlled classification statements without
adding other race classifications or pit-stop statistics. This improves experimental control and
redistributability but limits realism.

Using LightRAG's installed fixed-token chunker before freezing the protocol produced this structure:

| Fixed window | Token range | Stored size | Decisive target |
|---|---:|---:|---|
| 0 | 0–1799 | 1,800 | opening Bahrain statement; boundary Belgium statement |
| 1 | 1650–3449 | 1,800 | boundary Belgium statement through overlap |
| 2 | 3300–3840 | 541 | ending Abu Dhabi statement |

The complete fixture contains 3,841 tokens. The decisive sentence starts are fixed at:

- opening Bahrain statement: token 123, present only in window 0;
- boundary Belgium statement: token 1,651, present in windows 0 and 1;
- ending Abu Dhabi statement: token 3,525, present only in window 2.

This calibration used deterministic local tokenization and chunking only. It did not execute a
retrieval query or a model prediction.

## Predeclared claims and expected evidence phrases

The diagnostic contains exactly four atomic propositions:

1. `longdoc_opening_bahrain_winner`: Max Verstappen won the 2023 Bahrain Grand Prix
   (`SUPPORTED`). Expected passage: `Max Verstappen won the 2023 Bahrain Grand Prix.`
2. `longdoc_boundary_belgium_winner`: Max Verstappen won the 2023 Belgian Grand Prix
   (`SUPPORTED`). Expected passage: `Max Verstappen won the 2023 Belgian Grand Prix.`
3. `longdoc_ending_abu_dhabi_winner`: Charles Leclerc won the 2023 Abu Dhabi Grand Prix
   (`REFUTED`). Expected contradictory passage:
   `Max Verstappen won the 2023 Abu Dhabi Grand Prix. Charles Leclerc placed second`.
4. `longdoc_nee_pit_stop_duration`: the fastest pit stop across the selected races lasted exactly
   2.21 seconds (`NOT_ENOUGH_EVIDENCE`). The complete dossier contains no pit-stop duration or count
   and this claim has no gold document ID under the existing closed-corpus convention.

The first three claims use the one stable dossier document ID as gold evidence.

## Evaluation questions

The stored run and raw evidence will be inspected in this order:

1. Did all four cases complete without retrieval, provider, parse, or pipeline errors?
2. For each decisive claim, does at least one returned evidence chunk contain the predeclared exact
   target passage?
3. At what evidence rank does the first target-containing chunk appear?
4. Does each target-containing chunk map to the stable dossier document ID?
5. If the target passage is available, is the verdict correct and consistent with its reason?
6. Does each decisive output cite the dossier, and does the absent claim remain NEE without a
   citation?

Document-level Evidence Recall is insufficient on its own here: every chunk maps to the same gold
document, so an irrelevant chunk from that file can produce a nominal document hit. Target-passage
availability is therefore reported by exact inspection of persisted evidence text. This direct
audit does not add a new metric or change the prediction schema.

## Interpretation and stop rules

- If ingestion does not produce the three predeclared fixed windows, record a structural mismatch
  and do not change the fixture after the definition commit.
- If the expected target passage is absent within `top_k=5`, record a positional retrieval
  limitation and do not increase `top_k`, move the passage, or change chunking on these claims.
- If the target passage is retrieved but a verdict is wrong, record an evidence-interpretation
  limitation and do not tune the prompt or model on these claims.
- If a verdict is correct but its reason contradicts the evidence or its citation is missing,
  record that behavior separately rather than counting label accuracy as complete success.
- If the absent claim receives a decisive label or citation unsupported by the dossier, record an
  insufficient-evidence handling failure.
- If all cases pass, conclude only that three positions in one controlled synthetic document were
  feasible under this exact local configuration. Do not generalize to newspaper articles,
  arbitrary long-form text, different tokenizers, or cross-chunk aggregation.

The diagnostic is executed once after the definition is committed. Any broader or revised
experiment requires a new predeclared fixture, claim set, and protocol.
