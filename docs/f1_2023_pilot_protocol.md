# F1 2023 local pilot protocol

## Purpose

This pilot qualifies the complete local path before the full 2023-season corpus is built. It is
an engineering and methodology pilot, not a statistically generalizable result.

The fixed comparison is:

1. the local LLM without retrieval;
2. LightRAG over three race-classification documents (clean);
3. LightRAG over the identical clean corpus plus four true but non-gold qualifying/sprint
   documents (noisy).

Every condition uses the same ordered 18 claims, verifier model, prompt version, temperature,
seed request, retrieval mode, and `top_k=5` where retrieval applies.

## Selected events and source material

The pilot covers rounds 1, 12, and 22: Bahrain, Belgium, and Abu Dhabi. The committed source
plan in `data/sources/f1_2023_pilot_jolpica.yaml` pins seven Jolpica-F1 API URLs.

- Clean: the race result endpoint for each event.
- Noise: Bahrain qualifying, Belgium qualifying and sprint, and Abu Dhabi qualifying.

The deterministic transformer retains the podium (positions 1-3) from every classification. This
uniform pilot rule includes every decisive gold fact (positions 1/2 and the winner's grid field),
avoids claim-by-claim hand selection, and keeps local extraction and verification feasible. Raw
API responses remain complete and hash-preserved, so a later full-field transformation is
possible. The pilot must therefore be described as a podium-result pilot, not as coverage of each
weekend's complete classification.

Jolpica states that its API is free for non-commercial use and licenses the data under
CC BY-NC-SA 4.0. It also disclaims guaranteed correctness. Raw responses, generated corpus
text, local manifests, and acquisition metadata therefore remain ignored local artifacts. The
FIA timing/classification pages are used only for manual factual cross-checking; Formula1.com
content is not copied into this corpus.

## Local models

- Verification and LightRAG extraction/query LLM: local Ollama profile
  `ragcv-qwen3-4b-pilot:v1`, derived from `qwen3:4b-instruct-2507-q4_K_M` by the committed
  Modelfile. It fixes an 8192-token context and a 1024-token generation ceiling.
- Embeddings: `jinaai/jina-embeddings-v2-small-en` through FastEmbed on CPU, 512 dimensions.
- Initial fallback only if the 4B model fails the qualification gate: a Qwen3 8B Q4 instruct
  model. A broad model sweep is out of scope.

LightRAG's own guidance recommends substantially larger local models for high-quality entity and
relationship extraction. The 4B/8B choices are therefore cost- and hardware-constrained pilot
candidates, not an assumption that they meet LightRAG's recommended production capability.

The first real check uses two synthetic documents. The pilot data are ingested only after the
model returns schema-valid JSON and LightRAG can initialize, insert, retrieve, map document IDs,
and close cleanly.

For CPU execution, LightRAG uses one LLM worker, one parallel insert, no additional entity
gleaning pass, 1800-token chunks, and 150-token overlap. These parameters stay identical between
clean and noisy conditions and are included in the index-configuration hash.

## Gold-label construction

For every selected race, six atomic claims are derived:

- two `SUPPORTED` claims copied as propositions from recorded race fields;
- two `REFUTED` claims produced by one controlled mutation, with the same race document as
  contradicting gold evidence;
- two `NOT_ENOUGH_EVIDENCE` claims about pit-stop count or tyre compound, fields intentionally
  absent from both clean and noisy pilot documents.

The latter label is relative to this closed pilot corpus. It does not assert that the fact is
unknowable from every possible Formula One source. Decisive RAG verdicts must cite at least one
retrieved document; baseline outputs must not cite, and NEE may remain uncited.

## Noise and conflict policy

The main noisy condition contains only truth-preserving retrieval noise. It measures whether
closely related but non-gold sessions intrude into the top-k evidence, displace race evidence, or
change the verdict. Synthetic false/conflicting documents are reserved for a separate diagnostic
experiment because the current verifier has no source-trust hierarchy. Conflict results must not
be pooled into the main clean-versus-noisy metrics.

## Interpretation and gates

The 18-claim pilot reports descriptive values only: classification metrics, Evidence Recall@k,
MRR, structured-output/repair rates, technical errors, citations, latency, and paired clean-to-
noisy label changes. No significance, causality, or generalization claim is permitted.

Proceed to the full 2023 season only if:

1. every local source file and generated document validates and is provenance-hashed;
2. both indices pass their identity guards and never share a working directory;
3. all 54 planned cases persist, with technical failures kept separate from NEE;
4. retrieved paths map reliably to manifest document IDs;
5. the 4B model's JSON and citation behavior is usable, or the single documented 8B fallback is
   qualified under the same protocol.

## Current qualification and gate result

The synthetic local-stack qualification and a real clean-index query passed. The retrieved
Bahrain document mapped back to its manifest ID, and the verifier returned a valid `SUPPORTED`
verdict with the required citation on its first response.

An independently built noisy index did not preserve the exact clean graph: the same Bahrain
document yielded a different number of relationships despite temperature 0 and the requested
seed. Temperature and a seed are therefore recorded controls, not a determinism guarantee.

The resulting derived-index workflow is now implemented. Noisy-v5 copies the validated Clean-v4
snapshot, records its exact metadata hash and document content hashes, and submits only four new
Noise documents to LightRAG. All seven document statuses are `processed`; the final Noisy graph
contains 72 nodes and 24 edges.

The six-claim gate completed all 18 cases without technical errors, parsing failures, or repairs.
Clean and Noisy RAG each classified 6/6 correctly and retrieved every eligible gold document at
rank 1. These values are descriptive only. The gate also found two blockers: the baseline emitted
NEE for all six cases, and Noisy retrieval returned exactly the same three race documents as
Clean without retrieving any Noise document. The unchanged 54-case pilot would therefore not
measure actual noise exposure. See `docs/f1_2023_pilot_gate_report.md` for the complete gate
interpretation and next diagnostic decision.
