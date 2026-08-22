# F1 2023 long-document diagnostic report

- Run: `20260822T181104.020270Z-4db7a834`
- Executed: 2026-08-22
- Scope: four atomic claims under one synthetic Clean-RAG condition
- Definition commit: `43d87fb31f3ad07015c24214336eedfaef8ac314`
- Model: local `ragcv-qwen3-4b-pilot:v1`
- Prompt: `verification-v3-baseline-knowledge`

This report records one deliberately small architecture diagnostic. It is separate from the final
18-claim pilot and the multi-document diagnostic and must not be pooled with either experiment.

## Reproducibility and execution

The synthetic dossier, manifest, claims, configurations, exact target passages, interpretation
rules, and stop rules were committed before ingestion or querying. The benchmark metadata records
that definition commit and snapshots and hashes the exact claims, configuration, prompts,
dependency lock, source tree, and raw predictions.

The run records `git_dirty: true` because the unrelated local file
`docs/f1_2023_pilot_auswertung_de.md` remained intentionally untracked. At execution time there were
no uncommitted changes to the experiment definition or application source. The run's predictions
SHA-256 is:

```text
0a3998ce06fc644db385201410f22d4a15fbe562ece7f2ad64104b2b8e3166d8
```

The fresh index used corpus hash
`aca1e623c56b17ac31993aa080ba207c1729d305c18be57cf16630292736ada2` and document
SHA-256 `ab1cf26d726bda1f689871ddef13d6b26069c06707a450f3e03dd587a2438613`.
After the four queries, only `kv_store_llm_response_cache.json` differed from the pre-query index;
the other 12 index files, including documents, chunks, vectors, graph, and ingestion metadata,
remained byte-identical.

The benchmark completed all four cases in 1,016,115 ms (about 16 minutes and 56 seconds):

- 4/4 planned predictions were stored successfully;
- all four outputs were schema-valid on the first model response and required no JSON repair;
- no retrieval, parse, or pipeline error occurred;
- the first verification request timed out once and succeeded through the one configured provider
  retry, giving that case two provider attempts;
- offline evaluation reproduced the stored metrics.

An archived copy of this exact run can be checked without Ollama, LightRAG, or FastEmbed:

```powershell
.\.venv-pilot\Scripts\python.exe -m rag_claim_verification evaluate `
  --run-dir runs\20260822T181104.020270Z-4db7a834
```

For an independent new execution, first prepare the local Ollama, LightRAG, and FastEmbed stack as
documented in the [pilot reproduction guide](f1_2023_pilot_reproduction.md). The diagnostic
fixture, manifest, claims, and configurations are tracked, so no research-data preparation or
external source download is needed. In a fresh clone where the diagnostic index does not yet
exist, run:

```powershell
.\.venv-pilot\Scripts\python.exe -m rag_claim_verification validate-corpus `
  --manifest data\manifests\f1_2023_long_document_diagnostic.jsonl

.\.venv-pilot\Scripts\python.exe -m rag_claim_verification validate-config `
  --config configs\f1_2023_long_document_clean.yaml

.\.venv-pilot\Scripts\python.exe -m rag_claim_verification validate-config `
  --config configs\f1_2023_long_document_diagnostic.yaml

if (Test-Path -LiteralPath indices\f1_2023_long_document_diagnostic_v1) {
  throw "Long-document diagnostic index already exists; use a fresh clone or a separately approved target."
}

.\.venv-pilot\Scripts\python.exe -m rag_claim_verification ingest `
  --config configs\f1_2023_long_document_clean.yaml

.\.venv-pilot\Scripts\python.exe -m rag_claim_verification benchmark `
  --config configs\f1_2023_long_document_diagnostic.yaml
```

The benchmark prints its new run directory. Its derived files can then be regenerated without
further model or retrieval calls:

```powershell
.\.venv-pilot\Scripts\python.exe -m rag_claim_verification evaluate `
  --run-dir runs\NEW_RUN_ID
```

This creates a new observation and must not replace or be pooled with the recorded run above.

## Ingestion and stored chunk structure

LightRAG ingested the one 3,841-token dossier into the three predeclared fixed windows. The stored
second chunk re-encodes to 1,799 rather than 1,800 tokens because persisted chunk text is stripped
of boundary whitespace; rechunking the persisted full document still yields the configured
1,800/1,800/541 token windows.

| Chunk | Persisted token count | Controlled target passages |
|---|---:|---|
| `chunk-000` | 1,800 | opening Bahrain; boundary Belgium |
| `chunk-001` | 1,799 | boundary Belgium through overlap |
| `chunk-002` | 541 | ending Abu Dhabi |

The target distribution therefore matched the protocol exactly. Ingestion completed with 70 graph
nodes and 3 relations. The local extraction model omitted LightRAG's expected completion delimiter
for every chunk and produced one malformed relation record in the final chunk; LightRAG warned,
discarded the malformed record, and completed ingestion. These extraction warnings are separate
from the later retrieval and classification observations.

## Direct target-passage retrieval

Every query returned all three available chunks. Because all chunks map to the same document ID,
the generic document-level Evidence Recall@1 is 1.0 even when the decisive passage is not in the
first-ranked chunk. The predeclared direct passage audit is therefore the meaningful positional
result:

| Claim | Expected target location | First target-containing evidence rank | Available |
|---|---|---:|---:|
| Bahrain winner | opening, `chunk-000` | 2 | Yes |
| Belgium winner | overlap, `chunk-000` and `chunk-001` | 2 | Yes |
| Abu Dhabi winner contradiction | ending, `chunk-002` | 1 | Yes |
| Exact pit-stop duration | absent from complete dossier | n/a | Correctly absent |

The boundary target appeared at ranks 2 and 3 because it was preserved in both overlapping chunks.
The three decisive predictions cited the stable dossier document ID, and the NEE prediction cited
no document. Positional evidence availability and citation behavior therefore passed all
predeclared checks.

## Classification result

| Claim | Gold | Prediction | Correct |
|---|---|---|---:|
| Bahrain winner at opening | `SUPPORTED` | `REFUTED` | No |
| Belgium winner at overlap | `SUPPORTED` | `SUPPORTED` | Yes |
| Charles Leclerc won Abu Dhabi | `REFUTED` | `REFUTED` | Yes |
| Exact pit-stop duration | `NOT_ENOUGH_EVIDENCE` | `NOT_ENOUGH_EVIDENCE` | Yes |

Overall accuracy was 0.75 and Macro-F1 was approximately 0.7778.

### Opening Bahrain claim

The exact supporting sentence was present in `chunk-000` at evidence rank 2 and the output cited
the dossier. The reason nevertheless invented that the statement was later corrected and described
the dossier as conflicting, although no such correction or conflicting Bahrain result exists. The
`REFUTED` label is therefore an evidence-interpretation failure despite successful positional
retrieval.

This was also the slowest case. Its first provider request timed out and the configured retry
succeeded. Total case latency was 372,844 ms, of which 20,838 ms was retrieval and 352,006 ms was
generation across the two provider attempts.

### Boundary Belgium claim

The supporting sentence was available twice through overlap, first at rank 2. The output correctly
returned `SUPPORTED`, identified the marked boundary statement, and cited the dossier.

### Ending Abu Dhabi claim

The contradictory passage was available in the later-only chunk at rank 1. The output correctly
stated that Max Verstappen won and Charles Leclerc placed second, returned `REFUTED`, and cited the
dossier.

### NEE pit-stop claim

No evidence chunk contained the claimed `2.21` value. The output correctly returned
`NOT_ENOUGH_EVIDENCE` without a citation. Its reason correctly identified that no pit-stop duration
value was supplied, although the wording that the document contained no *mention* of pit-stop
durations was imprecise: the synthetic notices explicitly say that such values are absent.

## Interpretation and stop decision

The narrow positional retrieval question passed for all three predeclared locations. Under this
exact local setup, LightRAG exposed decisive text from the opening, a fixed-window overlap, and a
later-only chunk of one synthetic long document. The result also demonstrates why document-ID
retrieval metrics alone are insufficient for long single-document inputs: they cannot distinguish
the correct passage from another chunk of the same file.

Classification remained less reliable than retrieval. The local Qwen3 4B verifier converted three
cases correctly but invented an internal contradiction for the opening claim despite receiving the
supporting sentence. This is consistent with the separate multi-document diagnostic's broader
finding that complete evidence availability does not guarantee correct evidence interpretation.

The result does not establish performance on newspaper articles or arbitrary long-form text. Only
one artificial dossier, four constructed claims, three chunks, one model profile, and one run were
observed. All three chunks were returned for every query, so this diagnostic also does not show how
the current `top_k` behaves when a document produces more than five chunks or competes with other
long documents.

The predeclared stop rule applies. The fixture, claim wording, labels, prompt, model, chunking,
`top_k`, and evidence order will not be tuned against these outputs, and this diagnostic will not
be rerun as a replacement result. Remaining project work should synthesize the completed pilot and
the two separate diagnostics in the university report rather than add another experiment.
