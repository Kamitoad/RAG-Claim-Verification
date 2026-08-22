# F1 2023 multi-document diagnostic report

- Run: `20260822T101241.115926Z-38c9872d`
- Executed: 2026-08-22
- Scope: three aggregate claims under one Clean-RAG condition
- Git commit recorded by the run: `40bd9ee8c9e12508118983bfabfe82462120ea17`
- Model: local `ragcv-qwen3-4b-pilot:v1`
- Prompt: `verification-v3-baseline-knowledge`

This report records one deliberately small feasibility diagnostic. It is separate from the final
18-claim pilot and must not be pooled with that experiment's metrics.

## Reproducibility and execution

The diagnostic protocol, claims, and configurations were fixed before the single run. The run was
executed from the merged `main` commit while those new diagnostic files were still uncommitted, so
its metadata correctly records `git_dirty: true`. The run snapshots and hashes the exact claims,
configurations, prompts, manifest, ingestion metadata, and raw predictions used for evaluation.

The validated Clean-v4 index was copied before the run to
`indices/f1_2023_multidoc_diagnostic_v1`. The source and target initially contained the same 13
files with no SHA-256 differences. Both ingestion metadata files had SHA-256
`7aff80ae9ded8c1e78ed6a2284f405b12ac1fe3e7960dd8ba3e368167e3ea731`. Query-cache writes were
therefore isolated from the final pilot index.

The run completed in 217,540 ms (about 3 minutes and 38 seconds):

- 3/3 planned predictions were stored;
- all three predictions completed successfully and were valid on the first response;
- no retrieval, provider, parse, repair, or pipeline error occurred;
- offline evaluation reproduced the stored metrics;
- the raw predictions SHA-256 is
  `ce09222a8c2a834d35e8f3bece38a6056128891285c182326de86e88de856309`.

An archived copy of this exact run can be checked without Ollama or LightRAG:

```powershell
.\.venv-pilot\Scripts\python.exe -m rag_claim_verification evaluate `
  --run-dir runs\20260822T101241.115926Z-38c9872d
```

For an independent new execution, first prepare and ingest the Clean-v4 pilot index as documented
in `f1_2023_pilot_reproduction.md`. In a workspace where the diagnostic target does not exist, copy
the validated index and run the diagnostic configuration:

```powershell
if (Test-Path -LiteralPath indices\f1_2023_multidoc_diagnostic_v1) {
  throw "Diagnostic index target already exists; use a fresh workspace."
}

Copy-Item `
  -LiteralPath indices\f1_2023_pilot_clean_jolpica_podium_v4 `
  -Destination indices\f1_2023_multidoc_diagnostic_v1 `
  -Recurse

.\.venv-pilot\Scripts\python.exe -m rag_claim_verification benchmark `
  --config configs\f1_2023_multidoc_diagnostic.yaml
```

Such an execution is a new observation and must not replace or be pooled with the recorded run.

## Retrieval and citation result

LightRAG returned all three race documents for every claim. Their order was Bahrain at rank 1, Abu
Dhabi at rank 2, and Belgium at rank 3.

For the two decisive claims, each of the three documents was annotated as jointly required gold
evidence:

| Metric | Result |
|---|---:|
| Evidence Recall@1 | 0.3333 |
| Evidence Recall@3 | 1.0000 |
| Evidence Recall@5 | 1.0000 |
| Hit Rate@1/@3/@5 | 1.0000 |
| MRR of the first gold document | 1.0000 |

Both decisive predictions cited all three required document IDs. Direct all-required citation
coverage was therefore 2/2 cases, although the generic grounding metric only requires and reports
at least one gold citation. The NEE prediction used no citation.

Retrieval coverage and citation availability were not the limiting components in this diagnostic.

## Classification result

| Claim | Gold | Prediction | Correct |
|---|---|---|---:|
| Three wins across three races | `SUPPORTED` | `REFUTED` | No |
| Exactly two wins across three races | `REFUTED` | `SUPPORTED` | No |
| Exactly six pit stops across three races | `NOT_ENOUGH_EVIDENCE` | `NOT_ENOUGH_EVIDENCE` | Yes |

Overall accuracy and Macro-F1 were both 0.3333.

### Supported three-win claim

All three race documents were present and all three were cited. The Abu Dhabi evidence explicitly
records Max Verstappen at position 1, but the generated reason stated that Charles Leclerc won that
race and returned `REFUTED`. This is an evidence-interpretation failure despite complete retrieval.

### Refuted exactly-two-win claim

All three documents were again present and cited. The generated reason correctly stated that Max
Verstappen won all three races, explicitly described the exactly-two claim as false, and concluded
that it was refuted. The structured `label` nevertheless contained `SUPPORTED`. This is a semantic
label/reason inconsistency that remains schema-valid and is not detected by the current parser.

### NEE pit-stop claim

The model correctly observed that none of the three retrieved race classifications contained
pit-stop counts, returned `NOT_ENOUGH_EVIDENCE`, and cited no document. The closed-corpus
insufficient-evidence behavior therefore worked for this case.

## Interpretation and stop decision

This diagnostic answered the narrow feasibility question in two parts:

1. With three short one-chunk race documents and `top_k=5`, LightRAG retrieved the complete joint
   evidence set and the verifier cited all of it.
2. The local Qwen3 4B verifier did not reliably convert that complete evidence into the correct
   aggregate verdict. One answer misread a directly stated winner; another answer's structured
   label contradicted its own correct explanation.

The result therefore does not support claiming reliable multi-document aggregation for the current
model profile. It also does not show that multi-document RAG is generally infeasible: only three
constructed claims, one model profile, and one run were observed.

The predeclared stop rule applies. No prompt, label, model, `top_k`, evidence order, or metric will
be tuned against these claims, and the diagnostic will not be rerun as a replacement result. A
future long-document diagnostic must use a separate predeclared protocol, claims, index, and run.
