# CODEX Handoff

Last updated: 2026-08-18

## Current objective

Build a reproducible, locally runnable pilot for evidence-grounded verification of atomic claims
about the 2023 Formula One season. The fixed comparison is:

1. the same local LLM without retrieval;
2. LightRAG over a clean corpus;
3. LightRAG over that identical clean corpus plus reproducibly defined retrieval noise.

Labels remain `SUPPORTED`, `REFUTED`, and `NOT_ENOUGH_EVIDENCE`. This is not a general
fake-news detector or an article-level system.

## Repository and environment state

- Branch: `feature/f1-2023-baseline-diagnostic`.
- Base HEAD: `b04b1e1`.
- Seven reviewed pilot commits are recorded through the LightRAG lifecycle fix `802363e`.
- The retrieval diagnostic, corrected gate documentation, and related tests are committed as the
  subsequent focused diagnostic change.
- Baseline prompt diagnosis is committed in `acf394f`; the approved v3 prompt experiment is
  committed in `b4cc289`, was executed from a clean worktree, and its contract is now frozen in
  all four pilot configurations for the final run.
- Python 3.12.13 is installed through uv and used by the ignored `.venv-pilot` environment.
- The older ignored `.venv` uses Python 3.14.6 and is unsuitable for FastEmbed on this machine
  because Windows Application Control blocks its `_ctypes` module.
- The resolved local stack includes `lightrag-hku==1.5.4`, `fastembed==0.8.0`, and
  `openai==2.54.0`; `uv.lock` is updated.
- Ollama 0.32.14 serves the local profile `ragcv-qwen3-4b-pilot:v1`, derived from
  `qwen3:4b-instruct-2507-q4_K_M`. The downloaded model is about 2.50 GB.
- FastEmbed uses `jinaai/jina-embeddings-v2-small-en`, 512 dimensions, on CPU.
- No paid API, credential, or hosted model was used.

## Implemented pilot

### Source and corpus preparation

- `data/sources/f1_2023_pilot_jolpica.yaml` pins seven Jolpica-F1 URLs for Bahrain,
  Belgium, and Abu Dhabi: three race results plus four qualifying/sprint noise sessions.
- `scripts/prepare_f1_2023_pilot.py` validates the source plan and API identity, rate-limits
  downloads, preserves raw hashes, and deterministically creates text documents, manifests,
  provenance, and claims without overwriting existing outputs.
- The transformer keeps positions 1-3 from every selected session. Full downloaded JSON remains
  local and hash-preserved.
- The clean corpus has three race-podium documents. The noisy corpus is a strict content-preserving
  superset containing the same clean documents plus four true qualifying/sprint podium documents.
- Raw responses, generated corpus files, local manifests, provenance, indices, and runs remain
  ignored. The source plan and gold claims are tracked.

### Claims and prompts

- `data/ground_truth/f1_2023_pilot.jsonl` contains 18 atomic claims: six per label and six per
  event.
- Supported claims are direct propositions from race fields. Refuted claims use one controlled
  mutation. NEE claims concern tyre/stint or pit-stop information deliberately absent from every
  pilot document.
- RAG `SUPPORTED` and `REFUTED` outputs must cite at least one retrieved document. Baseline
  outputs must not cite. NEE may be uncited.
- The pilot now uses prompt version `verification-v3-baseline-knowledge`; the previous v2 prompt
  remains unchanged for provenance.
- The only material v3 clarification concerns baseline mode: missing retrieved evidence alone is
  not a reason for NEE. The RAG citation rules and output contract remain unchanged.

### Local adapters and configuration

- `FastEmbedConfig` is a strict, discriminated local embedding configuration.
- `LightRAGAdapter` loads FastEmbed lazily, validates vector shape and finite values, passes local
  model temperature/seed controls, and supports unauthenticated OpenAI-compatible endpoints.
- The adapter finalizes both instance storages and LightRAG 1.5.4 process-global shared data, so
  sequential Clean/Noisy adapters cannot reuse the preceding index's KV state.
- LightRAG concurrency, gleaning, and parallel insertion are explicit hash-covered settings.
- Benchmark metadata now records FastEmbed and OpenAI package versions.
- Corpus configs may declare a strict `derived_from` base. Derived metadata records the exact
  four-field base identity, base-ingestion metadata SHA-256, document IDs, and content hashes.
- Derived ingestion copies into a never-before-used directory and passes only documents absent
  from the base snapshot to LightRAG. Interrupted or mismatched directories remain unusable.
- `configs/f1_2023_pilot_*.yaml` declare the baseline, clean, and noisy conditions.
- `configs/ollama/qwen3_4b_pilot.Modelfile` fixes an 8192-token context, 1024-token output
  ceiling, temperature 0, and seed 17.
- `scripts/qualify_local_stack.py` checks Ollama, FastEmbed, real LightRAG insertion/retrieval,
  document-ID mapping, strict verdict JSON, citations, and cleanup using synthetic documents.
- `scripts/diagnose_f1_pilot_retrieval.py` records retrieval-only query-mode and sequential-index
  observations without invoking claim verification or inventing retrieval scores.

The methodology and limitations are documented in `docs/f1_2023_pilot_protocol.md`; the first
real gate is interpreted in `docs/f1_2023_pilot_gate_report.md`.

## Real qualification results

- Synthetic end-to-end local-stack qualification passed with a first-response valid verdict and
  correct citation.
- The current clean index is
  `indices/f1_2023_pilot_clean_jolpica_podium_v4`.
- Its ingestion metadata records three documents, manifest hash
  `fd3dd08a0c672a79f8a83c710d4cf4a2cf1b6ad38d59f38295db9b9e8eb8f364`, and a valid
  RAGCV index identity.
- A real Bahrain supported claim retrieved the correct document at rank 1 and returned
  `SUPPORTED` with the correct citation on the first model response. Observed timings were about
  8.4 seconds retrieval and 51.8 seconds generation on this machine.
- The 4B extraction model produced usable entities but mixed graph quality: two documents ended
  without the expected completion delimiter, one malformed relationship was skipped, and the
  Belgium document produced no relationships.

## Derived Noisy index and gate result

Building the noisy index independently was rejected because identical Clean input produced a
different graph despite temperature 0 and seed 17. The approved replacement is implemented:
`indices/f1_2023_pilot_noisy_jolpica_podium_v5` was copied from Clean-v4 and then received only
the four Noise documents.

- The copied base metadata SHA-256 is exactly
  `7aff80ae9ded8c1e78ed6a2284f405b12ac1fe3e7960dd8ba3e368167e3ea731`.
- All seven document statuses are `processed`.
- Noisy-v5 contains 7 chunk vectors, 72 entity vectors/nodes, and 24 relationship vectors/edges.
- Its target ingestion metadata records all seven content hashes and the complete base identity.
- Noise extraction remained semantically mixed: all four outputs lacked the completion delimiter,
  one malformed relation was rejected, relation wording leaked into entities for the sprint, and
  Abu Dhabi qualifying produced 20 entities but zero relations.

The first gate run `runs/20260817T092731.765244Z-70c65235` is superseded: LightRAG's process-global
shared KV state caused Noisy to reuse Clean chunks after Clean closed. It remains an observed
technical failure and must not be interpreted as a clean-versus-noisy comparison.

The adapter now releases that global state after storage finalization. Retrieval-only diagnostics
showed that `hybrid`, `naive`, and `mix` expose the same six gate claims to Noise, so there is no
current reason to change the fixed `hybrid` mode. A same-process sequence then loaded 3 Clean
chunks followed by 7 Noisy chunks.

Corrected gate run `runs/20260817T134240.218846Z-e5205a24` completed 18/18 planned cases in about
9.4 minutes with zero technical errors, zero repairs, and 18 first-pass-valid outputs. Offline
re-evaluation succeeded.

- Baseline: accuracy 0.3333, Macro-F1 0.1667; all six predictions were NEE.
- Clean RAG: accuracy/Macro-F1 1.0; all four eligible gold documents ranked first.
- Noisy RAG: accuracy/Macro-F1 1.0; all four eligible gold documents ranked first.
- Every decisive RAG prediction cited its gold document.
- Every Noisy case contained Noise evidence, with 13 Noise-document occurrences across the 30
  returned chunks.

The technical and noise-exposure gates passed. Baseline validity did not. These small descriptive
values are not research findings or evidence of general robustness.

The partial ignored noisy directory
`indices/f1_2023_pilot_noisy_jolpica_podium_v4` has no RAGCV ingestion metadata and must not be
used. Older ignored experimental index directories also remain untouched; do not delete or reuse
them without user approval.

## Recommended next action

The approved v3 clarification was executed once in
`runs/20260818T093758.404026Z-dad0f43b`: 6/6 first-pass-valid responses, no errors or citations,
but still 6/6 NEE with reasons based only on absent evidence. Accuracy remained 0.3333 and Macro-F1
0.1667. Offline re-evaluation succeeded. The clarification did not produce a model-knowledge
assessment, so Option A did not occur and the predeclared stop rule forbids further prompt tuning or
a model sweep.

The user approved adopting the methodologically clearer v3 contract while accepting the weak 4B
baseline as an observed limitation. The gate, full-pilot, Clean, and Noisy configurations now all
resolve to that prompt. No new benchmark was executed during this configuration-only change.

The next action is one final 18-claim, three-condition pilot run (54 predictions) after explicit
confirmation. A separate repeated six-claim gate is unnecessary because the corrected
three-condition gate already qualified both RAG paths and the isolated v3 diagnostic qualified the
only changed baseline instruction. Keep `hybrid`, claims, gold labels, corpora, model settings, and
retrieval settings fixed; do not tune further after seeing the final results. Latency remains
non-comparable across conditions because order, warm state, keyword caching, and evidence lengths
confound it.

## Verification commands

Use `.venv-pilot` for the local stack:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv-pilot\Scripts\python.exe -m pytest -p no:cacheprovider
.\.venv-pilot\Scripts\python.exe -m ruff check .
.\.venv-pilot\Scripts\python.exe -m ruff format --check .
.\.venv-pilot\Scripts\python.exe -m mypy src
uv lock --check
```

The standard suite must remain network-free. The opt-in external LightRAG test checks the pinned
installed SDK and should be enabled only in the prepared local environment.

Last verified on 2026-08-18:

- standard pytest: 64 passed, 3 opt-in tests skipped as designed;
- combined standard, local-data/gold, and pinned-LightRAG checks: 67 passed;
- Ruff lint: passed;
- Ruff format check: 70 files formatted;
- mypy strict check: 40 source files passed;
- `uv lock --check`: passed;
- benchmark, gate, clean, and derived Noisy-v5 configuration validation: passed;
- noisy-manifest clean-superset validation: 7 documents, passed;
- derived index identity and copied base-metadata hash validation: passed;
- corrected gate completeness and offline re-evaluation: 18/18 cases, passed;
- corrected Noisy retrieval: Noise evidence in 6/6 cases, gold at rank 1 in 4/4 eligible cases;
- v3 baseline diagnostic configuration and prompt contract: passed;
- v3 baseline run completeness and offline re-evaluation: 6/6 cases, passed;
- all four final pilot configurations resolve to the same v3 prompt contract: passed;
- `git diff --check`: passed (Git reports only expected Windows line-ending warnings).

## Important constraints

- Jolpica data are suitable here only for non-commercial university research under the recorded
  CC BY-NC-SA 4.0 terms; correctness is not guaranteed and claims still require spot checks.
- Local source downloads and generated corpora are not automatically redistributed or committed.
- The three-event podium pilot is an engineering/methodology gate, not evidence about the full
  2023 season and not a statistically generalizable result.
- `temperature=0` and a seed are controls and recorded provenance, not a determinism guarantee.
- Do not present synthetic qualification results as semantic research performance.
- Do not run paid providers, scrape additional sources, alter gold labels, or start the full
  benchmark without the relevant user decision.
