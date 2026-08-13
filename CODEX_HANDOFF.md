# CODEX Handoff

Last verified: 2026-08-12

## Current project goal and scope

Build a structured, reproducible Python research application for evidence-grounded
verification of atomic factual claims in a closed knowledge domain.

The current agreed research domain is **the complete 2023 Formula One season**. The
previous Michael Schumacher 1991-2012 domain is obsolete. The application code is
largely domain-independent, but the checked-in README, configurations, synthetic
documents, manifests, example claims, and some tests have **not yet been migrated** and
still refer to Michael Schumacher. Those files describe the old fixture domain, not the
current research target.

Each claim receives one of three labels:

- `SUPPORTED`
- `REFUTED`
- `NOT_ENOUGH_EVIDENCE`

The intended controlled comparison remains:

1. an LLM-only baseline without retrieval;
2. RAG over a clean 2023 Formula One corpus;
3. RAG over the identical clean corpus plus a reproducibly defined noisy document set.

The project accepts pre-formulated atomic claims. It is not a general fake-news
detector, does not extract claims from articles, and does not currently provide a UI or
an interactive wiki.

## Repository state at this handoff

- Branch: `feature/initial-codebase`
- HEAD: `daafbef` (`Chore: Add repository development guidelines`).
- `origin/feature/initial-codebase` is at `2d0fa99`; the local branch is two commits
  ahead and not behind. The two local commits add the deterministic smoke benchmark
  (`b4ee26a`) and the repository-wide `AGENTS.md` rules (`daafbef`).
- The worktree currently has unstaged changes in `README.md` and `CODEX_HANDOFF.md`;
  the new `uv.lock` is staged. No other paths appear in `git status`.
- The `README.md` worktree change documents the smoke runner and expanded run artifacts,
  but its Schumacher scope and examples remain stale.
- `uv.lock` is a 79-package uv resolution, passes `uv lock --check`, and is available to
  benchmark metadata for path/hash recording. Whether it should remain staged and be
  committed is still a project decision.
- `runs/` contains ignored deterministic synthetic smoke runs. They are engineering
  artifacts, not research results. The latest verified run is
  `runs/20260812T151029.999289Z-662dadf3`.
- No `indices/` directory exists.
- No `.env` file exists. The tracked `.env.example` contains only endpoint/model examples
  and empty API-key values.

The local ignored `.venv` uses Python 3.14.6 and contains the core and development
dependencies used for the verification commands recorded below. The optional LightRAG
extra is not installed in that environment.

## What is already implemented

The repository contains an offline-testable MVP with:

- a Python package using a `src` layout and Python `>=3.11`;
- a Typer CLI with six commands: `validate-config`, `validate-corpus`, `ingest`,
  `verify`, `benchmark`, and `evaluate`;
- strict Pydantic v2 models for documents, claims, evidence, predictions, and run
  metadata;
- YAML configuration loading with environment interpolation and config-relative path
  resolution;
- JSONL manifest and claim loading with duplicate-ID and line-specific validation;
- all-or-nothing document loading for UTF-8, missing-file, and empty-file checks;
- validation that a noisy manifest preserves every clean document's content and
  semantic metadata;
- an adapter for the pinned optional dependency `lightrag-hku==1.5.4`;
- ingestion metadata tying an index to corpus, content, configuration, and LightRAG
  hashes/version;
- a deterministic in-memory keyword retriever for tests and offline demonstrations;
- an asynchronous OpenAI-compatible Chat Completions client with bounded retries;
- versioned prompt files for baseline and RAG verification;
- strict JSON output parsing, citation allow-listing, and one bounded repair attempt;
- a sequential multi-condition benchmark runner;
- classification metrics, retrieval metrics, confusion matrices, failure records, and
  Markdown/JSON/CSV reports;
- re-evaluation of a stored `predictions.jsonl` without model or retriever calls;
- schema-versioned case manifests and one atomically checkpointed result per planned
  claim-condition case;
- exact snapshots of claims, prompts, source configs, manifests, available ingestion
  metadata, and hashes under each run's `inputs/` directory;
- explicit case, retrieval, parse, model, and technical-error states plus separately
  defined stage timings;
- provider-reported model revision, response ID, fingerprint, token usage, retry attempt
  count, requested model settings, seed, source-tree hash, and lockfile hash where
  available;
- deterministic classification, retrieval, grounding-proxy, structured-output,
  technical-error, latency, and completeness metrics;
- unit tests and one complete offline integration workflow using controlled fake
  components.

The tracked data under `data/` are synthetic teaching fixtures. Their labels and perfect
offline integration-test score are not research findings.

## Actual application execution flow

### Configuration and validation

1. `cli.py` receives a command and config path.
2. `config.py` loads `.env` without overriding existing environment values, reads YAML,
   expands `${NAME}` or `${NAME:-default}`, validates strict Pydantic models, and resolves
   declared paths relative to the YAML file.
3. Prompt files are loaded and checked for required placeholders.
4. Claims or manifests are parsed from JSONL. Duplicate IDs, malformed records, missing
   files, non-UTF-8 files, and empty documents are rejected.
5. When a clean manifest is supplied, `validate_noisy_superset` verifies that every clean
   document occurs unchanged in the noisy corpus.

### Corpus ingestion

1. `ingest` loads and validates the complete corpus before contacting LightRAG.
2. `LightRAGAdapter` checks that LightRAG 1.5.4 is installed and configures its LLM and
   embedding functions from the OpenAI-compatible settings.
3. `IngestionService` rejects a non-empty unknown working directory and rejects metadata
   belonging to different corpus content, index settings, or LightRAG version.
4. The adapter initializes LightRAG storage and calls `ainsert` with document text,
   stable manifest IDs, and declared file paths.
5. After a successful call, `ragcv_ingestion_metadata.json` is written to the index
   directory. The adapter then finalizes LightRAG storage.

This path is implemented and unit-tested with fakes, but it has not been executed here
against an installed LightRAG package or a real provider.

### Single-claim verification

1. `verify` loads one corpus configuration and creates a deterministic ad-hoc claim ID.
2. The production factory reloads the manifest, checks that the configured LightRAG
   index metadata matches it, and creates the retriever and verification client.
3. The retriever calls LightRAG `aquery_data` with the configured query mode and `top_k`.
4. Returned chunks are converted to ranked `Evidence`. Returned file paths are mapped
   back to manifest document IDs where possible. LightRAG retrieval scores remain
   `null`, because this adapter does not receive a documented per-chunk score.
5. `PromptBuilder` renders the claim and JSON evidence into the versioned prompt.
6. `ClaimVerifier` calls the OpenAI-compatible `chat/completions` endpoint.
7. The response must be one strict JSON object with `label`, `reason`, and
   `cited_document_ids`. Unknown fields, invalid labels, duplicate JSON keys,
   non-standard constants, and citations outside the retrieved document IDs are
   rejected.
8. One repair request is attempted after a parse/schema failure. A second failure creates
   a failed prediction with the raw and repair output; no fallback label is invented.

### Benchmark execution

1. `benchmark` loads all gold claims in file order and validates prompt and corpus inputs.
2. A unique run directory, resolved config, complete `case_manifest.jsonl`, exact input
   snapshots, and initial `metadata.json` are created before provider calls.
3. Conditions run sequentially. The baseline receives no evidence. Each RAG condition
   retrieves evidence before invoking the same verifier interface.
4. Per-claim retrieval, provider, parsing, and pipeline errors are recorded as failed
   predictions where possible.
5. `predictions.jsonl` is atomically checkpointed after every case. Caught run-level
   failures create explicit error results for every case not yet executed.
6. After all conditions finish, the runner writes deterministic aggregate metrics, CSV
   files, `failures.jsonl`, and `summary.md`, then hashes the final raw predictions.
7. `evaluate` validates the case-manifest, predictions, resolved-config, and claims-
   snapshot hashes plus the complete case matrix before regenerating derived files
   without external calls.

Generated run files are:

```text
runs/<run-id>/
|-- metadata.json
|-- resolved_config.yaml
|-- case_manifest.jsonl
|-- predictions.jsonl
|-- inputs/
|   |-- benchmark.yaml
|   |-- claims.jsonl
|   |-- hashes.json
|   |-- prompts/
|   |-- corpus_configs/
|   |-- manifests/
|   `-- ingestion_metadata/
|-- metrics.json
|-- metrics.csv
|-- confusion_matrix.csv
|-- failures.jsonl
`-- summary.md
```

## Important files and responsibilities

### Project and documentation

- `pyproject.toml` - package metadata, runtime/dev/LightRAG dependencies, CLI entry point,
  and pytest/Ruff/mypy configuration.
- `uv.lock` - current 79-package uv resolution; staged and valid according to
  `uv lock --check`, but not yet committed.
- `AGENTS.md` - committed repository-wide development, safety, reproducibility, and
  definition-of-done rules. Current-status details belong in this handoff instead.
- `README.md` - detailed user documentation for the MVP; its stated Schumacher domain is
  now stale.
- `CODEX_HANDOFF.md` - verified project state and next-step handoff.
- `.env.example` - names and example values for verification, LightRAG LLM, and embedding
  endpoints. It contains no real secret.
- `.gitignore` - excludes local environments, credentials, indices, generated runs, and
  local corpora.

### Configuration and prompts

- `configs/benchmark.yaml` - current three-condition example benchmark. It still uses the
  old experiment name and synthetic claims.
- `configs/baseline.yaml` - current baseline-only example.
- `configs/clean_corpus.yaml` / `configs/noisy_corpus.yaml` - current LightRAG corpus,
  model, embedding, prompt, and index settings. Corpus IDs and paths still use the old
  fixture domain.
- `prompts/verification_system.txt` - label definitions, baseline/RAG evidence rules, and
  exact JSON output contract.
- `prompts/verification_user.txt` - template for mode, labels, claim, and evidence.
- `prompts/verification_repair.txt` - versioned and hash-covered bounded repair prompt.
- `configs/smoke_benchmark.yaml` and its two corpus configs - deterministic synthetic
  full-path smoke benchmark using the production HTTP client and in-memory retriever.

### Data

- `data/README.md` - manifest format and local/copyrighted-data guidance.
- `data/manifests/*.jsonl` - current three-document clean and five-document noisy
  synthetic manifests.
- `data/ground_truth/claims.example.jsonl` - five synthetic Schumacher claims.
- `data/corpora/clean/*.txt` and `data/corpora/noisy/*.txt` - synthetic fixture text only.

No real 2023 Formula One corpus, provenance table, annotation protocol, or final claim
set exists yet.

### Application package

- `src/rag_claim_verification/cli.py` - CLI commands, exit codes, and top-level workflow
  wiring.
- `src/rag_claim_verification/config.py` - strict config models, interpolation, path
  resolution, comparability validation, and index config hashing.
- `src/rag_claim_verification/models/` - strict domain and persisted-run models.
- `src/rag_claim_verification/ingestion/manifest.py` - manifest parsing, corpus hashing,
  path resolution, and clean/noisy superset checks.
- `src/rag_claim_verification/ingestion/loader.py` - complete UTF-8 document validation
  and loading.
- `src/rag_claim_verification/ingestion/service.py` - guarded ingestion orchestration and
  index identity metadata.
- `src/rag_claim_verification/retrieval/base.py` - framework-independent retriever
  protocol.
- `src/rag_claim_verification/retrieval/in_memory_retriever.py` - deterministic whole-
  document keyword retriever used by tests/demos.
- `src/rag_claim_verification/retrieval/lightrag_adapter.py` - all LightRAG 1.5.4 imports,
  initialization, insertion, structured retrieval, path mapping, and cleanup.
- `src/rag_claim_verification/llm/base.py` - model-client protocol.
- `src/rag_claim_verification/llm/openai_compatible.py` - asynchronous Chat Completions
  HTTP client and retries.
- `src/rag_claim_verification/llm/structured_output.py` - strict JSON/schema/citation
  validation.
- `src/rag_claim_verification/verification/` - prompt rendering, baseline wrapper, and
  evidence-grounded verifier with one repair attempt.
- `src/rag_claim_verification/evaluation/benchmark.py` - component factory, claim loading,
  controlled run execution, metadata, and artifact orchestration.
- `src/rag_claim_verification/evaluation/classification_metrics.py` - accuracy, per-class
  precision/recall/F1, Macro-F1, and confusion matrix including `NO_PREDICTION`.
- `src/rag_claim_verification/evaluation/retrieval_metrics.py` - Evidence Recall@1/@3/@5
  and MRR from concrete document IDs.
- `src/rag_claim_verification/evaluation/reporting.py` - JSON/CSV/Markdown reports and
  observable failure categories.
- `src/rag_claim_verification/evaluation/evaluate.py` - stored-prediction re-evaluation.
- `src/rag_claim_verification/utils/` - atomic file output and SHA-256 helpers.

### Tests

- `tests/unit/` covers models, manifests, hashing, config comparability, prompt rendering,
  output parsing, repair behavior, both retriever boundaries, ingestion protection,
  HTTP payloads, and metrics.
- `tests/integration/test_benchmark_workflow.py` runs all three conditions and stored-run
  re-evaluation with fake verification and the in-memory retriever.
- `tests/integration/test_lightrag_external.py` only checks the installed LightRAG version
  when explicitly enabled. It does not perform ingestion, retrieval, or a provider call.

## Commands verified successfully on 2026-08-12

The following commands ran successfully in this repository on Windows with uv 0.12.0
and Python 3.14.6:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; .\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy src
uv lock --check
.\.venv\Scripts\python.exe scripts\run_smoke_benchmark.py
.\.venv\Scripts\python.exe -m rag_claim_verification evaluate `
  --run-dir runs/<smoke-run-id>
```

Observed results:

- pytest: **44 passed, 1 skipped**;
- skipped test: opt-in installed-version-only LightRAG test;
- Ruff lint: passed;
- Ruff format check: **59 files already formatted**;
- mypy strict check: no issues in 38 source files;
- `uv lock --check`: passed with 79 resolved packages;
- the deterministic synthetic smoke benchmark completed all 15 planned cases;
- three deliberately invalid first responses were repaired once and preserved in raw
  case results;
- all 15 cases succeeded, and `failures.jsonl` contains zero records;
- stored-run re-evaluation reproduced byte-identical `metrics.json`;
- latest smoke run: `runs/20260812T151029.999289Z-662dadf3`;
- latest `metrics.json` SHA-256:
  `6291E7BD53ED81ADA70304A5E915086A86BA7B071AC4CE418EBB3E073C1EA5B5`.

These checks made no external, paid, or LightRAG calls. The smoke benchmark used a local
deterministic OpenAI-compatible fixture and the production HTTP client.

## Known limitations, failures, and unfinished areas

### Domain and research data

- The agreed F1 2023 scope is not reflected in checked-in docs/configs/data/tests.
- There is no real F1 2023 clean corpus or defined noisy corpus.
- There is no provenance/licensing record for research data.
- There is no final claim set, annotation guide, independent review, or adjudication
  record.
- Gold document IDs in the synthetic examples do not establish a method for exhaustively
  identifying relevant evidence in a real corpus.
- No research benchmark has been executed and no current output may be reported as a
  research result.

### External integration

- `lightrag-hku` was not installed or called during the current verification.
- No real LightRAG ingestion or query has been run for this repository.
- `ingest`, provider-backed `verify`, and provider-backed `benchmark` are therefore not
  verified.
- No hosted model, local LLM, embedding model, API endpoint, or hardware profile has been
  selected for the final experiment.
- FastEmbed is not a dependency and no FastEmbed adapter exists.
- Local OpenAI-compatible endpoints are configurable in principle and the HTTP client is
  unit-tested with a mock transport, but Ollama/LM Studio/local-model operation has not
  been run.
- No LLM-wiki workflow or interface exists.

### Known implementation boundaries

- LightRAG retrieval scores are stored as `null`.
- Retrieval metrics become unavailable if LightRAG paths cannot be mapped to manifest
  document IDs.
- RAG `SUPPORTED` and `REFUTED` outputs are not currently required by schema validation
  to contain at least one citation; only unknown citations are rejected.
- Re-running ingestion with matching metadata calls the ingestor again rather than
  returning an explicit already-ingested result.
- Stored-run `evaluate` validates the hashes of `case_manifest.jsonl`,
  `predictions.jsonl`, `resolved_config.yaml`, and `inputs/claims.jsonl`, plus exact case
  completeness. It does not currently re-hash every prompt, corpus configuration,
  manifest, ingestion-metadata snapshot, or document-content hash listed under
  `inputs/`.
- Model output JSON rejects duplicate keys and non-standard numeric constants, but the
  generic JSONL/YAML input readers use the standard parsers and do not explicitly reject
  duplicate object/mapping keys. Strict Pydantic validation still rejects unknown
  fields after parsing.
- Confidence intervals, paired statistical tests, repeated-run aggregation, token/cost
  accounting, evidence-conflict detection, and temporal filtering are not implemented.

### Observed environment failures

- Bare `python` resolves to the unavailable Windows Store alias in the reviewed shell;
  use `uv run` or `.venv\Scripts\python.exe`.
- A previous `uv run --frozen pytest` without `--extra dev` failed because pytest/Ruff/
  mypy were not installed in that environment. The current repository-local `.venv`
  contains the development tools and produced the successful results above.
- `mypy --cache-dir NUL src` fails internally on Windows because `NUL` is a device path;
  the canonical `mypy src` invocation passes.
- Python 3.14.6 passed the offline suite, but the real LightRAG integration has not been
  verified on that interpreter.

## Decisions already made

- The target knowledge domain is the complete Formula One 2023 season.
- Inputs are atomic claims and outputs use the three fixed verdict labels.
- The main experiment compares baseline, clean RAG, and noisy RAG over the same claims.
- Retrieval and verification remain separate interfaces so failures can be analyzed
  independently.
- Domain/persistence models and model output are strictly validated, and invalid
  predictions do not receive fabricated fallback labels.
- Prompt files and run artifacts are versioned/hashed for reproducibility.
- LightRAG-specific code remains isolated behind the retriever adapter; the repository
  is currently pinned to LightRAG 1.5.4.
- Synthetic fixtures are demonstrations only.
- A local LLM, FastEmbed, or an LLM-wiki concept may be explored later, but none is an
  implemented or selected part of the current experiment.

## Decisions still open

- Exact inclusion/exclusion boundaries within the 2023 season: races only versus
  qualifying, sprints, practices, telemetry, weather, and narrative sources.
- Permissible source collection, acquisition method, normalization, provenance, and
  redistribution policy.
- Definition and reproducible construction of the noisy corpus.
- Number and class balance of claims.
- Annotation, independent review, disagreement, and adjudication protocol.
- Whether every evaluable claim needs exhaustive `gold_document_ids` or retrieval
  metrics use a documented subset.
- Hosted versus local verification and LightRAG models.
- Embedding provider/model/dimension, including whether FastEmbed is worth adding.
- Final target Python version and whether the currently staged `uv.lock` should be
  committed.
- Repeated-run policy and statistical analysis appropriate for the final claim count.
- Whether to harden LightRAG path-to-document mapping and require citations for decisive
  RAG verdicts before the pilot.
- Whether full stored-input hash verification and duplicate-key rejection for JSONL/YAML
  should be required before the pilot or recorded as post-pilot hardening.

## Single most sensible next milestone

Produce **one reproducible, provider-backed F1 2023 pilot benchmark**.

The milestone is complete only when:

1. the research-facing README, configurations, data examples, and tests use F1 2023;
   any retained Schumacher material is isolated and explicitly named as a legacy
   engineering fixture rather than the current research scope;
2. a small permissible 2023 clean corpus, a documented noisy superset, and reviewed
   pilot claims exist and pass both corpus validators;
3. the final pilot LLM and embedding choices are recorded;
4. LightRAG 1.5.4 is installed and both pilot indices ingest successfully;
5. returned evidence paths map to manifest document IDs;
6. baseline, clean RAG, and noisy RAG run over the same pilot claims;
7. the generated run contains predictions, metrics, failure analysis, metadata, and no
   unexplained pipeline or parse failures.

Do not add a UI, FastEmbed integration, LLM-wiki layer, additional retriever, or large
statistical subsystem before this pilot proves the existing provider boundary works.
