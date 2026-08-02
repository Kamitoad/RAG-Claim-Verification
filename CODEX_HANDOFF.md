# CODEX Handoff

## Original goal

Build a structured, reproducible Python research application for evidence-grounded
verification of atomic factual claims in a controlled domain. The initial domain is
Michael Schumacher's Formula One career from 1991 through 2012.

The system must classify each claim as one of:

- `SUPPORTED`
- `REFUTED`
- `NOT_ENOUGH_EVIDENCE`

The MVP was intended to compare three controlled experimental conditions over the same
claims and compatible model settings:

1. an LLM-only baseline without retrieval;
2. RAG over a clean domain corpus;
3. RAG over the complete clean corpus plus irrelevant or partially relevant documents.

The requested deliverable was a CLI-first application with validated JSONL/YAML inputs,
LightRAG ingestion and retrieval, an OpenAI-compatible model client, strict structured
predictions, reproducible run artifacts, classification and retrieval metrics, offline
tests, and English technical documentation. It is explicitly not a general fake-news
detector.

## Current state

The offline-testable MVP is implemented. It includes the application package, CLI,
configuration and domain models, ingestion validation, LightRAG adapter, deterministic
test retriever, OpenAI-compatible LLM client, verification pipeline, benchmark runner,
evaluation reports, synthetic example inputs, and tests.

The tracked example documents and claims are synthetic demonstration fixtures. They are
not real research data and their outputs must not be reported as research results.

No real provider-backed LightRAG ingestion or query was run during implementation.
Completing such a run requires installing the optional LightRAG dependency, selecting
providers, and supplying credentials or local OpenAI-compatible endpoints.

At handoff time, the MVP is committed as `d87aafb` (`Feat: First Version of Codebase`) on
`feature/initial-codebase`, and that commit is also the tip of
`origin/feature/initial-codebase`. Only this `CODEX_HANDOFF.md` file is untracked. The
pre-existing `.idea` directory was not modified and is ignored.

## Important architectural decisions

### Package and tooling

- Python 3.11+ with a `src` layout and `pyproject.toml`.
- Pydantic v2 models use strict validation and reject unknown fields.
- Typer provides the CLI.
- Ruff is used for linting and formatting, mypy runs in strict mode, and pytest provides
  unit and integration tests.
- JSONL is used for manifests, claims, and predictions; JSON and CSV are used for run
  metadata and reports.

### Separation of retrieval and verification

The verifier never performs retrieval. A retriever returns `list[Evidence]`, and the
claim verifier receives the claim and already-retrieved evidence. This separation is
intentional so benchmark analysis can distinguish retrieval failures from reasoning
failures.

Small protocols define the real exchange points:

- `Retriever.retrieve(query, top_k)` plus lifecycle methods;
- `LLMClient.generate(...)` plus lifecycle cleanup.

The deterministic in-memory keyword retriever exists for offline tests only. It is not
presented as a scientific replacement for LightRAG.

### LightRAG boundary and version

- The optional dependency is pinned to `lightrag-hku==1.5.4`.
- All LightRAG-specific behavior is confined to
  `src/rag_claim_verification/retrieval/lightrag_adapter.py` and the ingestion service.
- The adapter uses the public 1.5.4 APIs that were checked during implementation:
  `LightRAG`, `QueryParam`, explicit `initialize_storages()`, `ainsert(...)`,
  `aquery_data(...)`, and `finalize_storages()`.
- The adapter checks the installed LightRAG version at runtime.
- Stable document IDs and declared file paths are passed during ingestion.
- Returned LightRAG file paths are mapped back to manifest document IDs. Unknown or
  ambiguous mappings remain `null`; IDs are not invented.
- LightRAG 1.5.4 does not document a per-chunk retrieval score in the public structured
  query response, so LightRAG evidence stores `retrieval_score: null`.

### Reproducible ingestion

Manifest validation is all-or-nothing. Duplicate IDs, missing files, empty files, and
invalid records are rejected rather than skipped.

The noisy manifest must contain every clean document with the same content and semantic
metadata. It may add documents and may differ in location-only metadata such as tags or
file paths where allowed by the implemented comparison.

Each LightRAG working directory receives `ragcv_ingestion_metadata.json`, including the
corpus identity, content-sensitive hashes, document IDs, LightRAG version, and index
configuration hash. A non-empty directory without this metadata, or an index associated
with different content/configuration/version, is rejected rather than silently reused.
Clean and noisy configurations use separate working directories under ignored
`indices/`.

### Structured model output

- Prompts are versioned files under `prompts/`, not embedded long strings.
- RAG mode instructs the model to use only supplied evidence.
- Baseline mode is explicitly marked as having no external evidence and cannot cite
  documents.
- Model output is parsed strictly. Unknown fields, invalid labels, duplicate JSON keys,
  non-finite constants, code fences, and citations not present in retrieved evidence are
  rejected.
- At most one bounded JSON repair request is made. If repair also fails, the raw output
  is retained and the prediction is marked as a parse error. No fallback label is
  fabricated.
- Verification temperature defaults to `0.0`; this improves comparability but does not
  guarantee provider-level determinism.

### Benchmark comparability and reporting

- Claims are processed in file order and conditions run sequentially.
- With `enforce_comparability: true`, incompatible model settings, RAG `top_k`, or
  relevant retriever settings are rejected.
- Every benchmark creates a unique run directory and does not overwrite an existing
  run.
- Configuration snapshots do not contain API-key values. Model endpoints are sanitized
  before metadata is persisted.
- Missing predictions count as incorrect and as false negatives for their gold class;
  they are never assigned an invented label.
- Retrieval metrics are emitted only when gold document IDs exist and retrieved evidence
  can be mapped to concrete document IDs. Otherwise the metrics contain an explicit
  unavailability reason.
- Failure categories in `failures.jsonl` are based on observable facts and avoid
  unsupported causal claims.

## Files created or modified

### Modified

- `README.md` — expanded from the initial two-line file into the complete English MVP
  documentation, including scope, architecture, installation, workflows, metrics,
  reproducibility, limitations, copyright guidance, and roadmap.

### Created: project foundation

- `pyproject.toml`
- `.gitignore`
- `.env.example`
- `runs/.gitkeep`

### Created: configuration and prompts

- `configs/clean_corpus.yaml`
- `configs/noisy_corpus.yaml`
- `configs/benchmark.yaml`
- `configs/baseline.yaml`
- `prompts/verification_system.txt`
- `prompts/verification_user.txt`

### Created: synthetic example data

- `data/README.md`
- `data/manifests/clean_documents.jsonl`
- `data/manifests/noisy_documents.jsonl`
- `data/ground_truth/claims.example.jsonl`
- `data/corpora/clean/synthetic_1991.txt`
- `data/corpora/clean/synthetic_1994.txt`
- `data/corpora/clean/synthetic_2010.txt`
- `data/corpora/noisy/synthetic_astronomy.txt`
- `data/corpora/noisy/synthetic_motorsport_noise.txt`

### Created: application package

- `src/rag_claim_verification/__init__.py`
- `src/rag_claim_verification/__main__.py`
- `src/rag_claim_verification/py.typed`
- `src/rag_claim_verification/cli.py`
- `src/rag_claim_verification/config.py`
- `src/rag_claim_verification/errors.py`
- `src/rag_claim_verification/logging_config.py`
- `src/rag_claim_verification/models/` — base, claim, document, evidence, prediction, and
  run models plus package exports.
- `src/rag_claim_verification/ingestion/` — manifest parsing, document loading, and
  ingestion orchestration.
- `src/rag_claim_verification/retrieval/` — retriever protocol, deterministic in-memory
  retriever, and LightRAG adapter.
- `src/rag_claim_verification/llm/` — LLM protocol, OpenAI-compatible HTTP client, and
  strict structured-output parsing.
- `src/rag_claim_verification/verification/` — prompt builder, RAG verifier, and baseline
  verifier.
- `src/rag_claim_verification/evaluation/` — benchmark runner, classification metrics,
  retrieval metrics, reporting, and stored-run re-evaluation.
- `src/rag_claim_verification/utils/` — atomic file helpers and hashing.

### Created: tests

- `tests/conftest.py`
- `tests/unit/test_models.py`
- `tests/unit/test_manifest.py`
- `tests/unit/test_hashing.py`
- `tests/unit/test_config.py`
- `tests/unit/test_prompt_and_output.py`
- `tests/unit/test_verifier.py`
- `tests/unit/test_in_memory_retriever.py`
- `tests/unit/test_lightrag_adapter.py`
- `tests/unit/test_ingestion_service.py`
- `tests/unit/test_openai_compatible.py`
- `tests/unit/test_classification_metrics.py`
- `tests/unit/test_retrieval_metrics.py`
- `tests/integration/test_benchmark_workflow.py`
- `tests/integration/test_lightrag_external.py`

Package `__init__.py` files also exist within each application subpackage.

## Relevant commands

### Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Install the pinned real LightRAG integration when it is needed:

```powershell
python -m pip install -e ".[dev,lightrag]"
Copy-Item .env.example .env
```

Do not commit `.env`. Configure the verification LLM, LightRAG LLM, and embedding
endpoint variables listed in `.env.example`. The checked-in YAML defaults point to
OpenAI-compatible endpoints, but model names, URLs, keys, and local paths are not
hardcoded in application code.

### Validation and execution

```powershell
rag-claim-verification validate-config --config configs/benchmark.yaml

rag-claim-verification validate-corpus `
  --manifest data/manifests/clean_documents.jsonl

rag-claim-verification validate-corpus `
  --manifest data/manifests/noisy_documents.jsonl `
  --clean-manifest data/manifests/clean_documents.jsonl

rag-claim-verification ingest --config configs/clean_corpus.yaml
rag-claim-verification ingest --config configs/noisy_corpus.yaml

rag-claim-verification verify `
  --config configs/clean_corpus.yaml `
  --claim "Michael Schumacher won the 1994 championship."

rag-claim-verification benchmark `
  --config configs/benchmark.yaml `
  --claims data/ground_truth/claims.example.jsonl

rag-claim-verification benchmark --config configs/baseline.yaml

rag-claim-verification evaluate --run-dir runs/<run-id>
```

CLI exit codes are `0` for clean completion, `1` when verification/benchmark completes
with failed predictions, and `2` for invalid input, unmet prerequisites, or operational
failure.

### Quality checks

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m pip check
```

The optional external test is skipped unless `RAGCV_RUN_EXTERNAL_TESTS=1` is set and the
LightRAG extra is installed.

## Tests and checks already executed

The final recorded local verification used Python 3.12.3 and produced:

- `python -m pytest`: **37 passed, 1 skipped**. The skipped test is the opt-in external
  LightRAG installed-version check. Standard tests made no external or paid API calls.
- `python -m ruff check .`: **passed** (`All checks passed`).
- `python -m ruff format --check .`: **passed** (`55 files already formatted`).
- `python -m mypy src`: **passed** (`no issues found in 38 source files`).
- `python -m pip check`: **passed** (`No broken requirements found`).
- The installed `rag-claim-verification --help` command ran successfully and displayed
  all six required commands.
- `rag-claim-verification validate-config --config configs/benchmark.yaml`: **passed**.
- Clean and noisy manifest validation passed, including the noisy-superset check.
- A repository secret-pattern scan found no matching credentials.
- `runs/` contained only `.gitkeep`; no generated research runs were left in the
  repository.

The offline integration test exercises five synthetic claims across baseline, clean RAG,
and noisy RAG using fake components. It creates all required run artifacts and verifies
that stored-run re-evaluation reproduces the metrics. Its synthetic perfect score is a
test assertion, not a research result.

## Known problems, limitations, and failed approaches

No failed implementation approach was recorded in this work. The following known
limitations remain and must not be mistaken for completed provider validation:

- A real LightRAG ingestion/query was not executed locally because it requires the
  optional dependency plus model and embedding providers. Only the adapter behavior,
  version boundary, structured response mapping, and ingestion orchestration were tested
  with controlled components.
- LightRAG 1.5.4's public structured query result does not document per-chunk retrieval
  scores. The adapter deliberately emits `null` rather than calculating or inventing a
  score.
- LightRAG document IDs depend on mapping returned file paths back to manifest paths.
  Ambiguous or unknown paths produce `document_id: null` and make retrieval metrics
  unavailable for the affected condition.
- LightRAG hybrid/local/global/mix querying may invoke its configured LLM for keyword
  processing in addition to embedding operations. Provider usage and cost are therefore
  not limited to final claim verification.
- The in-memory retriever scores whole documents by query-token coverage. It exists only
  for deterministic tests and offline demonstrations.
- The example corpus and claim labels are synthetic, small, and unsuitable for scientific
  conclusions.
- The baseline has no external evidence and therefore cannot provide document citations.
- The system accepts pre-formulated atomic claims only. It does not extract claims from
  articles or aggregate multiple claim verdicts.
- Confidence intervals, significance tests, evidence-conflict detection, temporal
  filtering, calibrated confidence, reranking, and alternative retrieval baselines are
  not implemented.
- The dependency set pins LightRAG exactly but does not currently include a generated
  full transitive lock file.

## Unresolved questions

These decisions were intentionally not invented during MVP implementation:

- Which hosted or local OpenAI-compatible verification model, LightRAG extraction model,
  and embedding model will be used for the actual experiment?
- Can provider-side model revisions be pinned, and how will model changes during the
  study be controlled?
- What licensed or otherwise permissible source collection will form the real clean
  Schumacher corpus?
- What inclusion/exclusion rules define the closed knowledge domain and its temporal
  boundaries in practice?
- How will noisy documents be sampled, categorized, and stratified so that the noise
  intervention is reproducible rather than anecdotal?
- How many claims per class are required, and what annotation protocol, independent
  review, and adjudication process will establish the ground truth?
- Should every evaluable claim have exhaustive `gold_document_ids`, or will retrieval
  metrics be reported on a documented subset?
- How many repeated model runs are needed, and which confidence intervals or statistical
  tests will be used for condition comparisons?
- Should a dependency lock file be added for the target operating system/environment?

## Concrete recommended next steps

1. **Review and commit this handoff file if it should be retained.** The MVP itself is
   already committed as `d87aafb`; only `CODEX_HANDOFF.md` is currently untracked. Run
   the quality checks once more in the target development environment before further
   code changes. Do not commit `.env`, `indices/`, generated `runs/`, or real copyrighted
   document text.
2. **Create a dedicated environment with LightRAG.** Install `.[dev,lightrag]`, copy
   `.env.example` to a local `.env`, and choose either hosted providers or local
   OpenAI-compatible endpoints. Keep the configured embedding model and dimension stable
   for an existing index.
3. **Run a provider-backed smoke test.** Use only the synthetic corpus first. Validate the
   config, ingest clean and noisy indexes, verify several individual claims, and inspect
   returned evidence paths/IDs. This is the main integration point not yet exercised
   against the actual LightRAG package and providers.
4. **Design and document the real corpus protocol.** Define source eligibility, date and
   domain boundaries, provenance/licensing records, text normalization, and clean/noisy
   construction before collecting data. Keep full text outside Git unless redistribution
   rights are established.
5. **Develop the ground-truth protocol.** Write annotation guidelines for atomicity and
   the three labels, use independent annotation and adjudication, and record gold
   document IDs where the evidence set is sufficiently known.
6. **Build and validate real manifests.** Ensure the noisy manifest is a content-preserving
   superset of clean, then run both `validate-corpus` commands before ingestion.
7. **Run a small pilot benchmark.** Check retrieval coverage, citation mapping, parse
   failures, latency, and cost before scaling. Manually inspect cases where gold evidence
   was missed and cases where it was retrieved but the verdict was wrong.
8. **Freeze the experiment configuration.** Record provider/model revisions where
   available, decide whether to add a lock file, archive input hashes, and avoid tuning on
   the final evaluation claims.
9. **Add statistical analysis after the pilot is stable.** Define repeated-run policy,
   confidence intervals, significance tests, and noise levels before interpreting
   clean-versus-noisy differences.
10. **Only then consider retrieval extensions.** BM25 is a useful transparent next
    baseline; dense retrieval, reranking, temporal filtering, and claim extraction should
    remain later extensions through the existing protocols.

For fuller user-facing instructions and known limitations, see `README.md`. For corpus
format, copyright, and local-data handling guidance, see `data/README.md`.
