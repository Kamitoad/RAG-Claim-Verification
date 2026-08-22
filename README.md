# RAG Claim Verification

## Project overview

RAG Claim Verification is a modular research application for classifying atomic factual claims against a controlled document collection. The current research domain is the complete 2023 Formula One season. The implementation supports reproducible comparisons between a language-model baseline, retrieval over a clean corpus, and retrieval over the same corpus with added noise.

This project does **not** claim to detect fake news in general. It tests evidence-based verification inside a deliberately bounded knowledge domain.

## Research scope

The guiding question is:

> How reliably can a RAG system verify factual claims within a closed knowledge domain, and how does performance change when irrelevant documents are added to the knowledge base?

Every claim receives one of three labels:

- `SUPPORTED`: available evidence confirms every material part of the claim;
- `REFUTED`: available evidence contradicts at least one material part;
- `NOT_ENOUGH_EVIDENCE`: evidence is absent, ambiguous, unrelated, or insufficient.

The benchmark supports three controlled conditions:

1. an LLM-only baseline without retrieval;
2. RAG over a clean domain corpus;
3. RAG over that complete clean corpus plus additional irrelevant or partially relevant documents.

`enforce_comparability: true` rejects accidental model-setting differences and inconsistent RAG `top_k` values across conditions. Conditions share prompt files and generation parameters. The prompt carries an explicit mode marker because the baseline intentionally has no external evidence: RAG mode is restricted to retrieved evidence, while baseline mode uses parametric model knowledge and cannot cite documents.

## Current MVP

The implemented MVP provides:

- strict Pydantic models for documents, claims, evidence, predictions, and run metadata;
- JSONL manifest and ground-truth loading with line-specific validation errors;
- missing-file, empty-document, duplicate-ID, and clean-corpus-superset checks;
- a pinned LightRAG SDK adapter and reproducible ingestion metadata;
- a deterministic keyword retriever for tests and offline demonstrations;
- an OpenAI-compatible chat-completions client for hosted APIs and local servers;
- strict structured verdict parsing with at most one JSON repair request and no label fallback;
- CLI workflows for validation, ingestion, single-claim verification, benchmarking, and re-evaluation;
- classification, retrieval, reporting, and failure-analysis artifacts;
- tests that do not call paid APIs by default.

## Current local pilot

The active pilot uses three 2023 race weekends, 18 balanced claims, local Ollama inference,
FastEmbed embeddings, and Jolpica-F1 source data. Its exact source selection, transformation,
label construction, noise definition, and interpretation limits are documented in
[`docs/f1_2023_pilot_protocol.md`](docs/f1_2023_pilot_protocol.md).

The clean LightRAG index and a derived Noisy index have been qualified locally. The Noisy index
records the exact Clean metadata hash and preserved document hashes, then ingests only the four
additional documents. The final 18-claim, three-condition pilot completed all 54 planned cases
without technical or structured-output errors. Clean and Noisy RAG each classified 18/18 claims
correctly. Every Noisy case contained Noise evidence; 39 of 90 returned chunks came from Noise
documents, while no Noise document was cited. The baseline returned NEE for all 18 claims and is
therefore retained as a documented weak comparator.

The reproducibility record and cautious interpretation are documented in
[`docs/f1_2023_pilot_final_report.md`](docs/f1_2023_pilot_final_report.md).

## Reproducing the F1 pilot

Generated research data, LightRAG indexes, and run directories are intentionally ignored by Git.
The pilot can therefore be audited in two different ways:

1. Use a separately supplied archive of the final run to inspect its existing predictions,
   metrics, and reports. Running `evaluate` is optional and only verifies the stored hashes and
   regenerates the derived metrics without calling Ollama, LightRAG, or FastEmbed.
2. Rebuild the local Jolpica corpus and LightRAG indexes, then execute a new 54-case benchmark.
   This requires the documented external downloads and can produce a new, not necessarily
   byte-identical run.

The exact Windows/PowerShell commands, required artifacts, data-directory roles, integrity checks,
and reproducibility limits are in
[`docs/f1_2023_pilot_reproduction.md`](docs/f1_2023_pilot_reproduction.md).

## Explicit non-goals

The MVP does not implement a web UI, automatic web search, article collection, whole-article fake-news detection, mandatory claim extraction, late chunking, advanced reranking, multiple RAG frameworks, multi-model sweeps, end-user on-the-fly ingestion, or invented confidence scores. It also does not establish that the included synthetic examples are representative of the research domain.

## Architecture

```mermaid
flowchart LR
    A[Ground-truth claim] --> B{Condition}
    B -->|Baseline| D[Claim verifier]
    B -->|RAG| C[Retriever adapter]
    C --> E[Ranked evidence]
    E --> D
    A --> D
    D --> F[Strict prediction]
    F --> G[Metrics and error analysis]
    H[Document manifest] --> I[Validated ingestion]
    I --> C
```

Retrieval and verification are intentionally separate. This makes it possible to distinguish missing evidence from model interpretation errors. LightRAG-specific calls are confined to `retrieval/lightrag_adapter.py`; the verifier knows only the small retriever and LLM protocols.

## Repository structure

```text
.
├── configs/                  # Corpus and benchmark YAML configurations
├── data/
│   ├── corpora/              # Small tracked synthetic text fixtures only
│   ├── manifests/            # Document metadata as JSONL
│   └── ground_truth/         # Example claims as JSONL
├── prompts/                  # Versioned system and user prompts
├── src/rag_claim_verification/
│   ├── models/               # Strict domain and run models
│   ├── ingestion/            # Manifest parsing, loading, ingestion orchestration
│   ├── retrieval/            # Retriever protocol, keyword retriever, LightRAG adapter
│   ├── llm/                  # LLM protocol, HTTP client, structured parsing
│   ├── verification/         # Prompt rendering, RAG and baseline verification
│   ├── evaluation/           # Benchmark runner, metrics, reporting, re-evaluation
│   └── utils/                # Atomic files and hashing
├── tests/                    # Unit and offline integration tests
└── runs/                     # Generated runs; ignored except for .gitkeep
```

## Installation

Python 3.11 or newer is required. Python 3.12 is used for the local verification described below.

```bash
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the offline-testable core plus development tools:

```bash
python -m pip install -e ".[dev]"
```

Install the real LightRAG integration when ingestion or LightRAG retrieval is needed:

```bash
python -m pip install -e ".[dev,lightrag]"
```

Install the complete local Ollama/FastEmbed pilot stack with:

```bash
python -m pip install -e ".[dev,local]"
```

The optional dependency is pinned to `lightrag-hku==1.5.4`. The adapter follows the public SDK in the [LightRAG 1.5.4 source](https://github.com/HKUDS/LightRAG/tree/v1.5.4) and its [Core programming guide](https://github.com/HKUDS/LightRAG/blob/v1.5.4/docs/ProgramingWithCore.md), specifically explicit storage initialization/finalization, `ainsert`, and structured `aquery_data` retrieval.

## Environment configuration

Copy `.env.example` to `.env` and fill in keys locally. `.env` is ignored by Git.

```bash
cp .env.example .env
```

The main variables are:

| Variable | Purpose |
|---|---|
| `RAGCV_LLM_BASE_URL` | OpenAI-compatible endpoint used for verdict generation |
| `RAGCV_LLM_API_KEY` | Verification endpoint key |
| `RAGCV_LLM_MODEL` | Verification model name |
| `RAGCV_LIGHTRAG_LLM_*` | LightRAG extraction and query model settings |
| `RAGCV_EMBEDDING_*` | Optional remote LightRAG embedding endpoint, model, and dimension |

Configuration stores only the API-key environment-variable name, never the key. Model endpoints are stripped of credentials, query parameters, and fragments before run metadata is written. For a local endpoint such as LM Studio, set the base URL (commonly `http://localhost:1234/v1`) and set `api_key_required: false` in the relevant YAML object if the server does not require a key.

YAML supports `${NAME}` and `${NAME:-default}` interpolation. Paths in YAML are resolved relative to that YAML file. Paths in a document manifest are resolved relative to the manifest itself.

## Preparing a document corpus

The tracked files under `data/corpora/` are short, hand-written synthetic examples. They are not research data or research results.

For a real corpus, place permissible UTF-8 text files outside Git or under the ignored `data/corpora/local/` directory. Add one metadata object per line to a JSONL manifest:

```json
{"document_id":"f1_2023_r01_race","title":"2023 Bahrain Grand Prix race result","source":"Permitted source","event_date":"2023-03-05","topic":"2023 Bahrain Grand Prix","language":"en","file_path":"../corpora/local/f1_2023_r01_race.txt","corpus_tags":["clean","f1-2023"]}
```

Required fields are `document_id`, `title`, `source`, and `file_path`. Dates, topic, language, and tags are optional. `publication_date` and `event_date` are intentionally distinct.

Validate the manifest and every referenced file before ingestion:

```bash
rag-claim-verification validate-corpus \
  --manifest data/manifests/clean_documents.jsonl
```

Validate a noisy manifest as a content-preserving superset:

```bash
rag-claim-verification validate-corpus \
  --manifest data/manifests/noisy_documents.jsonl \
  --clean-manifest data/manifests/clean_documents.jsonl
```

Validation is all-or-nothing. Invalid records are never silently skipped.

## Preparing ground-truth claims

Claims use JSONL:

```json
{"claim_id":"r01_winner","claim":"Max Verstappen won the 2023 Bahrain Grand Prix race.","gold_label":"SUPPORTED","gold_document_ids":["f1_2023_r01_race"],"notes":"Direct factual claim."}
```

Benchmark records require a gold label. `gold_document_ids` is optional, but retrieval metrics can be computed only for claims that provide it. Claims should be atomic enough that a single three-way verdict is meaningful. The example file at `data/ground_truth/claims.example.jsonl` is synthetic and unsuitable for scientific evaluation.

## Validating configuration

```bash
rag-claim-verification validate-config --config configs/benchmark.yaml
```

This parses environment interpolation, validates cross-field constraints and referenced local inputs, and prints the resolved non-secret configuration. It does not contact providers or require an already-built index.

CLI exit code `0` means the requested workflow completed without failed predictions, `1` means a verify/benchmark workflow completed but recorded at least one failed prediction, and `2` indicates invalid input, missing prerequisites, or another operational error.

## Ingesting a corpus

Ingest the clean and noisy indexes separately:

```bash
rag-claim-verification ingest --config configs/clean_corpus.yaml
rag-claim-verification ingest --config configs/noisy_corpus.yaml
```

Ingestion performs complete manifest/file validation before contacting LightRAG. It passes stable document IDs and declared file paths to LightRAG, then writes `ragcv_ingestion_metadata.json` into the configured working directory. That file records the corpus ID, content-sensitive corpus hash, index-configuration hash, document IDs, content hashes, LightRAG version, timestamp, and LightRAG tracking ID. A working directory already associated with a different corpus, manifest content, index-relevant configuration, or LightRAG version is rejected instead of mixed or overwritten. A non-empty directory without RAGCV ingestion metadata is rejected as well.

The clean and noisy configurations use distinct index directories under `indices/`, which is ignored by Git. A corpus may declare `derived_from.corpus_config` when a controlled condition must inherit an immutable base index. In that workflow the application validates the complete base identity, copies it into a new directory, preserves the exact base-metadata hash, verifies unchanged base-document hashes, and sends only additional documents to LightRAG. A failed or pre-existing unrecognized target is never silently resumed or overwritten.

## Verifying a single claim

After ingesting the selected corpus:

```bash
rag-claim-verification verify \
  --config configs/f1_2023_pilot_clean.yaml \
  --claim "Max Verstappen won the 2023 Bahrain Grand Prix race."
```

The command prints one structured `Prediction`, including retrieved evidence, citations, raw model output, parsing status, and timing. It does not write a benchmark run.

## Running a benchmark

The example benchmark declares baseline, clean RAG, and noisy RAG conditions:

```bash
rag-claim-verification benchmark \
  --config configs/benchmark.yaml \
  --claims data/ground_truth/claims.example.jsonl
```

`--claims` is optional and overrides `claims_file`. Claims are processed in file order and conditions are run sequentially to avoid uncontrolled concurrency effects. Model or retrieval failures are recorded as failed predictions where possible, rather than converted into invented labels.

Run the baseline alone with:

```bash
rag-claim-verification benchmark --config configs/baseline.yaml
```

Run the deterministic synthetic smoke benchmark on Windows without credentials or
external services:

```powershell
.\.venv\Scripts\python.exe scripts\run_smoke_benchmark.py
```

The script starts a local deterministic Chat Completions fixture and then launches the
real `python -m rag_claim_verification benchmark` CLI with
`configs/smoke_benchmark.yaml`. The run exercises configuration loading, the production
HTTP client, structured-output repair, baseline verification, both in-memory retrieval
conditions, per-case checkpointing, metrics, and stored-run re-evaluation. Its synthetic
outputs are an engineering check only, never a research result.

## Run outputs and interpretation

Each benchmark creates a unique, never-overwritten directory:

```text
runs/<run-id>/
├── metadata.json
├── resolved_config.yaml
├── case_manifest.jsonl
├── predictions.jsonl
├── inputs/
│   ├── benchmark.yaml
│   ├── claims.jsonl
│   ├── hashes.json
│   ├── prompts/
│   ├── corpus_configs/
│   ├── manifests/
│   └── ingestion_metadata/
├── metrics.json
├── metrics.csv
├── confusion_matrix.csv
├── failures.jsonl
└── summary.md
```

- `metadata.json` records schema version, planned/completed counts, timing definitions,
  Git state, Python/platform/package versions, requested model settings, retriever,
  corpus and manifest hashes, prompt hashes, and success/failure counts.
- `resolved_config.yaml` snapshots the benchmark and corpus settings without API-key values.
- `case_manifest.jsonl` declares every expected claim-condition case before external calls.
- `inputs/` snapshots claims, prompts, source configurations, manifests, available
  ingestion metadata, and content-sensitive hashes.
- `predictions.jsonl` is atomically checkpointed after every case and preserves claim
  text, explicit case/retrieval/parse status, verdict, ground truth, ordered evidence,
  raw and repair outputs, provider response metadata, citations, technical errors, and
  stage timings.
- `metrics.json` is the authoritative structured metric output.
- the CSV files simplify analysis in spreadsheets and statistical tools.
- `failures.jsonl` assigns only observable diagnostic categories, such as missing gold evidence or an incorrect verdict despite retrieved gold evidence.
- `summary.md` describes measured values without extrapolating beyond them.

Regenerate derived artifacts without calling a model or retriever:

```bash
rag-claim-verification evaluate --run-dir runs/<run-id>
```

Re-evaluation verifies the persisted hashes of the case manifest, raw predictions,
resolved configuration, and claims snapshot before regenerating derived files.

## Evaluation metrics

Classification output includes:

- accuracy;
- precision, recall, F1, and support per class;
- unweighted Macro-F1 across all three labels;
- a confusion matrix with an explicit `NO_PREDICTION` column;
- parse-error and pipeline-error counts and rates.

Missing/failed predictions count as incorrect for accuracy and as false negatives for their gold class. They never receive a fabricated fallback label.

For retrieval, the application computes Evidence Recall and Hit Rate at 1, 3, and 5,
plus Mean Reciprocal Rank of the first relevant document. A condition receives retrieval
metrics only when gold IDs exist and every eligible retrieval can be mapped to concrete
document IDs. Otherwise `metrics.json` records coverage and an explicit unavailability
reason.

`metrics.json` keeps classification, retrieval, grounding proxies, structured-output
validity/repair, technical errors, and latency summaries in separate namespaces. Before
evaluation it requires exactly one prediction for every `case_manifest.jsonl` record;
missing or duplicate cases are rejected instead of producing metrics over a silent
subset.

Per-case timings use a monotonic clock. Retrieval time covers only the awaited retriever
call, initial and repair generation times include provider retries, total time starts at
retrieval for RAG or prompt rendering for the baseline, and benchmark duration covers
condition execution plus raw and derived artifact writing. The exact definitions are
also persisted in `metadata.json`.

No weighted aggregate is used as the sole headline metric. The MVP does not calculate confidence intervals or statistical significance tests.

## Reproducibility

The application protects reproducibility through:

- a `src` package and declared Python requirement;
- a fixed LightRAG version;
- versioned prompt files and prompt hashes;
- exact prompt, claim, configuration, and manifest snapshots inside each run;
- content-sensitive corpus hashes and claim/config file hashes;
- low default verification temperature (`0.0`);
- an optional recorded/requested model seed where the provider supports it;
- bounded retries and exactly one structured-output repair attempt;
- unique run IDs and refusal to overwrite run directories;
- resolved non-secret configuration snapshots;
- recorded runtime and package versions;
- provider-reported model revision, response ID, fingerprint, usage, and attempt count
  when the endpoint supplies them;
- deterministic offline components and tests.

Temperature zero reduces sampling variability but does not guarantee deterministic behavior across hosted model revisions, hardware, or provider implementations. For serious experiments, pin provider-side model revisions when the provider supports them and archive the exact raw inputs and predictions.

Run local quality checks with:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

Tests marked `external` are skipped by default. After installing the LightRAG extra, set `RAGCV_RUN_EXTERNAL_TESTS=1` to enable the pinned-version check. Standard tests never call an external or paid service.

## Data and copyright considerations

Do not commit copyrighted article full text merely because it is publicly reachable. Keep local documents under an ignored path or outside the repository and maintain independent provenance, licensing, acquisition-date, and transformation records. A manifest records technical metadata; it does not establish redistribution rights.

The tracked synthetic fixtures are deliberately short and clearly marked. They demonstrate mechanics only and must not be presented as measured research performance.

## Known limitations

- LightRAG 1.5.4's public `aquery_data` response exposes ordered chunks and citation file paths but does not document a per-chunk retrieval score. LightRAG evidence therefore stores `retrieval_score: null`; values are never simulated.
- LightRAG document IDs are recovered by mapping returned file paths to the ingestion manifest. Ambiguous or unknown paths disable retrieval metrics for the affected condition.
- Hybrid/local/global/mix LightRAG queries may invoke the configured LightRAG LLM for keyword processing in addition to embeddings used during indexing.
- The in-memory retriever ranks whole documents by query-token coverage. It exists for deterministic testing, not as a scientific substitute for LightRAG.
- The baseline has no external evidence or source attribution. Its output reflects parametric model knowledge and is explicitly marked as such.
- The verifier assumes pre-formulated atomic claims and does not aggregate verdicts across a full article.
- Evidence conflict detection, temporal filtering, calibrated uncertainty, and statistical comparison tooling are not part of the MVP.
- The local pilot has exercised real LightRAG ingestion and retrieval without paid credentials,
  but the small local extraction model produced a graph with mixed relationship quality and did
  not reproduce the exact same relations in an independent rebuild.

## Roadmap

Potential post-MVP extensions include BM25/dense/hybrid retrievers implementing the current protocol, reranking, temporal filters, late chunking, alternative RAG adapters, additional OpenAI-compatible or native model providers, local-model profiles, evidence-conflict detection, and statistical experiment comparison. Automatic claim extraction, a REST API, a web frontend, new knowledge domains, and whole-article aggregation remain explicit non-goals unless the project scope is changed.

These are extension points, not partially implemented subsystems. A new retriever implements `Retriever`; a new model provider implements `LLMClient`; new metrics remain independent evaluation functions; future claim extraction can feed the existing `Claim` model.
