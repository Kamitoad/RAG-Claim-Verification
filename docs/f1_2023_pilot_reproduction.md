# Reproducing and auditing the F1 2023 pilot

This guide separates two different goals:

1. **Audit the existing final run.** A supplied run archive already contains the raw predictions
   and derived results. No model, retrieval, embedding, or source-data download is required.
2. **Execute a new experiment.** Rebuild the local source data and LightRAG indexes and run all 54
   claim-condition cases again. This requires external downloads and local Ollama inference.

The first path reproduces the evaluation of the recorded observations. The second path tests whether
the complete experimental workflow can be repeated, but it creates a new run and is not expected to
be byte-identical.

## Recorded final run

The primary empirical result is:

| Field | Recorded value |
|---|---|
| Run ID | `20260818T161551.959752Z-248fa8fc` |
| Experimental Git commit | `e0bc15ce71bda7dd952414dca18a436150e0c731` |
| Prompt version | `verification-v3-baseline-knowledge` |
| Local model profile | `ragcv-qwen3-4b-pilot:v1` |
| Planned/completed cases | 54/54 |
| Predictions SHA-256 | `612767345ed3c444f0f6a00a33188e99776d089af02559a3fea1331d734396cf` |

Its measured results and limitations are recorded in
[`f1_2023_pilot_final_report.md`](f1_2023_pilot_final_report.md). The run itself is generated data
under `runs/` and is intentionally not stored in Git.

## Path A: inspect or verify the supplied final run

### What the sender supplies

From the repository root, the person who owns the final run creates an archive outside the
repository:

```powershell
Compress-Archive `
  -LiteralPath .\runs\20260818T161551.959752Z-248fa8fc `
  -DestinationPath ..\ragcv-final-run-20260818.zip

Get-FileHash `
  -Algorithm SHA256 `
  -LiteralPath ..\ragcv-final-run-20260818.zip
```

Send both the ZIP and the printed ZIP SHA-256 through the chosen transfer channel. The archive is
not added to Git.

### What the recipient does

Clone the documented branch and install the dependencies needed by the network-free evaluator.
The installation itself may download Python packages if they are not already cached:

```powershell
git clone --branch feature/f1-2023-baseline-diagnostic --single-branch `
  https://github.com/Kamitoad/RAG-Claim-Verification.git
cd RAG-Claim-Verification

$env:UV_PROJECT_ENVIRONMENT = ".venv-pilot"
uv sync --extra dev --python 3.12
```

First compare the following output with the ZIP hash supplied by the sender:

```powershell
Get-FileHash `
  -Algorithm SHA256 `
  -LiteralPath ..\ragcv-final-run-20260818.zip
```

Then extract the archive:

```powershell
Expand-Archive `
  -LiteralPath ..\ragcv-final-run-20260818.zip `
  -DestinationPath .\runs
```

The existing results can now be read directly in:

- `runs/20260818T161551.959752Z-248fa8fc/summary.md` for a concise summary;
- `runs/20260818T161551.959752Z-248fa8fc/metrics.json` for authoritative structured metrics;
- `runs/20260818T161551.959752Z-248fa8fc/predictions.jsonl` for all raw per-case observations;
- `runs/20260818T161551.959752Z-248fa8fc/metadata.json` for execution provenance.

No further command is required merely to inspect those files.

### Optional offline re-evaluation

The following command does **not** repeat the benchmark. It validates the persisted input and raw
prediction hashes, checks case completeness, and regenerates metrics and reports from the stored
predictions:

```powershell
.\.venv-pilot\Scripts\python.exe -m rag_claim_verification evaluate `
  --run-dir runs\20260818T161551.959752Z-248fa8fc
```

It does not call Ollama, LightRAG, FastEmbed, or Jolpica. This is the recommended short verification
for a project partner or assessor who wants to confirm that the published metrics follow from the
archived observations.

## Path B: execute the complete pilot again

Use a fresh clone or a workspace without existing pilot data and index targets. The preparation and
ingestion workflows deliberately refuse to overwrite existing generated artifacts.

### Prerequisites

- Windows PowerShell;
- Git;
- Python 3.12 available through `uv`;
- a running local Ollama installation;
- enough free disk space for the Python environment, Qwen model, embedding model, and LightRAG
  indexes;
- network access for dependency, model, embedding, and Jolpica downloads.

No paid model API is used by the documented configuration. Network downloads may still be subject
to the respective providers' terms and local network costs.

### 1. Clone and install the local stack

```powershell
git clone --branch feature/f1-2023-baseline-diagnostic --single-branch `
  https://github.com/Kamitoad/RAG-Claim-Verification.git
cd RAG-Claim-Verification

$env:UV_PROJECT_ENVIRONMENT = ".venv-pilot"
uv sync --extra dev --extra local --python 3.12
```

The recorded run was executed from commit `e0bc15ce71bda7dd952414dca18a436150e0c731`.
Later commits on this branch document the outcome; the run metadata remains the authority for the
experimental code state.

### 2. Prepare the local Ollama model profile

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama create ragcv-qwen3-4b-pilot:v1 `
  -f configs\ollama\qwen3_4b_pilot.Modelfile
```

The first LightRAG ingestion also downloads the configured FastEmbed model,
`jinaai/jina-embeddings-v2-small-en`.

### 3. Download and build the pilot data

```powershell
.\.venv-pilot\Scripts\python.exe scripts\prepare_f1_2023_pilot.py all
```

This explicit command contacts the seven Jolpica endpoints pinned in
`data/sources/f1_2023_pilot_jolpica.yaml`. It validates their season, round, race identity, and
payload structure before generating corpus documents. It also verifies that the deterministically
derived 18 claims still match the tracked ground-truth file.

Validate the generated inputs:

```powershell
.\.venv-pilot\Scripts\python.exe -m rag_claim_verification validate-corpus `
  --manifest data\manifests\local\f1_2023_pilot_v3_clean.jsonl

.\.venv-pilot\Scripts\python.exe -m rag_claim_verification validate-corpus `
  --manifest data\manifests\local\f1_2023_pilot_v3_noisy.jsonl `
  --clean-manifest data\manifests\local\f1_2023_pilot_v3_clean.jsonl

.\.venv-pilot\Scripts\python.exe -m rag_claim_verification validate-config `
  --config configs\f1_2023_pilot_benchmark.yaml
```

### 4. Build the two LightRAG indexes

Clean must be ingested before Noisy because the Noisy index is derived from the completed Clean
index and adds only the four declared Noise documents:

```powershell
.\.venv-pilot\Scripts\python.exe -m rag_claim_verification ingest `
  --config configs\f1_2023_pilot_clean.yaml

.\.venv-pilot\Scripts\python.exe -m rag_claim_verification ingest `
  --config configs\f1_2023_pilot_noisy.yaml
```

### 5. Run the 18-claim, three-condition benchmark

```powershell
.\.venv-pilot\Scripts\python.exe -m rag_claim_verification benchmark `
  --config configs\f1_2023_pilot_benchmark.yaml
```

The CLI prints the newly created `runs/<run-id>` path. It never overwrites the recorded final run.
Compare the new run with the expected results in the final report, but retain it as a separate
observation rather than replacing or merging it with the recorded run.

## What each ignored data directory means

| Path | Meaning | Needed for offline `evaluate`? | Needed for a fresh full run? |
|---|---|---:|---:|
| `data/raw/local/f1_2023_pilot/` | Untouched Jolpica JSON responses and a download manifest with source URLs and hashes | No | Generated by preparation |
| `data/corpora/local/f1_2023_pilot_v3/` | Generated UTF-8 documents that LightRAG ingests | No | Generated by preparation |
| `data/manifests/local/f1_2023_pilot_v3_*.jsonl` | Stable document IDs, metadata, and paths connecting corpus files to retrieval results | No | Generated by preparation |
| `data/provenance/local/f1_2023_pilot_v3.json` | Build receipt containing source, transformer, license, and generated-file hashes | No | Generated by preparation |
| `data/ground_truth/f1_2023_pilot.jsonl` | Tracked claims with reference labels and relevant document IDs used only by evaluation | Included in Git and snapshotted in the run | Included in Git |
| `indices/f1_2023_pilot_*` | Generated LightRAG graph, vector, document, and cache state | No | Generated by ingestion |
| `runs/<run-id>/` | Immutable raw predictions, input snapshots, metadata, and derived reports | This is the input | Generated by benchmark |

The ground-truth labels are not inserted into model prompts. The benchmark stores them alongside
predictions so the offline evaluator can compare the model verdicts with the predeclared reference
answers afterward.

## Exact data snapshot versus fresh source download

Running the preparation script is the simplest reproducibility route and recreates the same logical
v3 directory structure. Because an external API may change, a later download is not guaranteed to
produce byte-identical raw data.

If an assessor requires the exact source snapshot in addition to the exact run, transfer a second
archive preserving these paths:

- `data/raw/local/f1_2023_pilot/`;
- `data/corpora/local/f1_2023_pilot_v3/`;
- `data/manifests/local/f1_2023_pilot_v3_clean.jsonl`;
- `data/manifests/local/f1_2023_pilot_v3_noisy.jsonl`;
- `data/provenance/local/f1_2023_pilot_v3.json`.

Also supply that archive's SHA-256. Do not include obsolete local pilot versions. Sharing the
LightRAG indexes is normally unnecessary: they are larger, runtime-specific generated state and can
be rebuilt from the exact data snapshot. The Ollama and FastEmbed model files should be obtained
from their original distribution channels rather than redistributed with the project.

The source plan records Jolpica-F1 and the CC BY-NC-SA 4.0 terms for this non-commercial university
project. The manifest and provenance records support attribution and integrity; they do not by
themselves grant rights beyond the recorded license.

## Reproducibility limits

- Offline `evaluate` is deterministic for an unchanged archived run and does not invoke external
  components.
- A new benchmark is a new observation. Temperature 0 and seed 17 reduce variability but do not
  guarantee identical Ollama or LightRAG output across machines and software/hardware states.
- A future Jolpica response or a freshly downloaded model can differ from the recorded snapshot.
- The final results describe 18 deliberately constructed claims from three race weekends, not the
  complete Formula One season and not general fake-news detection.
