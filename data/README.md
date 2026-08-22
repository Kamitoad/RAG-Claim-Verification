# Data layout and preparation

All files currently tracked under `data/corpora/` are short, hand-written synthetic fixtures. They exist only for tests and CLI demonstrations. Their labels and later model outputs are not scientific results and must not be used as an evaluation dataset.

## Local research documents

Store licensed or otherwise permissible research documents outside Git, or under `data/corpora/local/` (ignored by default). The MVP loader accepts UTF-8 text files. Add one JSON object per line to a manifest. Relative `file_path` values are resolved from the manifest directory.

Required document fields are:

- `document_id`: stable and unique inside the manifest;
- `title`;
- `source`;
- `file_path`.

Optional fields are `publication_date`, `event_date`, `topic`, `language`, and `corpus_tags`. Keep publication and event dates distinct. Dates use ISO `YYYY-MM-DD` format.

Example:

```json
{"document_id":"doc_001","title":"Example title","source":"Example source","publication_date":"2024-01-01","event_date":"1994-11-13","topic":"1994 championship","language":"en","file_path":"../corpora/local/doc_001.txt","corpus_tags":["clean","schumacher"]}
```

Run `rag-claim-verification validate-corpus --manifest <manifest.jsonl>` before ingestion. No invalid record is skipped.

## Ground-truth claims

Claim files are JSONL. Benchmark records require `claim_id`, `claim`, and `gold_label`; `gold_document_ids` and `notes` are optional. Allowed labels are `SUPPORTED`, `REFUTED`, and `NOT_ENOUGH_EVIDENCE`.

Gold document IDs should list the documents expected to contain decisive evidence. Retrieval metrics are omitted when these IDs or reliable retrieved document IDs are unavailable.

## Copyright and provenance

Do not commit scraped full text merely because it is technically accessible. Record the source, license or permission basis, acquisition date, and any transformations in research documentation. Manifests contain metadata, not permission to redistribute the referenced content.
