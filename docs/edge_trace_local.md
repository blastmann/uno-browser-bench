# Local Edge trace extraction

This workflow turns the local Microsoft Edge Chromium `History` SQLite
database into a local-only, weakly labelled trace set. It is intended for
personalization experiments, not for publishing browsing history.

## What is retained

Each trace prefix contains only:

- a local sequential trace id;
- relative event positions;
- Chromium transition names such as `link`, `typed`, `reload`, and
  `form_submit`;
- coarse page categories shared with `analyze_edge_history.py`;
- the observed category of the next visit.

The extractor does not write URLs, titles, hostnames, query strings, paths,
timestamps, or URL-derived hashes. It opens the History database with
`mode=ro&immutable=1` and makes no network calls. The generated JSONL and
audit files belong under `work/`, which is ignored by Git.

## Run on Windows PowerShell

```powershell
$history = "$env:LOCALAPPDATA\Microsoft\Edge\User Data\Default\History"
python scripts/extract_edge_trace_local.py `
  --history $history `
  --output work/edge_trace_v3_local.jsonl `
  --audit-output work/edge_trace_v3_local_audit.json

python scripts/evaluate_edge_trace_baseline.py `
  --input work/edge_trace_v3_local.jsonl `
  --output work/edge_trace_v3_baseline.json
```

The split is chronological by session: 70% train, 15% development, 15% test.
The baseline predicts the next coarse category from the last category and
transition type.

## Interpretation

`observed_next_page_category` is a weak behavioral label. It is not a verified
semantic intent label: a visit to a search page does not prove what the user
wanted, and `other` is intentionally broad. Always inspect `non_other_top1`
and `macro_f1` alongside overall accuracy; an apparently high overall score
can be dominated by the `other` class.

For a semantic Browser Observation model, the next stage is to sample these
local prefixes and add human-confirmed intent labels plus a small set of
redacted observation fields. The local trace file should remain on the host.
