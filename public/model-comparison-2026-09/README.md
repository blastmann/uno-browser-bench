# Public model comparison export

This directory contains a privacy-reduced export of the Decider-2B, Jev, and NanoJev experiments.

Included:

- `aggregate_metrics.json`: aggregate benchmark metrics and paired comparisons.
- `per_example_metrics.jsonl`: anonymized per-example outcomes, confidence, gold probability, and latency.
- `report.md`: readable experiment report.
- `manifest.json`: export schema and privacy declarations.

Excluded intentionally:

- browser states and event traces;
- HTML, page titles, URLs, and website names;
- DOM candidate IDs, candidate text, model choices, and raw probability maps;
- local filesystem paths, usernames, API keys, and service request metadata.

Identifiers in the JSONL file are dataset-scoped truncated SHA-256 hashes. The per-example file is therefore suitable for aggregate/error-rate analysis, not for reconstructing the original browser sessions.

The Mind2Web experiment is a fixed-candidate research protocol and does not claim to reproduce the official candidate-rank evaluation. The Uno trace experiment is synthetic and should not be interpreted as a general browser-agent benchmark.
