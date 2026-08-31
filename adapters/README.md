# Experiment adapter boundary

`tabm_factor_adapter.py` is the evidence-producing adapter. It loads the private
runtime YAML named by `runtime_config`, selects executable factor IDs, trains
paired baseline and candidate TabM models for every requested seed, and emits
aggregate comparison metrics following `schemas/result.schema.json`.

`mock_adapter.py` remains only for orchestration tests. It emits synthetic
metrics that are never forecasting evidence and is not the default adapter.

The fixed TabM implementation is bundled at `runtime/multi_station_tabm/`; do
not import a sibling project. In the office environment, copy
`runtime/config.example.yaml` to the path configured by `runtime_config` and
fill the private data/output paths. The real adapter accepts:

```text
--config /absolute/path/to/config.json
--request /absolute/path/to/experiment.json
--output /absolute/path/to/result.json
```

The request supplies factor IDs, seeds, target semantics, protected hashes,
leakage checks, and required slices. Baseline and candidate inputs must have
identical row IDs, targets, splits, station sets, and non-factor data. Learned
feature state is fit on the training partition only; only aggregate metrics are
written to the result.

Keep `runtime/multi_station_tabm/model.py` protected. New executable factors are
implemented in `factor_library/implementations/` and explicitly bound in
`factor_library/implementations/registry.py`. Catalog entries without a registry
binding remain hypotheses and are rejected before training.
