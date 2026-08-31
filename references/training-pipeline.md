# Stable training and evaluation pipeline

The runtime deliberately separates responsibilities so factor experiments can
change input columns without changing the model or evaluation protocol.

## Layer boundaries

1. `data.py` reads all configured station files, derives or validates stable
   station IDs, and sorts pooled rows by time and station.
2. `features.py` aligns each row's station-local target and issued-weather index,
   preserves feature order, creates station-aware deterministic `row_id` values,
   and appends only factor columns selected through the executable registry. It
   never joins values across stations.
3. `splits.py` assigns development-train, validation, confirmation, or sealed
   final-test periods. No model code decides split membership.
4. `preprocessing.py` checks finiteness, fits `QuantileTransformer` only on
   development-train rows, transforms validation rows, scales labels, and emits
   input/output/preprocessor fingerprints.
5. `model.py` contains only device/reproducibility setup, the fixed scalar TabM
   factory, and normalized ensemble inference.
6. `trainer.py` initializes AdamW, executes batches, computes un-clipped
   validation RMSE for early stopping, restores the best state, and saves model,
   optimizer, preprocessing, history, and artifact manifests.
7. `evaluator.py` loads frozen artifacts, rejects runtime-contract changes,
   applies inverse label scaling and final clipping, and emits aligned prediction
   rows plus an audit fingerprint.
8. `metrics.py` computes pooled, per-station, station-macro, worst-station,
   daily, monthly, horizon, and 0-1h/1-2h/2-4h grouped metrics. It does not train
   or select a model.
9. `api.py` is orchestration only. It writes the complete run manifest and never
   reads final-test labels during `train()`.
10. `factor_library/implementations/registry.py` is the executable boundary
    between catalog hypotheses and TabM. It rejects unknown, duplicate,
    conditional, unavailable, or unbound factors before training and records
    catalog/implementation hashes in every candidate run.

## Numerical parity rules

- Feature order is future GHI/temperature/wind speed/wind direction at the
  target index, 96 `Power` lags, hour, then month.
- The feature transformer is fit on training rows with legacy noise seed 0.
- TabM uses `LinearReLUEmbeddings`, no categorical columns, and `d_out=1`.
- Each ensemble member receives the repeated scalar target under MSE.
- AdamW, learning rate, weight decay, batch size, gradient clipping, and patience
  remain protected.
- Early stopping uses inverse-scaled, **unclipped** validation predictions, as in
  the supplied baseline. Reported confirmation/final predictions are clipped.
- Endpoint mode uses step 16. Curve mode repeats the same scalar pipeline for
  steps 1-16; `predict_hour` is the integer hour of the aligned target timestamp.

## Required artifacts

Each run must contain:

```text
config_resolved.yaml
run_manifest.json
training_summary_by_horizon.csv
validation_metrics_by_horizon.csv
validation_metrics_by_station.csv
validation_station_macro_summary.csv
validation_metrics_daily.csv
validation_metrics_monthly.csv
validation_monthly_score_summary.csv
validation_metrics_horizon_groups.csv
validation_predictions.parquet
models/model_hXX.pt
preprocessors/preprocessor_hXX.joblib
manifests/manifest_hXX.json
histories/history_hXX.csv
```

The manifests bind ordered feature names, row IDs, raw matrices, transformed
matrices, station identities/counts, learned preprocessor state, model weights,
model contract, training contract, seed state, runtime package/device
environment, and artifact identity through SHA-256 fingerprints. A run directory
is complete only after
`run_manifest.json` has been written; a directory without it is a failed or
interrupted run and must not enter comparisons.

## Paired factor experiment gate

Before comparing baseline and candidate, require equality of:

- station-file population, station IDs/counts, and target definition;
- horizon and row-ID fingerprints;
- train/validation split fingerprints;
- seed, fixed runtime contract hash, and runtime-environment fingerprint;
- target values and non-factor baseline columns;
- metric implementation and clipping policy.

Pooled sample-weighted metrics and station-macro metrics answer different
questions and must both be reported. A factor cannot be promoted when pooled
gain is caused by large stations while materially degrading the declared
worst-station or station-macro gate.

Only candidate factor columns and the resulting feature-order, prepared-data,
learned-preprocessor, and trained-weight fingerprints may differ. The learned
state and weight fingerprints are integrity checks within each run, not values
that baseline and candidate should share. Reject the experiment if any protected
equality or within-run integrity check fails.

## Real paired adapter

`adapters/tabm_factor_adapter.py` loads the configured station population once,
then retrains the baseline (`factor_ids=[]`) and candidate (requested factor IDs)
for each seed. Exploration uses the development validation predictions;
confirmation and final stages evaluate their separately declared time blocks.
It refuses comparisons when station, row, split, target, horizon, runtime,
environment, or evaluation fingerprints differ. A negative `delta_rmse` means
the candidate improves on the baseline.
