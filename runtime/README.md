# Multi-station shared TabM runtime

This is the only training runtime in the Skill. It is self-contained and does
not import code from another project.

The default `endpoint` mode preserves the scalar modeling kernel and physical
target semantics of `pv_tabm_baseline`, while pooling rows from multiple station
files into one shared model:

- `data.parquet_root` and `data.parquet_glob: station=*.parquet` discover the
  same private station files as `pvreglab`;
- `station_id` comes from the configured parquet `station` column, never from
  the filename, and each parquet must contain exactly one station value;
- every row uses only its own station's 96 recent `Power` values and issued
  future GHI, temperature, wind speed, and wind direction;
- `GHI_real` is validated as an element-for-element historical array aligned
  with `Power`;
- `station_info.csv` can attach capacity, longitude, and latitude through a
  canonical station-ID join; all stations use `Asia/Shanghai`;
- station identity remains metadata and is not a TabM feature;
- parquet input is processed one station file at a time: required columns and
  metadata are validated, all requested horizons and selected factors are
  constructed immediately, and only the resulting numerical feature frames
  are pooled; raw array columns are not concatenated across station files;
- all non-target station dates and older target history train one shared model;
- only the configured target station's historical tail or explicit historical
  interval is used for validation;
- only that target station is scored in confirmation/final-test target-time
  ranges, with a four-hour purge around the historical validation window;
- the 96 power lags and target are divided by each row's stable station
  capacity; TabM predicts this ratio directly, clips it to `[0, 1.2]`, and
  restores physical power with the same capacity before scoring;
- non-target dates may overlap the target evaluation range under the explicit
  offline `all_available` source policy;
- 2025 remains the default sealed target-station final-test range.

The runtime is separated into auditable stages: `data.py`, `features.py`,
`splits.py`, `preprocessing.py`, `model.py`, `trainer.py`, `evaluator.py`, and
`metrics.py`. `api.py` only orchestrates these stages. Completed runs include
station-set, row, input, preprocessing, model-weight, and environment hashes.
Treat a run directory without `run_manifest.json` as incomplete.

Validation, confirmation, and final metrics describe only the target station.
The reference primary score follows daily physical-power RMSE, monthly mean,
then an equal mean across months present in the interval. Horizon, day, month,
and regime slices remain separate so aggregate gains do
not hide a target-station temporal regression.

Set `features.prediction_mode: curve` to train the same scalar model separately
for horizons 1 through 16. This changes target indexing, not model architecture.

The runtime also accepts catalog factor IDs through `--factor`. The top-level
`test_demo.py --factor ...` runs an immediate paired validation smoke test;
evidence-producing comparisons should use the request-driven paired adapter so
protected hashes and experiment stages are checked automatically.

## Office use

```bash
python3 -m pip install -r runtime/requirements.txt
cp runtime/config.example.yaml runtime/config.private.yaml
python3 -m runtime.multi_station_tabm.cli train \
  --config runtime/config.private.yaml --seed 0
```

For development and catalog/schema tests, install
`runtime/requirements-dev.txt`. The repository includes a `pyproject.toml` and
CI runs both dependency-light workflow tests and runtime integration tests.

For a direct candidate smoke run:

```bash
python3 -m runtime.multi_station_tabm.cli train \
  --config runtime/config.private.yaml --seed 0 \
  --factor factor.power.multiscale-ramp
```

For a first metadata/GHI family experiment, select one factor at a time, for
example `factor.power.capacity-ratio` or `factor.solar.position`. The registry
rejects a selected factor before training if its required metadata, weather
role, or aligned `GHI_real` input is absent.

For a valid paired experiment, use `scripts/create_experiment.py` followed by
`scripts/run_experiment.py`; the default `tabm_factor_adapter.py` retrains both
branches and writes deltas by horizon, station, month, and seed.

Confirmation is a separate command. Final test additionally requires the exact
confirmation phrase enforced by the CLI. Private configs, station identifiers,
data, checkpoints, predictions, and logs stay in the office environment.
