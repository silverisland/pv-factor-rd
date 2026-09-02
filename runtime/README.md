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
- `GHI_real` is read and validated only when a selected factor needs historical
  irradiance; the empty-factor baseline reads the same numerical columns as
  `pv_tabm_baseline`;
- `station_info.csv` can attach capacity, longitude, and latitude through a
  canonical station-ID join; all stations use `Asia/Shanghai`;
- station identity remains metadata and is not a TabM feature;
- parquet input is processed one station file at a time: required columns and
  metadata are validated, all requested horizons and selected factors are
  constructed immediately, and only the resulting numerical feature frames
  are pooled; raw array columns are not concatenated across station files;
- `split.train_stations` optionally restricts the training population;
  `null` means all discovered stations;
- `split.validation_stations` and `split.test_stations` independently control
  early stopping and final scoring;
- train, validation, and test rows use the three explicit inclusive ranges in
  `split` and filter the forecast-origin `timestamp`, exactly like the reference;
- the 96 power lags and target are divided by each row's stable station
  capacity; TabM predicts this ratio directly, clips it to `[0, 1.2]`, and
  restores physical power with the same capacity before scoring;
- `test_demo.py` reports final-test metrics, not validation metrics;

The runtime is separated into auditable stages: `data.py`, `features.py`,
`splits.py`, `preprocessing.py`, `model.py`, `trainer.py`, `evaluator.py`, and
`metrics.py`. `api.py` only orchestrates these stages. Completed runs include
station-set, row, input, preprocessing, model-weight, and environment hashes.
Treat a run directory without `run_manifest.json` as incomplete.

Validation, confirmation, and final metrics describe only configured test stations.
The reference primary score follows daily physical-power RMSE, monthly mean,
then an equal mean across months present in the interval. Horizon, day, month,
and regime slices remain separate so aggregate gains do
not hide a test-station temporal regression.

Set `features.prediction_mode: curve` to train the same scalar model separately
for horizons 1 through 16. This changes target indexing, not model architecture.

The runtime also accepts catalog factor IDs through `--factor`. The top-level
`test_demo.py --factor ...` runs an immediate paired validation smoke test;
evidence-producing comparisons should use the request-driven paired adapter so
protected hashes and experiment stages are checked automatically.

## Office use

Configure each partition explicitly:

```yaml
split:
  train_start: 2024-01-01 00:00:00
  train_end: 2024-10-31 23:59:59
  validation_start: 2024-11-01 00:00:00
  validation_end: 2024-11-30 23:59:59
  test_start: 2025-01-01 00:00:00
  test_end: 2025-12-31 23:59:59
  train_stations: [station_a, station_b, station_target]
  validation_stations: [station_target]
  test_stations: [station_target]
```

To exclude test stations from the training partition, keep the lists disjoint.
They can still drive early stopping without contributing gradient-training rows:

```yaml
split:
  train_stations: [station_a, station_b]
  validation_stations: [station_c, station_d]
  test_stations: [station_c, station_d]
```

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
