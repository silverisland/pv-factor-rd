# Multi-station shared TabM runtime

This is the only training runtime in the Skill. It is self-contained and does
not import code from another project.

The default `endpoint` mode preserves the scalar modeling kernel of the supplied
`code/tabm4pv.py`, while pooling rows from multiple station files into one
shared model:

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
- all stations use the same time split boundaries;
- label scale is 500 and reported predictions are clipped to `[0, 465]`;
- 2024-09 through 2024-12 is development data, with the last five days of each
  month used for validation;
- 2025 remains sealed as final test by default.

The runtime is separated into auditable stages: `data.py`, `features.py`,
`splits.py`, `preprocessing.py`, `model.py`, `trainer.py`, `evaluator.py`, and
`metrics.py`. `api.py` only orchestrates these stages. Completed runs include
station-set, row, input, preprocessing, model-weight, and environment hashes.
Treat a run directory without `run_manifest.json` as incomplete.

Pooled metrics are sample weighted. Per-station metrics, station-macro metrics,
and worst-station values are emitted separately so large stations cannot hide
small-station regressions.

Set `features.prediction_mode: curve` to train the same scalar model separately
for horizons 1 through 16. This changes target indexing, not model architecture.

The runtime also accepts catalog factor IDs through `--factor`. This trains one
candidate model only; evidence-producing comparisons should use the paired
adapter described in the Skill workflow so baseline and candidate invariants are
checked automatically.

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
