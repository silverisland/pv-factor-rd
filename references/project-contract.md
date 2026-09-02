# Project contract

## Forecast object

The forecast object is a shared model for multiple PV stations. Each training
row belongs to exactly one station and contains that station's own power history,
issued future weather, and future power target. Rows from all configured station
files are pooled to train one scalar TabM per horizon.

This is not province forecasting or station aggregation. Never average or join
weather, power, or targets across stations. `station_id` is metadata used for row
identity, split coverage, audit, and per-station metrics; it is not a numerical or
categorical model input in the fixed baseline.

The supplied `code/tabm4pv.py` defines the scalar modeling kernel:

- 15-minute cadence and 96 historical `Power` points;
- issued future arrays for GHI, temperature, wind speed, and wind direction;
- `Power_predict` as the future target array;
- endpoint horizon step 16 (`TARGET_INDEX = 15`) by default;
- scalar TabM regression with `LinearReLUEmbeddings`;
- per-station capacity-ratio label, ratio clip `[0, 1.2]`, AdamW, MSE,
  gradient clipping, validation early stopping, and a capacity score using 465;
- a configurable train/test-station transfer split around the unchanged kernel.

The bundled implementation discovers `station=*.parquet`, reads the station
column, and pools permitted training rows. The TabM numerical feature set and
training kernel remain unchanged.

The fixed baseline divides the 96 power lags and target power by the stable
capacity of the row's station. TabM learns that generation coefficient directly,
clips it to `[0, 1.2]`, and multiplies by the same station capacity before
physical-power evaluation. The common score denominator remains 465; it is a
metric convention, not the model's normalization capacity.

The bundled implementation is `runtime/multi_station_tabm/`. It is the only
training runtime the Skill may import.

## Confirmed expert-factor inputs

The private environment also provides canonical station metadata in
`station_info.csv`: `plantid`, installed capacity `GCCAPACITY`, longitude
`LONGITUDE`, and latitude `LATITUDE`. All stations use `Asia/Shanghai`.
`GHI_real` is an observed historical array aligned element-for-element and at
the final timestamp with `Power`. The loader validates this contract before a
factor run. See `station-metadata-and-ghi.md` for aliases, overrides, and the
executable expert-factor families.

Static installed capacity is valid for feature normalization but does not by
itself authorize changing labels to a generation coefficient. That is a
separate protected experiment protocol.

## Station and time split policy

The scenario is configurable offline multi-station transfer:

- `evaluation.training_stations: null` admits every discovered station to the
  training role; an explicit list restricts that role;
- training-only stations contribute all rows under
  `source_station_time_policy=all_available`;
- a test station also named for training contributes only rows before the
  historical validation window; a held-out test station contributes none;
- the configured tail or fixed interval of test-station history is validation-
  only and drives early stopping and exploration metrics;
- confirmation and final-test metrics contain only configured test stations and
  the declared target-time interval;
- overlapping test-station training, validation, and evaluation boundaries use
  `target_timestamp` and have a maximum-horizon purge on both sides of
  validation;
- station identity, row counts, boundaries, and source-time policy are
  fingerprinted in every run.

Because training-only stations may contribute dates overlapping or later than
the test interval, report this as offline transfer rather than strict
chronological online backtesting. Test-station rows from the declared evaluation
interval never enter training or validation.

## Endpoint and curve modes

`endpoint` mode reproduces horizon step 16. `curve` mode trains 16 independent
instances of the same scalar TabM for steps 1 through 16. Curve mode may change
the target and matching future-weather index only; it must not change model
architecture, optimizer, preprocessing, split, station population, or metrics
during a factor comparison.

Never describe endpoint mode as a full four-hour curve.

## Protected experiment behavior

The experiment adapter receives one request JSON and returns aggregate result
JSON. It must:

1. Resolve the same configured station-file population for baseline and candidate.
2. Load exactly the requested factor IDs and implementation hashes.
3. Build baseline and candidate inputs from identical station-aware `row_id` values.
4. Align each station's future weather and `Power_predict` to the same horizon.
5. Fit transformations only on configured training rows plus permitted
   overlapping test-station history.
6. Keep station set, model, label scaling, split, clipping, and seed unchanged.
7. Keep confirmation and final-test periods out of factor exploration.
8. Return protected hashes plus pooled, per-station, station-macro, horizon, and
   regime metrics required by the result contract.

## Protected files

Hash all files declared in `config.json` before and after a run. A factor trial
may edit factor implementations and the experiment adapter, but not
`runtime/multi_station_tabm/model.py`. Static inspection must not start training.
