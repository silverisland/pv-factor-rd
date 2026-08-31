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
- label scale 500, prediction clip `[0, 465]`, AdamW, MSE, gradient clipping,
  validation early stopping, and a capacity score using 465;
- development rows from September-December 2024, with the last five days of
  each month held out for validation;
- 2025 as the original test period.

The source script selected one filename prefix. The bundled implementation
generalizes only the data population: an empty prefix loads all station files,
adds stable station metadata, applies the same time boundaries to every station,
and pools the rows. The TabM numerical feature set and training kernel remain
unchanged.

The fixed baseline retains raw-power label scaling by 500 and a common scoring
capacity of 465. Before pooling stations with materially different capacities,
audit units and capacity distributions. Per-station capacity normalization is a
separate protected data protocol and must not be introduced inside an ordinary
factor comparison.

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

The default scenario is known-station future forecasting:

- every configured station contributes training rows;
- every station uses the same development and evaluation time boundaries;
- development train and validation must contain the same station set;
- confirmation and final evaluation reject unseen stations and missing trained
  stations unless a separate protocol explicitly changes the policy;
- station identity and row counts are fingerprinted in every run.

A full-station OOD holdout is a different experiment protocol and must not be
mixed into ordinary factor comparisons.

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
5. Fit transformations on pooled development-train rows only.
6. Keep station set, model, label scaling, split, clipping, and seed unchanged.
7. Keep confirmation and final-test periods out of factor exploration.
8. Return protected hashes plus pooled, per-station, station-macro, horizon, and
   regime metrics required by the result contract.

## Protected files

Hash all files declared in `config.json` before and after a run. A factor trial
may edit factor implementations and the experiment adapter, but not
`runtime/multi_station_tabm/model.py`. Static inspection must not start training.
