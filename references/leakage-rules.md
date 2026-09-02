# Time availability and leakage rules

## Canonical timestamps

Every value has explicit time semantics:

- `event_time`: physical time described by the value;
- `available_time`: earliest time the forecasting system can consume it;
- `issue_time`: publication time for a forecast product;
- `forecast_origin`: time at which the PV forecast is produced;
- `target_time`: `forecast_origin + horizon`.

A feature is legal only when every dependency satisfies
`available_time <= forecast_origin`. Future NWP is legal only when
`issue_time <= forecast_origin`. A future measurement is never legal.

## Mandatory checks

- Rolling and lagged observations are right-closed at the forecast origin.
- Interpolation does not use a future endpoint. Backfill across the origin is
  prohibited.
- Scalers, imputers, climatologies, quantiles, regime classifiers, clipping
  thresholds, capacity estimates, feature selection, and correlation pruning
  are fitted on training data only.
- A previous-day feature uses the value that would actually have arrived by the
  current origin, not a retrospectively repaired series unless that repair is
  available operationally.
- Prefer retaining NWP issue time and forecast time. When issue time is present,
  the adapter verifies every row satisfies `issue_time <= forecast_origin`.
  When it is absent, the result must report `contract_assumed`, cannot claim a
  fully verified leakage audit, and the factor cannot be promoted beyond
  implementation from that experiment alone.
- Each station's capacity and configuration use the version effective at the
  event time. No other station's telemetry is in scope.
- Preserve the protected baseline split during factor comparison: explicit
  inclusive ranges are applied to forecast-origin `timestamp`, followed by the
  configured station lists. Any stricter target-time split is a separate
  protocol experiment, not a factor ablation.
- Exploration cannot inspect confirmation or final-test labels.

## Suspicious patterns

Reject or investigate:

- negative shifts on observed columns;
- centered rolling windows;
- `bfill` or two-sided interpolation;
- normalization before temporal splitting;
- whole-dataset group statistics;
- weather forecast arrays with no issue-time lineage;
- factor thresholds chosen on final-test error;
- shuffled cross-validation for time series;
- row loss that differs between baseline and candidate without explanation.

The adapter result must report zero future-observation violations, zero
evaluation rows in training, and zero exploration/confirmation overlap.
