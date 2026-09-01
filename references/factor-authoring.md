# Factor authoring

## Authoring sequence

1. Start from a mechanism in `knowledge/mechanisms.json`.
2. Confirm required raw fields and their availability semantics.
3. Define one family with bounded, physically meaningful parameters.
4. State output shape, expected horizon effect, assumptions, failure modes, and
   minimum diagnostics.
5. Add the factor to `factor_library/factors.json` and run the catalog validator.
6. Implement only when the office data contract supplies the required fields.
7. Bind executable factors explicitly in
   `factor_library/implementations/registry.py`. Add a behavior test proving that
   row population, target, and baseline columns remain unchanged when the factor
   is appended. A catalog entry without a registry binding remains a hypothesis
   and cannot be submitted to TabM.

Avoid arbitrary Cartesian products. Windows should reflect sampling resolution,
the four-hour horizon, thermal inertia, or diurnal recurrence. Parameter variants
remain one family until experiments justify separate promotion.

## Required distinction

- `available`: all required fields are present with adequate time lineage;
- `conditional`: valuable but needs an optional field or issue-time metadata;
- `unavailable`: cannot currently be implemented;
- `proposed`: definition only;
- `implemented`: code and invariants exist;
- `validated`, `accepted`, `rejected`, `deprecated`: evidence states controlled
  by the experiment protocol.

Expected direction is a hypothesis, not a requirement. A factor can help one
horizon and hurt another; record both.

## Current executable boundary

The registry supports causal power dynamics, aligned `GHI_real` and power
states, solar/PV physics, future-weather coverage, target-prefix future-weather
change, and forecast clear-sky-index factors. The two future-weather-derived
families inherit the baseline contract that their arrays were issued no later
than `forecast_origin`; without an explicit NWP issue-time column this remains a
disclosed lineage assumption rather than a row-level verification.
