---
name: pv-factor-rd
description: Develop, implement, and evaluate expert-grounded time-series factors for a shared TabM trained on multiple PV stations over the next 0-4 hours. Use for pooled multi-station factor research, leakage audits, feature ablations, experiment evidence, and Claude Code execution in the private office environment; do not use for province aggregation, cross-station feature joins, model tuning, or image data.
---

# PV Factor R&D

Improve the data layer of the fixed TabM PV forecasting pipeline. Treat factor
definitions as hypotheses until private-environment experiments support them.

## Start

1. Read [references/project-contract.md](references/project-contract.md) before
   connecting to a training repository or changing an adapter.
2. Read [references/leakage-rules.md](references/leakage-rules.md) before
   implementing any factor that uses forecasts, interpolation, rolling windows,
   or future-known covariates.
3. Read
   [references/station-metadata-and-ghi.md](references/station-metadata-and-ghi.md)
   before using capacity, coordinates, `GHI_real`, solar, PV-physics, regime, or
   operational-score factors.
4. Run:

   ```bash
   python3 scripts/validate_catalog.py
   python3 scripts/inspect_project.py --config config.json
   ```

   If `config.json` is absent, copy `config.example.json` and fill only local
   paths and commands. Never put private data, station names, or credentials in
   the portable catalog.
5. For a training run, copy `runtime/config.example.yaml` to a private local
   config, fill its data and output paths, and call the bundled package. Never
   import training code from outside this Skill.
6. Keep `config.json.runtime_config`, task mode, and horizons aligned with the
   private runtime YAML. The default adapter is the real paired TabM adapter;
   never use `mock_adapter.py` as factor evidence.

## Research modes

- For source curation or new expert mechanisms, read
  [references/source-policy.md](references/source-policy.md).
- For factor design or catalog maintenance, read
  [references/factor-authoring.md](references/factor-authoring.md).
- For real-data trials, promotion, rejection, or model analysis, read
  [references/experiment-protocol.md](references/experiment-protocol.md).
- Before changing or running the training/evaluation runtime, read
  [references/training-pipeline.md](references/training-pipeline.md).

## Non-negotiable invariants

- The prediction target is the next 0-4 hours at 15-minute resolution unless
  the configured project contract explicitly narrows the evaluated horizons.
- Use structured time-series inputs only. Do not add sky-camera, satellite-image,
  or learned visual-embedding dependencies.
- Do not modify the TabM architecture, loss, optimizer, split policy, metric,
  preprocessing, target construction, or clipping while judging a factor.
- Import TabM training only from `runtime/multi_station_tabm/`; external
  project-folder imports are forbidden.
- Train one shared TabM on pooled rows from the configured station files. Keep
  `station_id` as metadata for identity, coverage checks, and metrics; do not use
  it as a model feature in the fixed baseline.
- Every row uses only its own station's power history and issued NWP. Do not
  introduce province rows, cross-station joins, capacity-weighted aggregation,
  or neighbor telemetry.
- Join capacity and coordinates only through the canonical station metadata
  contract. Require `GHI_real` to align element-for-element with `Power`; never
  use an observed GHI value after the forecast origin.
- Capacity-normalized features are allowed factor candidates. Changing the
  target to a generation coefficient is a separate protected protocol, not a
  factor-library experiment.
- Bind every run to the hashes of the effective model and evaluation protocol.
- Keep data loading, preprocessing, model initialization, training, evaluation,
  and metric computation in their existing separate modules. Do not bypass
  manifests or merge these layers inside an experiment adapter.
- Fit imputers, scalers, climatologies, thresholds, and feature selectors only
  on the permitted training partition.
- A forecast value is usable only when its issue time is no later than the
  forecast origin. A future observation is never a future-known covariate.
- Keep exploration/confirmation data separate. The final test remains sealed
  until a human explicitly authorizes one frozen-library evaluation.
- Preserve rejected and inconclusive experiments; do not repeatedly rediscover
  failed factors.

## Factor R&D loop

1. Diagnose one error slice: horizon, season, weather regime, ramp,
   low sun, clipping, curtailment, or data-quality state.
2. Retrieve relevant mechanisms from `knowledge/mechanisms.json` and existing
   factors from `factor_library/factors.json`.
3. State one falsifiable hypothesis, expected horizon effect, assumptions, and
   failure modes. Prefer one mechanism or feature family per experiment.
4. Implement the smallest factor change in the implementation file named by the
   factor record and bind its ID in
   `factor_library/implementations/registry.py`. Add a synthetic causal invariant
   test before training. An unbound catalog record is not executable.
5. Create a request with `scripts/create_experiment.py`. Run it through the
   configured `tabm_factor_adapter.py`; it must retrain paired baseline and
   candidate TabM models for identical seeds and rows. Do not call the final
   test by default.
6. Analyze paired pooled and per-station metrics. Overall RMSE alone is
   insufficient: inspect station macro/worst-site, horizon, month, regime, ramp,
   coverage, and seed stability.
7. Record the result with `scripts/record_result.py`. Promote evidence only when
   [references/experiment-protocol.md](references/experiment-protocol.md) allows it.
8. Stop when the experiment budget is reached, protected hashes change, leakage
   checks fail, or no supported hypothesis remains.

The default execution model is one Claude Code agent using deterministic local
scripts. Do not introduce multi-agent scheduling or bandit control until the
registry contains enough trustworthy experiments to justify it.
