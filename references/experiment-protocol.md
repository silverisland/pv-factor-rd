# Fixed-model factor experiment protocol

## Stage 0: endpoint parity

Reproduce the supplied horizon-step-16 baseline before factor research. Verify
feature order, row IDs, split membership, preprocessing, label scale, clipping,
seed, and validation RMSE. Record protected hashes. Do not use final-test labels
for parity debugging unless a separately authorized one-time audit is intended.

## Stage 1: curve parity

When the product requires a full four-hour curve, repeat the same scalar TabM
for horizon steps 1 through 16. Only target index, matching future-weather index,
and horizon-derived time may change. Establish this no-factor curve baseline
before evaluating factors.

## Stage 2: cheap validity

- schema and implementation tests;
- shape, finite-value, coverage, unit, and range checks;
- forecast-origin availability audit;
- train-only redundancy diagnostics;
- a small temporal-slice smoke run.

## Stage 3: exploration

Test one factor family or conceptual mechanism. Compare paired baseline and
candidate runs on identical target-station validation rows and seeds. Report:

- overall MAE/RMSE and capacity-normalized forms;
- target-station overall metrics;
- each 15-minute horizon and 0-1h, 1-2h, 2-4h groups;
- month/season and daylight/low-sun slices;
- stable/ramp and available operational states;
- missingness, factor coverage, runtime, and feature count;
- configured-seed mean and dispersion.

## Stage 4: confirmation

Use a time block never inspected during exploration. Require consistent
direction across configured seeds and the predeclared important horizon groups.
Fix promotion thresholds before reading confirmation results.

## Final test

Freeze the factor library, code hashes, scalar model, protocol, and seeds. Run
once only after explicit human confirmation. Never continue factor search from
the same final-test labels.

## Decision states

- `accepted`: confirmation and stability gates pass;
- `validated`: useful evidence exists but promotion gates are incomplete;
- `rejected`: the hypothesis is falsified or material harm exceeds limits;
- `inconclusive`: implementation, coverage, variance, or lineage prevents a
  conclusion.

Preserve all outcomes. Lower feature correlation or better physical fit alone
is not success; downstream fixed-model prediction evidence is required.

## Leakage-audit interpretation

Distinguish dynamically verified checks from input-contract assumptions. A
configured NWP issue-time column is checked row by row against the forecast
origin. Without that column, an experiment may run under the issued-NWP input
contract, but its result must set `fully_verified: false` and disclose the
assumption. Registry admission means the feature formula was reviewed for
causal structure; it does not independently prove upstream data provenance.
