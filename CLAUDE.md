# PV Factor R&D for Claude Code

Read `SKILL.md` and follow it as the authoritative workflow. The model backend
may be DeepSeek V4; do not depend on provider-specific APIs.

This package is intended to be copied into the private office environment. Raw
rows, arrays, timestamps, station identifiers, predictions, private paths,
source code, configuration, and logs stay inside that environment. Only the
portable knowledge catalog and explicitly approved aggregate evidence may leave.

## Bootstrap

1. Copy `config.example.json` to `config.json` and fill local paths.
2. Install `runtime/requirements.txt` in the office Python environment, copy
   `runtime/config.example.yaml` to `runtime/config.private.yaml`, and fill only
   private data/output paths. Keep its prediction mode and horizons identical to
   `config.json`.
3. Use the bundled real adapter `adapters/tabm_factor_adapter.py`. Do not switch
   to `mock_adapter.py` for evidence-producing experiments.
4. Run:

   ```bash
   python3 scripts/validate_catalog.py
   python3 scripts/inspect_project.py --config config.json
   python3 -m unittest discover -s tests -v
   ```

5. Establish an unchanged baseline before proposing a factor experiment.

Create and run a real paired factor experiment with:

```bash
python3 scripts/create_experiment.py --config config.json \
  --factor factor.power.multiscale-ramp --stage exploration
python3 scripts/run_experiment.py --config config.json \
  --request state/requests/<experiment-id>.json
python3 scripts/record_result.py --config config.json \
  --result state/results/<experiment-id>.json
```

## Working roles

Perform these as distinct reasoning passes even though one agent executes them:

1. Domain researcher: retrieve mechanisms and sources.
2. Leakage reviewer: challenge availability and split assumptions.
3. Experiment designer: specify one minimal discriminating trial.
4. Implementer: change only allowed feature/adapter files and add tests.
5. Result analyst: judge paired evidence and update the registry.

The implementer does not promote its own factor without the result-analysis
pass. Do not modify protected TabM or evaluation files. Do not run a sealed
final test without explicit human approval.
