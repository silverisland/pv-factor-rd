#!/usr/bin/env python3
"""Append an aggregate experiment result to the immutable local ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalog_lib import load_config, read_json, sha256_json, utc_now


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    config, _, project_root = load_config(args.config)
    result_path = Path(args.result).expanduser().resolve()
    result = read_json(result_path)
    required = {
        "schema_version", "experiment_id", "status", "stage", "factor_ids",
        "protected_snapshot", "row_fingerprints", "leakage_audit", "metrics",
    }
    missing = sorted(required - set(result))
    if missing:
        raise SystemExit(f"Result missing required keys: {missing}")
    if result["schema_version"] != "1.0.0":
        raise SystemExit("Unsupported result schema_version")
    if not result["protected_snapshot"].get("unchanged"):
        raise SystemExit("Refusing to record result with changed protected files")
    if not result["row_fingerprints"].get("identical"):
        raise SystemExit("Refusing to record unpaired baseline/candidate rows")
    if not result["leakage_audit"].get("passed"):
        raise SystemExit("Refusing to record a result that failed the leakage audit")

    record = {
        "recorded_at": utc_now(),
        "result_sha256": sha256_json(result),
        "result_path": str(result_path),
        "result": result,
    }
    ledger = project_root / config.get("state_dir", "state") / "experiments.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if ledger.is_file():
        with ledger.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    existing_ids.add(json.loads(line)["result"]["experiment_id"])
    if result["experiment_id"] in existing_ids:
        raise SystemExit(f"Experiment already recorded: {result['experiment_id']}")
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Recorded {result['experiment_id']} in {ledger}")
    print("Factor status was not auto-promoted; apply the confirmation gates before catalog edits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
