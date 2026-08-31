#!/usr/bin/env python3
"""Run the configured private adapter while guarding protected hashes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from catalog_lib import build_snapshot, load_config, read_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    config, _, project_root = load_config(args.config)
    request_path = Path(args.request).expanduser().resolve()
    request = read_json(request_path)
    before = build_snapshot(config, project_root)
    expected = request["protected_snapshot"]["combined_sha256"]
    if before["combined_sha256"] != expected:
        raise SystemExit("Protected model/protocol hash differs from experiment request; create a new reviewed request")

    output_path = Path(args.output).expanduser().resolve() if args.output else project_root / config.get("state_dir", "state") / "results" / f"{request['experiment_id']}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    replacements = {
        "{python}": sys.executable,
        "{config}": str(Path(args.config).expanduser().resolve()),
        "{request}": str(request_path),
        "{output}": str(output_path),
        "{project_root}": str(project_root),
    }
    command = []
    for part in config["adapter_command"]:
        expanded = str(part)
        for token, value in replacements.items():
            expanded = expanded.replace(token, value)
        command.append(expanded)

    completed = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        timeout=config["experiment"].get("command_timeout_seconds", 86400),
    )
    after = build_snapshot(config, project_root)
    if after["combined_sha256"] != before["combined_sha256"]:
        raise SystemExit("Protected files changed during the experiment; result is invalid")
    if completed.returncode != 0:
        raise SystemExit(f"Adapter failed with exit code {completed.returncode}")
    if not output_path.is_file():
        raise SystemExit(f"Adapter did not create result: {output_path}")
    result = read_json(output_path)
    if result.get("experiment_id") != request["experiment_id"]:
        raise SystemExit("Result experiment_id does not match request")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
