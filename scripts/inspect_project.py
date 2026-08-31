#!/usr/bin/env python3
"""Create a static protected-file snapshot; never import training code."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

from catalog_lib import build_snapshot, load_config, protected_files, write_json


INTERESTING_NAMES = {
    "LOOKBACK", "HISTORY_STEPS", "HORIZON", "HORIZON_STEPS", "TARGET_IDX",
    "TARGET_INDEX", "N_EPOCHS", "BATCH_SIZE", "SEED", "RANDOM_STATE",
}


def static_constants(path: Path) -> dict[str, object]:
    if path.suffix != ".py":
        return {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}
    constants: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id.upper() in INTERESTING_NAMES:
                try:
                    constants[target.id] = ast.literal_eval(value_node)
                except (ValueError, TypeError):
                    constants[target.id] = "<dynamic>"
    return constants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output")
    args = parser.parse_args()

    config, _, project_root = load_config(args.config)
    snapshot = build_snapshot(config, project_root)
    snapshot["static_constants"] = {
        record["path"]: static_constants(path)
        for record, path in zip(snapshot["files"], sorted(protected_files(config, project_root), key=str))
    }
    output = Path(args.output).expanduser().resolve() if args.output else project_root / config.get("state_dir", "state") / "project_snapshot.json"
    write_json(output, snapshot)
    print(f"Wrote {snapshot['snapshot_id']} to {output}")
    print(f"Protected combined SHA-256: {snapshot['combined_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

