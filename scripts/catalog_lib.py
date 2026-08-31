"""Shared, dependency-free utilities for the PV factor R&D skill."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path, Path]:
    path = Path(config_path).expanduser().resolve()
    config = read_json(path)
    project_root = (path.parent / config.get("project_root", ".")).resolve()
    return config, path, project_root


def protected_files(config: dict[str, Any], project_root: Path) -> list[Path]:
    paths: list[Path] = []
    for key in ("model_paths", "protocol_paths"):
        for item in config.get(key, []):
            candidate = Path(item)
            paths.append(candidate if candidate.is_absolute() else (project_root / candidate).resolve())
    return paths


def build_snapshot(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    files = protected_files(config, project_root)
    if not files:
        raise ValueError("No protected model_paths or protocol_paths are configured")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing protected files: " + ", ".join(missing))

    records = []
    for path in sorted(files, key=str):
        try:
            display_path = str(path.relative_to(project_root))
        except ValueError:
            display_path = str(path)
        records.append({"path": display_path, "sha256": sha256_file(path), "bytes": path.stat().st_size})

    combined = sha256_json(records)
    return {
        "snapshot_id": f"snapshot-{combined[:12]}",
        "created_at": utc_now(),
        "combined_sha256": combined,
        "files": records,
    }


def load_factors() -> list[dict[str, Any]]:
    return read_json(SKILL_ROOT / "factor_library" / "factors.json")["factors"]


def factor_index() -> dict[str, dict[str, Any]]:
    return {factor["id"]: factor for factor in load_factors()}

