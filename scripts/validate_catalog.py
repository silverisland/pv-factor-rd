#!/usr/bin/env python3
"""Validate catalog schemas plus cross-record and implementation integrity."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from catalog_lib import SKILL_ROOT, read_json


ALLOWED_EVIDENCE = {
    "K0", "K1", "K2", "K3", "K4", "K5", "K6", "E1", "E2", "E3"
}
ALLOWED_STATUS = {"proposed", "implemented", "validated", "accepted", "rejected", "inconclusive", "deprecated"}
ALLOWED_OUTPUTS = {"static", "history_sequence", "future_known_sequence"}
ALLOWED_AVAILABILITY = {"available", "conditional", "unavailable"}
VISUAL_TOKENS = {"image", "camera", "satellite_image", "sky_image", "pixel", "visual_embedding"}


def _unique_ids(records: list[dict], label: str, errors: list[str]) -> set[str]:
    ids = [record.get("id") for record in records]
    if any(not value for value in ids):
        errors.append(f"{label}: every record needs a non-empty id")
    duplicates = sorted({value for value in ids if value and ids.count(value) > 1})
    if duplicates:
        errors.append(f"{label}: duplicate ids: {duplicates}")
    return {value for value in ids if value}


def _validate_json_schema(
    document: dict, schema_name: str, label: str, errors: list[str]
) -> bool:
    """Use the checked-in schema when jsonschema is installed."""
    try:
        import jsonschema
    except ImportError:
        return False
    schema = read_json(SKILL_ROOT / "schemas" / schema_name)
    validator = jsonschema.Draft202012Validator(schema)
    for error in sorted(
        validator.iter_errors(document),
        key=lambda item: tuple(str(value) for value in item.path),
    ):
        location = ".".join(str(value) for value in error.absolute_path) or "<root>"
        errors.append(f"{label} schema at {location}: {error.message}")
    return True


def validate() -> list[str]:
    errors: list[str] = []
    sources_doc = read_json(SKILL_ROOT / "knowledge" / "sources.json")
    mechanisms_doc = read_json(SKILL_ROOT / "knowledge" / "mechanisms.json")
    factors_doc = read_json(SKILL_ROOT / "factor_library" / "factors.json")
    schema_checked = [
        _validate_json_schema(
            mechanisms_doc, "mechanism.schema.json", "mechanisms", errors
        ),
        _validate_json_schema(factors_doc, "factor.schema.json", "factors", errors),
    ]
    if any(schema_checked) and not all(schema_checked):
        errors.append("JSON Schema validation ran incompletely")

    for name, doc in (("sources", sources_doc), ("mechanisms", mechanisms_doc), ("factors", factors_doc)):
        if doc.get("schema_version") != 1:
            errors.append(f"{name}: schema_version must be 1")

    sources = sources_doc.get("sources", [])
    mechanisms = mechanisms_doc.get("mechanisms", [])
    factors = factors_doc.get("factors", [])
    source_ids = _unique_ids(sources, "sources", errors)
    mechanism_ids = _unique_ids(mechanisms, "mechanisms", errors)
    _unique_ids(factors, "factors", errors)

    for mechanism in mechanisms:
        mid = mechanism.get("id", "<missing>")
        if mechanism.get("evidence_level") not in ALLOWED_EVIDENCE:
            errors.append(f"{mid}: invalid evidence_level")
        unknown = set(mechanism.get("source_ids", [])) - source_ids
        if unknown:
            errors.append(f"{mid}: unknown source_ids {sorted(unknown)}")
        for field in ("claim", "causal_chain", "required_variables", "expected_horizon", "failure_modes"):
            if not mechanism.get(field):
                errors.append(f"{mid}: missing {field}")

    for factor in factors:
        fid = factor.get("id", "<missing>")
        unknown = set(factor.get("mechanism_ids", [])) - mechanism_ids
        if unknown:
            errors.append(f"{fid}: unknown mechanism_ids {sorted(unknown)}")
        if factor.get("evidence_level") not in ALLOWED_EVIDENCE:
            errors.append(f"{fid}: invalid evidence_level")
        if factor.get("status") not in ALLOWED_STATUS:
            errors.append(f"{fid}: invalid status")
        if factor.get("output_type") not in ALLOWED_OUTPUTS:
            errors.append(f"{fid}: invalid output_type")
        if factor.get("data_availability") not in ALLOWED_AVAILABILITY:
            errors.append(f"{fid}: invalid data_availability")
        for field in ("required_inputs", "formula", "availability_rule", "expected_horizon", "failure_modes", "diagnostics"):
            if not factor.get(field):
                errors.append(f"{fid}: missing {field}")

        input_tokens = {str(item).lower() for item in factor.get("required_inputs", [])}
        if any(any(token in item for token in VISUAL_TOKENS) for item in input_tokens):
            errors.append(f"{fid}: visual/image inputs are outside this skill's scope")

        if factor.get("output_type") == "future_known_sequence":
            rule = str(factor.get("availability_rule", "")).lower()
            if not any(token in rule for token in ("origin", "issue", "deterministic", "historical", "training", "fit")):
                errors.append(f"{fid}: future-known factor lacks an origin/issue-time availability rule")

        implementation = factor.get("implementation")
        if factor.get("status") == "implemented" and not implementation:
            errors.append(f"{fid}: implemented status requires an implementation")
        if implementation:
            try:
                relative_path, function_name = implementation.split(":", 1)
            except ValueError:
                errors.append(f"{fid}: implementation must be path.py:function")
                continue
            module_path = SKILL_ROOT / relative_path
            if not module_path.is_file():
                errors.append(f"{fid}: implementation file does not exist: {relative_path}")
                continue
            spec = importlib.util.spec_from_file_location(f"factor_impl_{fid.replace('.', '_')}", module_path)
            if spec is None or spec.loader is None:
                errors.append(f"{fid}: cannot load implementation module")
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not callable(getattr(module, function_name, None)):
                errors.append(f"{fid}: implementation function does not exist: {function_name}")

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("Catalog validation FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    sources = len(read_json(SKILL_ROOT / "knowledge" / "sources.json")["sources"])
    mechanisms = len(read_json(SKILL_ROOT / "knowledge" / "mechanisms.json")["mechanisms"])
    factors = len(read_json(SKILL_ROOT / "factor_library" / "factors.json")["factors"])
    print(f"Catalog validation passed: {sources} sources, {mechanisms} mechanisms, {factors} factors")
    if importlib.util.find_spec("jsonschema") is None:
        print(
            "Catalog note: install runtime requirements to enable checked-in JSON Schema validation",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
