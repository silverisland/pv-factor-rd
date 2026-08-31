from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_strings_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def numeric_array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float32)
    canonical = np.ascontiguousarray(array.astype("<f4", copy=False))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(canonical.shape)).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def split_fingerprint(row_ids: Sequence[str]) -> dict[str, Any]:
    values = [str(value) for value in row_ids]
    return {"rows": len(values), "row_id_sha256": ordered_strings_sha256(values)}


def named_numeric_arrays_sha256(values: Mapping[str, Any]) -> str:
    """Hash named tensor/array contents without depending on their container type."""
    items: list[dict[str, Any]] = []
    for name in sorted(values):
        value = values[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        items.append(
            {
                "name": str(name),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "values_sha256": numeric_array_sha256(array),
            }
        )
    return canonical_json_sha256(items)
