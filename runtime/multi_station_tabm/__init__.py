"""Self-contained multi-station shared TabM runtime."""

from __future__ import annotations

from typing import Any


def train(*args: Any, **kwargs: Any) -> Any:
    from .api import train as implementation

    return implementation(*args, **kwargs)


def evaluate(*args: Any, **kwargs: Any) -> Any:
    from .api import evaluate as implementation

    return implementation(*args, **kwargs)


def predict(*args: Any, **kwargs: Any) -> Any:
    from .api import predict as implementation

    return implementation(*args, **kwargs)


__all__ = ["train", "evaluate", "predict"]
