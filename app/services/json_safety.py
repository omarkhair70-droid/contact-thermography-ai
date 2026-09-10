from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any


def sanitize_for_json(value: Any) -> Any:
    """Recursively convert non-finite numeric values to JSON-safe ``None``.

    The API intentionally uses strict JSON semantics: NaN and +/-Inf must never
    escape into FastAPI responses or persisted exam JSON.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    if hasattr(value, "item"):
        try:
            return sanitize_for_json(value.item())
        except (TypeError, ValueError):
            pass
    return value
