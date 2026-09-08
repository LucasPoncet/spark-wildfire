import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def to_json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [to_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    return value


def append_metrics_record(record: Mapping[str, Any], metrics_path: Path) -> Path:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(to_json_safe(record), sort_keys=True) + "\n")
    return metrics_path


def write_metrics_document(document: Mapping[str, Any], metrics_path: Path) -> Path:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(to_json_safe(document), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics_path
