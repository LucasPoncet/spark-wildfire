import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def to_json_safe(value: Any) -> Any:
    """Converts numpy scalars and arrays into JSON-serializable Python values.

    Args:
        value: Any nested structure of mappings, sequences and numpy values.

    Returns:
        The same structure with numpy types replaced by built-ins.
    """
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
    """Appends one metric record as a JSON line.

    Args:
        record: The record to append.
        metrics_path: Destination file, created with its parents if absent.

    Returns:
        The path written to.
    """
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(to_json_safe(record), sort_keys=True) + "\n")
    return metrics_path


def write_metrics_document(document: Mapping[str, Any], metrics_path: Path) -> Path:
    """Writes one whole metrics document, replacing any existing file.

    Args:
        document: The document to write.
        metrics_path: Destination file, created with its parents if absent.

    Returns:
        The path written to.
    """
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(to_json_safe(document), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics_path
