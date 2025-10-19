import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple
""" System Modules """


"""
Script for loading env from given Configuration.

@author: Nikhil Bhargava
@date: 2024-06-15
@license: Apache-2.0
@description: This module loads environment variables from a JSON configuration file.
"""

def _to_env_key(path_tuple: Tuple[str, ...],
                prefix: str = ""
                ) -> str:
    """
    Load nested JSON keys as flattened env vars:
      { "train": { "batch_size": 2 } } -> RAPTOR_TRAIN_BATCH_SIZE=2

    :param path_tuple: Path tuple, e.g. ("train", "batch_size")
    :param prefix: Optional prefix, e.g. "RAPTOR_"

    :return: Env var key, e.g. "RAPTOR_TRAIN_BATCH_SIZE"
    """
    key = "_".join(p.upper().replace("-", "_") for p in path_tuple)
    return f"{prefix}{key}" if prefix else key

def _coerce_to_str(val: Any) -> str:
    """
    Coerce a value to a string for env var storage.
    :param val: Value to coerce
    :return: String representation
    """
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        # join lists as comma-separated strings
        return ",".join(_coerce_to_str(v) for v in val)
    return str(val)

def _flatten(d: Dict[str, Any], prefix_path=()):
    """
    Recursively flatten a nested dictionary, yielding (path_tuple, value) pairs.
    1. If value is a dict, recurse with updated prefix_path.
    2. Else yield (prefix_path + (key,), value).
    3. Initial call should use prefix_path=().
    4. Example: { "a": { "b": 1, "c": 2 }, "d": 3 } yields:
       (("a","b"), 1), (("a","c"), 2), (("d",), 3)

    :param d: Dictionary to flatten
    :param prefix_path: Current path tuple (for recursion)

    :return: Yields (path_tuple, value) pairs
    """
    for k, v in d.items():
        p = prefix_path + (k,)
        if isinstance(v, dict):
            yield from _flatten(v, p)
        else:
            yield p, v

def load_env_from_json(json_path: str = "config.json",
                       prefix: str = "RAPTOR_",
                       override: bool = True
                       ) -> None:
    """
    Load env vars from JSON, flattening nested keys:
      { "train": { "batch_size": 2 } } -> RAPTOR_TRAIN_BATCH_SIZE=2
    If override=False, existing os.environ keys are preserved.

    :param override: Whether to override existing env vars (default True)
    :param json_path: Path to JSON file
    :param prefix: Optional prefix for env vars

    :return: None
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Env JSON not found: {json_path}")
    data = json.loads(path.read_text())

    for path_tuple, value in _flatten(data):
        env_key = _to_env_key(path_tuple, prefix=prefix)
        if not override and env_key in os.environ:
            continue
        os.environ[env_key] = _coerce_to_str(value)

    # Also map a few friendly keys to widely used names
    if os.getenv("RAPTOR_LOGGING_LEVEL"):
        os.environ.setdefault("LOG_LEVEL", os.getenv("RAPTOR_LOGGING_LEVEL"))
    if os.getenv("RAPTOR_PATHS_LOG_FILE"):
        os.environ.setdefault("LOG_FILE", os.getenv("RAPTOR_PATHS_LOG_FILE"))
    if os.getenv("RAPTOR_LOGGING_FORMAT"):
        os.environ.setdefault("LOG_FORMAT", os.getenv("RAPTOR_LOGGING_FORMAT"))
    if os.getenv("RAPTOR_LOGGING_ROTATE_MAX_BYTES"):
        os.environ.setdefault("LOG_ROTATE_MAX_BYTES", os.getenv("RAPTOR_LOGGING_ROTATE_MAX_BYTES"))
    if os.getenv("RAPTOR_LOGGING_ROTATE_BACKUPS"):
        os.environ.setdefault("LOG_ROTATE_BACKUPS", os.getenv("RAPTOR_LOGGING_ROTATE_BACKUPS"))
