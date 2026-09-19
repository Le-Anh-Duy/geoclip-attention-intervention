from __future__ import annotations

import copy
import hashlib
import json
import runpy
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    namespace = runpy.run_path(str(path))
    exported = {key for key in namespace if not key.startswith("__")}
    if exported != {"config"}:
        raise ValueError(
            f"Config {path} must export exactly 'config'; found {sorted(exported)}"
        )
    value = namespace["config"]
    if not isinstance(value, dict):
        raise TypeError(f"Config {path} 'config' must be a dict, got {type(value).__name__}")
    return copy.deepcopy(value)


def config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
