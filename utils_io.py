import json
import pickle
from pathlib import Path
from typing import Any


def read_jsonl_or_json(path: str) -> list:
    """Read a JSON list or a JSONL file into Python objects."""
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        return json.loads(p.read_text(encoding="utf-8"))


def write_pickle(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)