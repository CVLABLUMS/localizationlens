from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator


def stable_id(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def unique_images(rows: Iterable[dict]) -> Iterator[dict]:
    seen: set[str] = set()
    for row in rows:
        image_id = row["image_id"]
        if image_id not in seen:
            seen.add(image_id)
            yield row


def stratified_image_limit(rows: list[dict], limit: int | None) -> list[dict]:
    """Limit unique images while retaining each available dataset split."""
    rows = list(unique_images(rows))
    if limit is None or len(rows) <= limit:
        return rows
    by_split: dict[str, list[dict]] = {}
    for row in rows:
        by_split.setdefault(row.get("split", "train"), []).append(row)
    selected: list[dict] = []
    remaining = limit
    splits = sorted(by_split)
    for index, split in enumerate(splits):
        if index == len(splits) - 1:
            count = remaining
        else:
            proportional = round(limit * len(by_split[split]) / len(rows))
            count = min(len(by_split[split]), max(1, proportional))
        selected.extend(by_split[split][:count])
        remaining -= count
    return selected[:limit]
