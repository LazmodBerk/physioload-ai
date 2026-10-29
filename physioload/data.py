"""Dataset records, leakage-safe splits and reproducibility manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SampleRecord:
    image_id: str
    dataset: str
    subject_id: str
    trial_id: str
    camera_id: str
    image_path: str
    pose_label: str | None = None
    reference_path: str | None = None
    view_label: str | None = None
    frame_index: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def dataset_fingerprint(records: Iterable[SampleRecord]) -> str:
    rows = sorted((record.to_dict() for record in records), key=lambda row: row["image_id"])
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def group_split(
    records: list[SampleRecord],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[SampleRecord]]:
    """Split by subject so the same person never crosses partitions."""

    if not records:
        return {"train": [], "validation": [], "test": []}
    if train_fraction <= 0 or validation_fraction < 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must leave a positive test partition")
    subjects = np.array(sorted({record.subject_id for record in records}), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = max(1, int(round(n * train_fraction)))
    n_validation = int(round(n * validation_fraction))
    if n >= 3:
        n_train = min(n_train, n - 2)
        n_validation = max(1, min(n_validation, n - n_train - 1))
    else:
        n_train = min(n_train, n)
        n_validation = min(n_validation, n - n_train)
    train_subjects = set(subjects[:n_train])
    validation_subjects = set(subjects[n_train : n_train + n_validation])
    split = {"train": [], "validation": [], "test": []}
    for record in records:
        destination = (
            "train"
            if record.subject_id in train_subjects
            else "validation"
            if record.subject_id in validation_subjects
            else "test"
        )
        split[destination].append(record)
    return split


def assert_no_group_leakage(splits: dict[str, list[SampleRecord]]) -> None:
    groups = {name: {row.subject_id for row in rows} for name, rows in splits.items()}
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(f"subject leakage between {left} and {right}: {sorted(overlap)}")


def write_manifest(path: Path, *, version: str, params: dict, records: list[SampleRecord]) -> dict:
    payload = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": dataset_fingerprint(records),
        "params": params,
        "record_count": len(records),
        "subjects": len({record.subject_id for record in records}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != payload["fingerprint"] or previous.get("params") != params:
            raise ValueError("manifest mismatch; create a new processed-data version")
        return previous
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload

