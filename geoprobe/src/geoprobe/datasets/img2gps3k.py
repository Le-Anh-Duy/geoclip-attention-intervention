from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from geoprobe.registry import Registry

DATASETS = Registry("dataset")


@dataclass(frozen=True)
class DatasetRecord:
    sample_id: str
    image_path: Path
    latitude: float
    longitude: float
    split: str

    @property
    def name(self) -> str:
        return self.sample_id


def make_filename_split(names: Sequence[str], discovery_count: int = 1079) -> dict[str, str]:
    values = list(names)
    if any(not isinstance(name, str) or not name for name in values):
        raise ValueError("Every sample name must be a non-empty string")
    if len(values) != len(set(values)):
        raise ValueError("Sample names must be unique")
    if not 0 <= discovery_count <= len(values):
        raise ValueError(f"discovery_count {discovery_count} is outside [0, {len(values)}]")
    ordered = sorted(values, key=lambda name: (hashlib.sha256(name.encode("utf-8")).digest(), name))
    discovery = set(ordered[:discovery_count])
    return {name: ("discovery" if name in discovery else "holdout") for name in values}


def split_fingerprint(records: Sequence[DatasetRecord]) -> str:
    identity = sorted((row.sample_id, row.split) for row in records)
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


@DATASETS.register("Img2GPS3KDataset")
class Img2GPS3KDataset(Sequence[DatasetRecord]):
    REQUIRED_COLUMNS = ("name", "LAT", "LON")

    def __init__(
        self,
        root: str | Path,
        csv: str | Path = "test_set/im2gps3k_places365.csv",
        images: str | Path = "test_set/im2gps3ktest/im2gps3ktest",
        expected_count: int | None = 2997,
        discovery_count: int = 1079,
    ) -> None:
        root_path = Path(root).expanduser().resolve()
        csv_path = root_path / csv
        image_root = root_path / images
        if not csv_path.is_file():
            raise FileNotFoundError(f"Img2GPS3k CSV does not exist: {csv_path}")
        if not image_root.is_dir():
            raise FileNotFoundError(f"Img2GPS3k image directory does not exist: {image_root}")
        frame = pd.read_csv(csv_path, dtype={"name": str})
        missing = [column for column in self.REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Img2GPS3k CSV {csv_path} is missing columns: {missing}")
        if expected_count is not None and len(frame) != expected_count:
            raise ValueError(f"Img2GPS3k expected {expected_count} rows, found {len(frame)} in {csv_path}")
        names = frame["name"].tolist()
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"Img2GPS3k CSV {csv_path} contains empty sample names")
        duplicates = frame.loc[frame["name"].duplicated(keep=False), "name"].tolist()
        if duplicates:
            raise ValueError(f"Img2GPS3k CSV contains duplicate names: {sorted(set(duplicates))[:5]}")
        coordinates = frame[["LAT", "LON"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(coordinates).all():
            raise ValueError(f"Img2GPS3k CSV {csv_path} contains non-finite coordinates")
        if np.any((coordinates[:, 0] < -90.0) | (coordinates[:, 0] > 90.0)):
            raise ValueError("Img2GPS3k latitude must be in [-90, 90]")
        if np.any((coordinates[:, 1] < -180.0) | (coordinates[:, 1] > 180.0)):
            raise ValueError("Img2GPS3k longitude must be in [-180, 180]")
        splits = make_filename_split(names, discovery_count)
        records: list[DatasetRecord] = []
        for index, name in enumerate(names):
            image_path = (image_root / name).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Img2GPS3k image for sample {name!r} does not exist: {image_path}")
            records.append(DatasetRecord(name, image_path, float(coordinates[index, 0]), float(coordinates[index, 1]), splits[name]))
        if sum(row.split == "discovery" for row in records) != discovery_count:
            raise AssertionError("Discovery split count invariant failed")
        self.root = root_path
        self.csv_path = csv_path
        self.image_root = image_root
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> DatasetRecord:
        return self.records[index]

    def __iter__(self) -> Iterator[DatasetRecord]:
        return iter(self.records)
