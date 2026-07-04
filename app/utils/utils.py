"""Shared utilities."""


from __future__ import annotations

from pathlib import Path


def normalize_dataset_name(name: str) -> str:
    """Turn `olist_<table>_dataset` into `<table>`."""
    if name.startswith("olist_"):
        name = name[len("olist_"):]
    if name.endswith("_dataset"):
        name = name[: -len("_dataset")]
    return name




def discover_csv_files(data_dir: Path) -> dict[str, Path]:
    """Discover and map all CSV files in data directory to table names.
    """
    return {
        normalize_dataset_name(path.stem): path
        for path in data_dir.glob("*.csv")
    }