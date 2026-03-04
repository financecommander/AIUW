"""Unit tests for the data ingestion module."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.training.data_ingestion import (
    IngestionConfig,
    generate_ingestion_report,
    ingest_data,
    load_raw_data,
    validate_schema,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_lending_df(
    n_approved: int = 200,
    n_rejected: int = 100,
    include_protected: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a minimal synthetic lending DataFrame."""
    rng = np.random.default_rng(seed)
    n = n_approved + n_rejected

    approved_flag = np.array([1] * n_approved + [0] * n_rejected)
    default_flag = np.where(
        approved_flag == 1, rng.integers(0, 2, size=n), 0
    )

    data: dict = {
        "application_id": [f"APP-{i:05d}" for i in range(n)],
        "approved_flag": approved_flag,
        "default_flag": default_flag,
        "credit_score": rng.integers(300, 850, size=n).astype(float),
        "dti_ratio": rng.uniform(0.1, 0.6, size=n),
        "income": rng.uniform(30_000, 200_000, size=n),
        "legacy_score": rng.uniform(0.0, 1.0, size=n),
    }
    if include_protected:
        data["race"] = rng.choice(["A", "B", "C"], size=n)

    return pd.DataFrame(data)


@pytest.fixture
def tmp_parquet(tmp_path: Path) -> Path:
    """Write a synthetic lending DataFrame to a temp Parquet file."""
    df = _make_lending_df()
    path = tmp_path / "lending.parquet"
    df.to_parquet(path, index=False)
    return path


# ── Tests: load_raw_data ───────────────────────────────────────────────────────

def test_load_raw_data_success(tmp_parquet: Path) -> None:
    df = load_raw_data(tmp_parquet)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_raw_data_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "does_not_exist.parquet")


def test_load_raw_data_invalid_file(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.parquet"
    bad_file.write_bytes(b"not a parquet file")
    with pytest.raises(ValueError):
        load_raw_data(bad_file)


# ── Tests: validate_schema ─────────────────────────────────────────────────────

def test_validate_schema_passes() -> None:
    df = _make_lending_df()
    cfg = IngestionConfig(data_path="irrelevant")
    validate_schema(df, cfg)  # Should not raise


def test_validate_schema_missing_column() -> None:
    df = _make_lending_df().drop(columns=["default_flag"])
    cfg = IngestionConfig(data_path="irrelevant")
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(df, cfg)


# ── Tests: ingest_data ─────────────────────────────────────────────────────────

def test_ingest_data_splits_populations(tmp_parquet: Path) -> None:
    cfg = IngestionConfig(
        data_path=tmp_parquet,
        protected_class_columns=["race"],
    )
    result = ingest_data(cfg)

    assert result.row_counts["approved"] + result.row_counts["rejected"] == result.row_counts["total"]
    assert all(result.approved["approved_flag"] == 1)
    assert all(result.rejected["approved_flag"] == 0)


def test_ingest_data_feature_columns_exclude_meta(tmp_parquet: Path) -> None:
    cfg = IngestionConfig(data_path=tmp_parquet)
    result = ingest_data(cfg)

    assert "application_id" not in result.feature_columns
    assert "approved_flag" not in result.feature_columns
    assert "default_flag" not in result.feature_columns


def test_ingest_data_detects_protected_columns(tmp_path: Path) -> None:
    df = _make_lending_df(include_protected=True)
    path = tmp_path / "lending_protected.parquet"
    df.to_parquet(path, index=False)

    cfg = IngestionConfig(
        data_path=path,
        protected_class_columns=["race"],
    )
    result = ingest_data(cfg)

    assert "race" in result.protected_columns_present
    assert "race" not in result.feature_columns


def test_ingest_data_drops_high_missingness_column(tmp_path: Path) -> None:
    df = _make_lending_df()
    # Introduce a column that is 90% NaN
    df["mostly_missing"] = np.nan
    df.loc[: int(len(df) * 0.1), "mostly_missing"] = 1.0
    path = tmp_path / "missing.parquet"
    df.to_parquet(path, index=False)

    cfg = IngestionConfig(data_path=path, drop_na_threshold=0.50)
    result = ingest_data(cfg)

    assert "mostly_missing" in result.dropped_columns


# ── Tests: generate_ingestion_report ──────────────────────────────────────────

def test_generate_ingestion_report_structure(tmp_parquet: Path) -> None:
    cfg = IngestionConfig(data_path=tmp_parquet)
    result = ingest_data(cfg)
    report = generate_ingestion_report(result)

    assert "row_counts" in report
    assert "feature_columns" in report
    assert "num_features" in report
    assert report["num_features"] == len(result.feature_columns)
