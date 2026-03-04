"""Data ingestion module for the AI Underwriting Engine.

Responsible for:
- Loading historical lending data (Parquet format).
- Validating the schema against expected columns and types.
- Handling missing values and performing basic feature engineering prep.
- Splitting the dataset into approved and rejected populations (required for
  reject-inference downstream).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Required columns that every raw lending dataset must contain ───────────────
REQUIRED_COLUMNS: list[str] = [
    "application_id",
    "approved_flag",
    "default_flag",
]

# ── Columns that represent numeric credit features (non-exhaustive; validated
#    at load time against whatever is present in the file). ───────────────────
EXPECTED_NUMERIC_PREFIXES: tuple[str, ...] = (
    "credit_score",
    "dti",
    "ltv",
    "income",
    "loan_amount",
    "derogatory",
    "months_employed",
    "revolving_utilization",
    "num_open_accounts",
    "num_delinquencies",
)


@dataclass
class IngestionConfig:
    """Configuration for the data-ingestion step."""

    data_path: str | Path
    target_column: str = "default_flag"
    approved_flag_column: str = "approved_flag"
    application_id_column: str = "application_id"
    protected_class_columns: list[str] = field(default_factory=list)
    drop_na_threshold: float = 0.50  # Drop columns with >50 % missing values


@dataclass
class IngestionResult:
    """Output produced by :func:`ingest_data`."""

    approved: pd.DataFrame
    rejected: pd.DataFrame
    feature_columns: list[str]
    protected_columns_present: list[str]
    dropped_columns: list[str]
    row_counts: dict[str, int]


# ── Public API ─────────────────────────────────────────────────────────────────

def load_raw_data(data_path: str | Path) -> pd.DataFrame:
    """Load raw historical lending data from a Parquet file.

    Parameters
    ----------
    data_path:
        Filesystem path to the ``.parquet`` file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with the file contents.

    Raises
    ------
    FileNotFoundError
        When *data_path* does not exist.
    ValueError
        When the file cannot be parsed as Parquet.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading raw data from %s", path)
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"Failed to read Parquet file '{path}': {exc}") from exc

    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))
    return df


def validate_schema(df: pd.DataFrame, config: IngestionConfig) -> None:
    """Validate that *df* contains all required columns.

    Parameters
    ----------
    df:
        DataFrame to validate.
    config:
        Ingestion configuration.

    Raises
    ------
    ValueError
        When any required column is absent.
    """
    required = {
        config.application_id_column,
        config.approved_flag_column,
        config.target_column,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )
    logger.info("Schema validation passed — all required columns present.")


def _drop_high_missingness_columns(
    df: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns whose missing-value rate exceeds *threshold*.

    Returns the cleaned DataFrame and the list of dropped column names.
    """
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()
    if cols_to_drop:
        logger.warning(
            "Dropping %d column(s) with >%.0f%% missing values: %s",
            len(cols_to_drop),
            threshold * 100,
            cols_to_drop,
        )
        df = df.drop(columns=cols_to_drop)
    return df, cols_to_drop


def _identify_feature_columns(
    df: pd.DataFrame,
    config: IngestionConfig,
) -> tuple[list[str], list[str]]:
    """Identify modelling feature columns and present protected-class columns.

    Feature columns are all numeric columns that are *not* the target,
    the approved flag, the application ID, or any protected-class column.
    """
    exclude = {
        config.target_column,
        config.approved_flag_column,
        config.application_id_column,
        *config.protected_class_columns,
    }
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    protected_present = [
        c for c in config.protected_class_columns if c in df.columns
    ]
    return feature_cols, protected_present


def ingest_data(config: IngestionConfig) -> IngestionResult:
    """Execute the full data-ingestion pipeline step.

    Steps performed:
    1. Load raw Parquet data.
    2. Validate schema.
    3. Drop columns with excessive missing values.
    4. Split into *approved* and *rejected* populations.
    5. Identify feature and protected-class columns.

    Parameters
    ----------
    config:
        Ingestion configuration.

    Returns
    -------
    IngestionResult
        Populations, feature lists, and summary statistics.
    """
    df = load_raw_data(config.data_path)
    validate_schema(df, config)

    df, dropped = _drop_high_missingness_columns(df, config.drop_na_threshold)

    approved = df[df[config.approved_flag_column] == 1].copy()
    rejected = df[df[config.approved_flag_column] == 0].copy()

    feature_cols, protected_present = _identify_feature_columns(df, config)

    row_counts: dict[str, int] = {
        "total": len(df),
        "approved": len(approved),
        "rejected": len(rejected),
    }
    logger.info(
        "Ingestion complete — total=%d, approved=%d, rejected=%d",
        row_counts["total"],
        row_counts["approved"],
        row_counts["rejected"],
    )

    if protected_present:
        logger.warning(
            "Protected-class columns detected in dataset — these will NOT be "
            "used as model features: %s",
            protected_present,
        )

    return IngestionResult(
        approved=approved,
        rejected=rejected,
        feature_columns=feature_cols,
        protected_columns_present=protected_present,
        dropped_columns=dropped,
        row_counts=row_counts,
    )


def generate_ingestion_report(result: IngestionResult) -> dict[str, Any]:
    """Produce a summary report dict suitable for logging or MLflow artifact.

    Parameters
    ----------
    result:
        Output of :func:`ingest_data`.

    Returns
    -------
    dict
        Human-readable summary of the ingestion run.
    """
    approved_default_rate = (
        result.approved["default_flag"].mean()
        if "default_flag" in result.approved.columns and len(result.approved) > 0
        else None
    )
    return {
        "row_counts": result.row_counts,
        "feature_columns": result.feature_columns,
        "protected_columns_present": result.protected_columns_present,
        "dropped_columns": result.dropped_columns,
        "approved_default_rate": approved_default_rate,
        "num_features": len(result.feature_columns),
    }
