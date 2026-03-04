"""Unit tests for the proxy variable detection module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.compliance.proxy_detection import (
    ProxyDetectionConfig,
    detect_proxy_variables,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_df(
    n: int = 500,
    seed: int = 42,
    include_correlated_proxy: bool = False,
) -> pd.DataFrame:
    """Build a synthetic DataFrame for proxy detection tests."""
    rng = np.random.default_rng(seed)

    race = rng.integers(0, 2, size=n)  # Binary protected class

    data: dict = {
        "credit_score": rng.integers(300, 850, size=n).astype(float),
        "dti_ratio": rng.uniform(0.1, 0.6, size=n),
        "income": rng.uniform(30_000, 200_000, size=n),
        "race": race,
    }

    if include_correlated_proxy:
        # Create a feature that is highly correlated with race
        data["zip_code_group"] = race * 10 + rng.integers(0, 3, size=n)

    return pd.DataFrame(data)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_no_protected_columns_present() -> None:
    df = _make_df()
    result = detect_proxy_variables(
        df=df,
        feature_columns=["credit_score", "dti_ratio", "income"],
        config=ProxyDetectionConfig(protected_class_columns=["sex"]),  # Not in df
    )
    assert result.flagged_features == []
    assert result.clean_features == ["credit_score", "dti_ratio", "income"]


def test_clean_features_pass_scan() -> None:
    df = _make_df()
    cfg = ProxyDetectionConfig(
        protected_class_columns=["race"],
        correlation_threshold=0.4,
    )
    result = detect_proxy_variables(
        df=df,
        feature_columns=["credit_score", "dti_ratio", "income"],
        config=cfg,
    )
    # Independent features should not be flagged
    assert len(result.flagged_features) == 0
    assert set(result.clean_features) == {"credit_score", "dti_ratio", "income"}


def test_highly_correlated_feature_is_flagged() -> None:
    df = _make_df(n=1000, include_correlated_proxy=True)
    cfg = ProxyDetectionConfig(
        protected_class_columns=["race"],
        correlation_threshold=0.3,
        significance_level=0.05,
    )
    result = detect_proxy_variables(
        df=df,
        feature_columns=["credit_score", "dti_ratio", "income", "zip_code_group"],
        config=cfg,
    )
    assert "zip_code_group" in result.flagged_features
    assert "zip_code_group" not in result.clean_features


def test_flagged_feature_excluded_from_clean() -> None:
    df = _make_df(n=1000, include_correlated_proxy=True)
    cfg = ProxyDetectionConfig(
        protected_class_columns=["race"],
        correlation_threshold=0.3,
    )
    result = detect_proxy_variables(
        df=df,
        feature_columns=["credit_score", "dti_ratio", "income", "zip_code_group"],
        config=cfg,
    )
    # Flagged and clean should be disjoint and cover all features
    all_features = set(["credit_score", "dti_ratio", "income", "zip_code_group"])
    assert set(result.flagged_features) | set(result.clean_features) == all_features
    assert set(result.flagged_features) & set(result.clean_features) == set()


def test_report_has_expected_keys() -> None:
    df = _make_df()
    cfg = ProxyDetectionConfig(protected_class_columns=["race"])
    result = detect_proxy_variables(
        df=df,
        feature_columns=["credit_score", "dti_ratio"],
        config=cfg,
    )
    report = result.report
    assert "total_features_scanned" in report
    assert "flagged_features" in report
    assert "clean_features" in report
    assert "num_flagged" in report
    assert "num_clean" in report


def test_proxy_flags_detail() -> None:
    df = _make_df(n=1000, include_correlated_proxy=True)
    cfg = ProxyDetectionConfig(
        protected_class_columns=["race"],
        correlation_threshold=0.3,
    )
    result = detect_proxy_variables(
        df=df,
        feature_columns=["zip_code_group"],
        config=cfg,
    )
    if result.proxy_flags:
        flag = result.proxy_flags[0]
        assert flag.feature == "zip_code_group"
        assert flag.protected_column == "race"
        assert 0.0 <= flag.correlation <= 1.0
        assert 0.0 <= flag.p_value <= 1.0


def test_missing_feature_column_skipped() -> None:
    df = _make_df()
    cfg = ProxyDetectionConfig(protected_class_columns=["race"])
    result = detect_proxy_variables(
        df=df,
        feature_columns=["credit_score", "nonexistent_column"],
        config=cfg,
    )
    # Should not raise; nonexistent column is silently skipped
    assert "nonexistent_column" not in result.flagged_features
