"""Unit tests for the baseline metrics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.baseline_metrics import (
    BaselineMetricsConfig,
    compute_baseline_metrics,
    compute_gini,
    compute_ks_statistic,
    compute_performance_metrics,
    compute_psi,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_approved_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    default_flag = rng.integers(0, 2, size=n)
    # Legacy score: weakly correlated with default_flag so that AUC is
    # reliably above 0.5 but far below 1.0, ensuring min_ks=0.99 is not met.
    legacy_score = np.clip(
        default_flag * 0.15 + rng.uniform(0.0, 0.85, size=n), 0.0, 1.0
    )
    return pd.DataFrame(
        {
            "application_id": [f"A{i}" for i in range(n)],
            "default_flag": default_flag,
            "legacy_score": legacy_score,
        }
    )


# ── Tests: compute_ks_statistic ────────────────────────────────────────────────

def test_ks_statistic_range() -> None:
    df = _make_approved_df()
    ks, threshold = compute_ks_statistic(df["default_flag"], df["legacy_score"])
    assert 0.0 <= ks <= 1.0


def test_ks_statistic_perfect_separation() -> None:
    """A perfect classifier should produce KS = 1."""
    y = pd.Series([0] * 100 + [1] * 100)
    # Non-defaults score 0, defaults score 1
    scores = pd.Series([0.0] * 100 + [1.0] * 100)
    ks, _ = compute_ks_statistic(y, scores)
    assert ks == pytest.approx(1.0, abs=1e-3)


def test_ks_statistic_random_classifier() -> None:
    """A random classifier should produce KS near 0."""
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, size=1000))
    scores = pd.Series(rng.uniform(0, 1, size=1000))
    ks, _ = compute_ks_statistic(y, scores)
    assert ks < 0.15


# ── Tests: compute_gini ────────────────────────────────────────────────────────

def test_compute_gini_formula() -> None:
    assert compute_gini(0.75) == pytest.approx(0.5, abs=1e-6)
    assert compute_gini(0.5) == pytest.approx(0.0, abs=1e-6)
    assert compute_gini(1.0) == pytest.approx(1.0, abs=1e-6)


# ── Tests: compute_psi ────────────────────────────────────────────────────────

def test_psi_same_distribution_is_stable() -> None:
    rng = np.random.default_rng(0)
    scores = pd.Series(rng.uniform(0, 1, size=1000))
    result = compute_psi(scores, scores.copy(), n_bins=10)
    assert result.psi_total < 0.1
    assert result.interpretation == "stable"


def test_psi_different_distribution_is_significant() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.uniform(0, 0.3, size=1000))     # Low scores
    comparison = pd.Series(rng.uniform(0.7, 1.0, size=1000))  # High scores
    result = compute_psi(ref, comparison, n_bins=10)
    assert result.psi_total >= 0.25
    assert result.interpretation == "significant_shift"


def test_psi_interpretation_moderate() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0.5, 0.1, size=2000))
    comparison = pd.Series(rng.normal(0.6, 0.15, size=2000))
    result = compute_psi(ref, comparison, n_bins=10)
    assert result.interpretation in ("stable", "moderate_shift", "significant_shift")


# ── Tests: compute_performance_metrics ────────────────────────────────────────

def test_compute_performance_metrics_basic() -> None:
    df = _make_approved_df()
    metrics = compute_performance_metrics(df, "legacy_score", "default_flag", "test")
    assert 0.0 <= metrics.ks_statistic <= 1.0
    assert 0.5 <= metrics.auc_roc <= 1.0
    # Gini and AUC are each rounded to 4 d.p. independently, so allow 2 ULPs
    assert metrics.gini == pytest.approx(2 * metrics.auc_roc - 1, abs=2e-4)
    assert metrics.source == "test"


def test_compute_performance_metrics_missing_score_column() -> None:
    df = _make_approved_df()
    with pytest.raises(KeyError, match="Score column"):
        compute_performance_metrics(df, "nonexistent_score", "default_flag")


def test_compute_performance_metrics_missing_target_column() -> None:
    df = _make_approved_df()
    with pytest.raises(KeyError, match="Target column"):
        compute_performance_metrics(df, "legacy_score", "nonexistent_target")


def test_compute_performance_metrics_insufficient_data() -> None:
    df = _make_approved_df()[:5]
    with pytest.raises(ValueError, match="Insufficient data"):
        compute_performance_metrics(df, "legacy_score", "default_flag")


# ── Tests: compute_baseline_metrics ───────────────────────────────────────────

def test_compute_baseline_metrics_with_legacy_score() -> None:
    approved = _make_approved_df()
    cfg = BaselineMetricsConfig(
        legacy_score_column="legacy_score",
        target_column="default_flag",
        min_ks=0.0,
        min_auc=0.0,
    )
    result = compute_baseline_metrics(approved, cfg)

    assert result.legacy_metrics is not None
    assert result.meets_minimum_ks is True
    assert result.meets_minimum_auc is True
    assert "legacy_metrics" in result.report


def test_compute_baseline_metrics_without_legacy_score() -> None:
    approved = _make_approved_df().drop(columns=["legacy_score"])
    result = compute_baseline_metrics(approved)

    assert result.legacy_metrics is None
    assert result.meets_minimum_ks is None
    assert result.meets_minimum_auc is None


def test_compute_baseline_metrics_with_psi() -> None:
    approved = _make_approved_df(n=500)
    comparison = _make_approved_df(n=300, seed=99)
    cfg = BaselineMetricsConfig(legacy_score_column="legacy_score")
    result = compute_baseline_metrics(approved, cfg, comparison_df=comparison)

    assert result.psi is not None
    assert isinstance(result.psi.psi_total, float)
    assert result.psi.interpretation in ("stable", "moderate_shift", "significant_shift")


def test_baseline_metrics_flags_below_minimum_ks() -> None:
    """When min_ks is set very high, meets_minimum_ks should be False."""
    approved = _make_approved_df()
    cfg = BaselineMetricsConfig(
        legacy_score_column="legacy_score",
        min_ks=0.99,  # Unreachably high
        min_auc=0.0,
    )
    result = compute_baseline_metrics(approved, cfg)
    assert result.meets_minimum_ks is False
