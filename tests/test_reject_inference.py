"""Unit tests for the reject inference module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.reject_inference import (
    FuzzyConfig,
    IterativeConfig,
    apply_reject_inference,
    fuzzy_augmentation,
    iterative_reclassification,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_populations(
    n_approved: int = 300,
    n_rejected: int = 150,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build synthetic approved and rejected populations."""
    rng = np.random.default_rng(seed)

    approved = pd.DataFrame(
        {
            "application_id": [f"A{i}" for i in range(n_approved)],
            "approved_flag": 1,
            "default_flag": rng.integers(0, 2, size=n_approved),
            "credit_score": rng.uniform(400, 850, size=n_approved),
            "dti_ratio": rng.uniform(0.1, 0.5, size=n_approved),
            "income": rng.uniform(30_000, 150_000, size=n_approved),
        }
    )

    rejected = pd.DataFrame(
        {
            "application_id": [f"R{i}" for i in range(n_rejected)],
            "approved_flag": 0,
            "default_flag": np.nan,  # Unknown outcome
            "credit_score": rng.uniform(300, 650, size=n_rejected),
            "dti_ratio": rng.uniform(0.3, 0.8, size=n_rejected),
            "income": rng.uniform(15_000, 80_000, size=n_rejected),
        }
    )

    feature_cols = ["credit_score", "dti_ratio", "income"]
    return approved, rejected, feature_cols


# ── Tests: iterative_reclassification ─────────────────────────────────────────

def test_iterative_reclassification_augments_data() -> None:
    approved, rejected, features = _make_populations()
    result = iterative_reclassification(approved, rejected, features)

    assert len(result.augmented_data) == len(approved) + len(rejected)
    assert result.method == "iterative_reclassification"
    assert result.reject_weights is None


def test_iterative_reclassification_labels_are_binary() -> None:
    approved, rejected, features = _make_populations()
    result = iterative_reclassification(approved, rejected, features)

    unique_labels = set(result.reject_labels.unique())
    assert unique_labels.issubset({0, 1})


def test_iterative_reclassification_convergence() -> None:
    approved, rejected, features = _make_populations()
    cfg = IterativeConfig(max_iterations=5, convergence_threshold=0.001)
    result = iterative_reclassification(approved, rejected, features, config=cfg)

    assert result.iterations_run is not None
    assert 1 <= result.iterations_run <= 5


def test_iterative_reclassification_respects_cutoff_percentile() -> None:
    """A high cutoff percentile means a HIGH score threshold, so fewer rejects
    exceed it and the inferred default rate is lower."""
    approved, rejected, features = _make_populations()

    result_low = iterative_reclassification(
        approved, rejected, features,
        config=IterativeConfig(max_iterations=3, cutoff_percentile=10),
    )
    result_high = iterative_reclassification(
        approved, rejected, features,
        config=IterativeConfig(max_iterations=3, cutoff_percentile=80),
    )
    # Low percentile threshold → more rejects score above it → higher default rate
    assert result_low.reject_labels.mean() >= result_high.reject_labels.mean()


# ── Tests: fuzzy_augmentation ─────────────────────────────────────────────────

def test_fuzzy_augmentation_doubles_rejected_rows() -> None:
    approved, rejected, features = _make_populations()
    result = fuzzy_augmentation(approved, rejected, features)

    expected = len(approved) + 2 * len(rejected)
    assert len(result.augmented_data) == expected
    assert result.method == "fuzzy_augmentation"


def test_fuzzy_augmentation_weights_sum_to_one_per_reject() -> None:
    """For each rejected applicant the two fractional weights should sum to ~1."""
    approved, rejected, features = _make_populations(n_approved=100, n_rejected=50)
    result = fuzzy_augmentation(approved, rejected, features)

    aug = result.augmented_data
    # Only the reject rows have _sample_weight != 1.0
    reject_rows = aug[aug["approved_flag"] == 0]
    default_rows = reject_rows[reject_rows["default_flag"] == 1]
    non_default_rows = reject_rows[reject_rows["default_flag"] == 0]

    assert len(default_rows) == len(non_default_rows) == len(rejected)

    # Weights: default copy = P(default), non-default copy = 1 - P(default)
    # → they sum to 1.0 per applicant when default_weight == non_default_weight == 1.0
    total_weights = (
        default_rows["_sample_weight"].values
        + non_default_rows["_sample_weight"].values
    )
    np.testing.assert_allclose(total_weights, 1.0, atol=1e-6)


def test_fuzzy_augmentation_weights_in_range() -> None:
    approved, rejected, features = _make_populations()
    result = fuzzy_augmentation(approved, rejected, features)

    weights = result.augmented_data["_sample_weight"]
    assert (weights >= 0).all()
    assert (weights <= 1.0).all()


# ── Tests: apply_reject_inference (dispatch) ───────────────────────────────────

def test_apply_reject_inference_iterative() -> None:
    approved, rejected, features = _make_populations()
    result = apply_reject_inference(
        approved, rejected, features, method="iterative_reclassification"
    )
    assert result.method == "iterative_reclassification"


def test_apply_reject_inference_fuzzy() -> None:
    approved, rejected, features = _make_populations()
    result = apply_reject_inference(
        approved, rejected, features, method="fuzzy_augmentation"
    )
    assert result.method == "fuzzy_augmentation"


def test_apply_reject_inference_both() -> None:
    approved, rejected, features = _make_populations()
    result = apply_reject_inference(approved, rejected, features, method="both")
    assert result.method == "both"
    assert result.reject_weights is not None


def test_apply_reject_inference_invalid_method() -> None:
    approved, rejected, features = _make_populations()
    with pytest.raises(ValueError, match="Unknown reject-inference method"):
        apply_reject_inference(approved, rejected, features, method="unknown_method")
