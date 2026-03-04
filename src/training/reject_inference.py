"""Reject-inference module for the AI Underwriting Engine.

Addresses the fundamental selection bias problem in credit modelling: historical
data only contains performance outcomes for *approved* applicants, yet the
production model must score *all* applicants including those who would have been
declined.

Two complementary reject-inference strategies are implemented:

1. **Iterative Reclassification** — trains an initial model on approved data,
   scores the rejected population, and iteratively re-classifies rejects as
   "default" or "non-default" based on score cutoffs, adding them to the
   training set.

2. **Fuzzy Augmentation** — assigns fractional (probability-based) weights to
   rejected observations rather than hard binary labels, allowing gradient-
   boosted models to learn from the full applicant population.

References
----------
* Anderson, R. (2007). *The Credit Scoring Toolkit*. Oxford University Press.
* Hand, D. & Henley, W. (1993). Can reject inference ever work?
  *IMA Journal of Mathematics Applied in Business & Industry*, 5, 45–55.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class IterativeConfig:
    """Hyper-parameters for iterative reclassification."""

    max_iterations: int = 10
    convergence_threshold: float = 0.001
    cutoff_percentile: int = 30  # Bottom N% of rejects → classified as defaults


@dataclass
class FuzzyConfig:
    """Hyper-parameters for fuzzy augmentation.

    Both weights default to 1.0 so that each rejected applicant contributes
    total sample weight 1.0 (= one rejected applicant's weight budget) split
    proportionally between a default copy (weight = P(default)) and a
    non-default copy (weight = 1 - P(default)).
    """

    default_weight: float = 1.0
    non_default_weight: float = 1.0


@dataclass
class RejectInferenceResult:
    """Output of a reject-inference run."""

    augmented_data: pd.DataFrame          # Full dataset including inferred rejects
    reject_labels: pd.Series              # Inferred labels for rejected applicants
    reject_weights: pd.Series | None      # Fuzzy weights (None for hard reclassification)
    iterations_run: int | None            # Iterations used (iterative method only)
    method: str


# ── Shared utilities ───────────────────────────────────────────────────────────

def _fit_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[LogisticRegression, StandardScaler]:
    """Fit a scaled logistic regression for reject scoring.

    A simple logistic regression is intentionally used here as the initial
    seed model — it is interpretable, fast, and avoids overfitting on the
    approved-only sample.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
    model.fit(X_scaled, y_train)
    return model, scaler


def _score_population(
    model: LogisticRegression,
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return P(default) for *X*."""
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]


# ── Iterative Reclassification ─────────────────────────────────────────────────

def iterative_reclassification(
    approved: pd.DataFrame,
    rejected: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "default_flag",
    config: IterativeConfig | None = None,
) -> RejectInferenceResult:
    """Apply iterative reclassification to infer labels for rejected applicants.

    Algorithm
    ---------
    1. Train a logistic regression on the approved population.
    2. Score all rejected applicants.
    3. Assign hard labels: rejects in the bottom *cutoff_percentile* of scores
       are labelled as defaults (1); the rest as non-defaults (0).
    4. Add newly labelled rejects to the training set and re-train.
    5. Repeat until the label assignment converges or *max_iterations* is
       reached.

    Parameters
    ----------
    approved:
        Approved applicants with observed *target_column* outcomes.
    rejected:
        Rejected applicants without observed outcomes.
    feature_columns:
        Names of the numeric feature columns used for modelling.
    target_column:
        Binary 0/1 default indicator column name.
    config:
        Iterative reclassification hyper-parameters.

    Returns
    -------
    RejectInferenceResult
    """
    cfg = config or IterativeConfig()

    X_approved = approved[feature_columns].fillna(approved[feature_columns].median())
    y_approved = approved[target_column]
    X_rejected = rejected[feature_columns].fillna(X_approved.median())

    # Initial labels: probabilistic seed before iteration
    reject_labels = pd.Series(
        np.zeros(len(rejected), dtype=int), index=rejected.index
    )
    prev_labels = reject_labels.copy()

    iterations_run = 0
    for iteration in range(1, cfg.max_iterations + 1):
        # Combine approved + currently-labelled rejects
        X_combined = pd.concat([X_approved, X_rejected], axis=0)
        y_combined = pd.concat([y_approved, reject_labels], axis=0)

        model, scaler = _fit_logistic(X_combined, y_combined)
        reject_scores = _score_population(model, scaler, X_rejected)

        # Hard cut: bottom percentile → default
        cutoff = np.percentile(reject_scores, cfg.cutoff_percentile)
        new_labels = pd.Series(
            (reject_scores >= cutoff).astype(int), index=rejected.index
        )

        # Check convergence
        label_change_rate = (new_labels != prev_labels).mean()
        logger.info(
            "Iteration %d/%d — label change rate: %.4f (threshold: %.4f)",
            iteration,
            cfg.max_iterations,
            label_change_rate,
            cfg.convergence_threshold,
        )

        reject_labels = new_labels
        iterations_run = iteration

        if label_change_rate < cfg.convergence_threshold:
            logger.info("Converged after %d iteration(s).", iteration)
            break

        prev_labels = reject_labels.copy()

    # Build augmented dataset
    rejected_augmented = rejected.copy()
    rejected_augmented[target_column] = reject_labels.values
    augmented = pd.concat([approved, rejected_augmented], axis=0).reset_index(drop=True)

    inferred_default_rate = reject_labels.mean()
    logger.info(
        "Iterative reclassification complete — inferred default rate for "
        "rejects: %.2f%%",
        inferred_default_rate * 100,
    )

    return RejectInferenceResult(
        augmented_data=augmented,
        reject_labels=reject_labels,
        reject_weights=None,
        iterations_run=iterations_run,
        method="iterative_reclassification",
    )


# ── Fuzzy Augmentation ─────────────────────────────────────────────────────────

def fuzzy_augmentation(
    approved: pd.DataFrame,
    rejected: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "default_flag",
    config: FuzzyConfig | None = None,
) -> RejectInferenceResult:
    """Apply fuzzy augmentation to create probability-weighted reject records.

    Instead of hard labels, each rejected applicant is duplicated: one copy
    carrying a *default_weight* contribution as a default (1), and another
    carrying a *non_default_weight* contribution as a non-default (0).

    A logistic regression trained on approved data provides the probability
    used to scale these weights per applicant.

    Parameters
    ----------
    approved:
        Approved applicants with observed outcomes.
    rejected:
        Rejected applicants without observed outcomes.
    feature_columns:
        Numeric feature column names.
    target_column:
        Binary 0/1 default indicator.
    config:
        Fuzzy augmentation hyper-parameters.

    Returns
    -------
    RejectInferenceResult
    """
    cfg = config or FuzzyConfig()

    X_approved = approved[feature_columns].fillna(approved[feature_columns].median())
    y_approved = approved[target_column]
    X_rejected = rejected[feature_columns].fillna(X_approved.median())

    model, scaler = _fit_logistic(X_approved, y_approved)
    reject_scores = _score_population(model, scaler, X_rejected)

    # Each rejected applicant contributes total weight 1.0 to the training set.
    # The weight is split between a "default" copy and a "non-default" copy
    # proportionally to the model's P(default).  cfg.default_weight and
    # cfg.non_default_weight act as scaling multipliers (both default to 1.0).
    default_weights = pd.Series(
        reject_scores * cfg.default_weight, index=rejected.index
    )
    non_default_weights = pd.Series(
        (1 - reject_scores) * cfg.non_default_weight, index=rejected.index
    )

    # Create two augmented copies of rejected records
    rejects_as_default = rejected.copy()
    rejects_as_default[target_column] = 1
    rejects_as_default["_sample_weight"] = default_weights.values

    rejects_as_non_default = rejected.copy()
    rejects_as_non_default[target_column] = 0
    rejects_as_non_default["_sample_weight"] = non_default_weights.values

    approved_copy = approved.copy()
    approved_copy["_sample_weight"] = 1.0

    augmented = pd.concat(
        [approved_copy, rejects_as_default, rejects_as_non_default], axis=0
    ).reset_index(drop=True)

    # Inferred labels: dominant class per reject based on score
    reject_labels = pd.Series(
        (reject_scores >= 0.5).astype(int), index=rejected.index
    )
    reject_weights = default_weights  # Primary weight series for reference

    logger.info(
        "Fuzzy augmentation complete — augmented dataset size: %d rows "
        "(approved=%d, reject copies=%d)",
        len(augmented),
        len(approved),
        len(rejected) * 2,
    )

    return RejectInferenceResult(
        augmented_data=augmented,
        reject_labels=reject_labels,
        reject_weights=reject_weights,
        iterations_run=None,
        method="fuzzy_augmentation",
    )


# ── Combined entry point ───────────────────────────────────────────────────────

def apply_reject_inference(
    approved: pd.DataFrame,
    rejected: pd.DataFrame,
    feature_columns: list[str],
    method: str = "iterative_reclassification",
    target_column: str = "default_flag",
    iterative_config: IterativeConfig | None = None,
    fuzzy_config: FuzzyConfig | None = None,
) -> RejectInferenceResult:
    """Dispatch to the selected reject-inference method.

    Parameters
    ----------
    approved:
        Approved applicants.
    rejected:
        Rejected applicants (no outcome labels).
    feature_columns:
        Numeric modelling features.
    method:
        One of ``"iterative_reclassification"``, ``"fuzzy_augmentation"``,
        or ``"both"`` (runs iterative first, returns fuzzy on the result).
    target_column:
        Default indicator column.
    iterative_config:
        Hyper-parameters for iterative reclassification.
    fuzzy_config:
        Hyper-parameters for fuzzy augmentation.

    Returns
    -------
    RejectInferenceResult
    """
    if method == "iterative_reclassification":
        return iterative_reclassification(
            approved, rejected, feature_columns, target_column, iterative_config
        )
    elif method == "fuzzy_augmentation":
        return fuzzy_augmentation(
            approved, rejected, feature_columns, target_column, fuzzy_config
        )
    elif method == "both":
        # Run iterative first to produce hard labels, then apply fuzzy on top
        iter_result = iterative_reclassification(
            approved, rejected, feature_columns, target_column, iterative_config
        )
        # For "both", return the iterative result with fuzzy weights appended
        fuzzy_result = fuzzy_augmentation(
            approved, rejected, feature_columns, target_column, fuzzy_config
        )
        combined = iter_result.augmented_data.copy()
        return RejectInferenceResult(
            augmented_data=combined,
            reject_labels=iter_result.reject_labels,
            reject_weights=fuzzy_result.reject_weights,
            iterations_run=iter_result.iterations_run,
            method="both",
        )
    else:
        raise ValueError(
            f"Unknown reject-inference method: '{method}'. "
            "Choose from 'iterative_reclassification', 'fuzzy_augmentation', 'both'."
        )
