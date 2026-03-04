"""Baseline metrics module for the AI Underwriting Engine.

Establishes quantitative performance benchmarks for Phase 1 by:

1. Evaluating the **legacy rules engine** score column on the approved population.
2. Reporting key credit-model performance statistics:
   - **KS (Kolmogorov-Smirnov) statistic** — separation between default and
     non-default score distributions.
   - **AUC-ROC** — area under the receiver operating characteristic curve.
   - **Gini coefficient** — 2×AUC − 1, a common credit-industry metric.
   - **Population Stability Index (PSI)** — measures score-distribution
     stability between two populations (e.g., training vs. out-of-time).

These metrics form the acceptance gate before Phase 2 model development:
new ensemble models must demonstrate statistically significant lift over the
legacy baseline on all three primary metrics (KS, AUC, Gini).

References
----------
* Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.
* Yeh, I.-C. & Lien, C.-h. (2009). The comparisons of data mining techniques for
  the predictive accuracy of probability of default. *Expert Systems with
  Applications*, 36(2), 2473–2480.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


@dataclass
class BaselineMetricsConfig:
    """Configuration for baseline metrics computation."""

    legacy_score_column: str = "legacy_score"
    target_column: str = "default_flag"
    psi_bins: int = 10
    min_ks: float = 0.20
    min_auc: float = 0.65


@dataclass
class PerformanceMetrics:
    """Container for a single set of model performance metrics."""

    ks_statistic: float
    ks_threshold: float          # Score threshold at maximum KS separation
    auc_roc: float
    gini: float
    default_rate: float
    num_observations: int
    source: str                  # Label for this metrics set (e.g. "legacy")


@dataclass
class PSIResult:
    """Population Stability Index result."""

    psi_total: float
    bin_psi: list[float]
    interpretation: str          # "stable" | "moderate_shift" | "significant_shift"


@dataclass
class BaselineMetricsResult:
    """Full output of the baseline metrics computation."""

    legacy_metrics: PerformanceMetrics | None
    psi: PSIResult | None
    meets_minimum_ks: bool | None
    meets_minimum_auc: bool | None
    report: dict


# ── Metric calculations ────────────────────────────────────────────────────────

def compute_ks_statistic(
    y_true: pd.Series, scores: pd.Series
) -> tuple[float, float]:
    """Compute the Kolmogorov-Smirnov statistic.

    Parameters
    ----------
    y_true:
        Binary default labels (1 = default, 0 = non-default).
    scores:
        Predicted default probability or score (higher = higher risk).

    Returns
    -------
    (ks_statistic, ks_threshold)
        KS value in [0, 1] and the score threshold that achieves it.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    ks_values = tpr - fpr
    idx = int(np.argmax(ks_values))
    ks_stat = float(ks_values[idx])
    ks_threshold = float(thresholds[idx])
    return ks_stat, ks_threshold


def compute_auc(y_true: pd.Series, scores: pd.Series) -> float:
    """Compute AUC-ROC."""
    return float(roc_auc_score(y_true, scores))


def compute_gini(auc: float) -> float:
    """Compute Gini coefficient from AUC: Gini = 2 × AUC − 1."""
    return 2.0 * auc - 1.0


def compute_psi(
    reference: pd.Series,
    comparison: pd.Series,
    n_bins: int = 10,
) -> PSIResult:
    """Compute Population Stability Index between two score distributions.

    PSI < 0.1  → stable
    PSI < 0.25 → moderate shift (monitor)
    PSI ≥ 0.25 → significant shift (investigate)

    Parameters
    ----------
    reference:
        Baseline score distribution (e.g., training set).
    comparison:
        New score distribution to compare against (e.g., validation set).
    n_bins:
        Number of equal-width bins.

    Returns
    -------
    PSIResult
    """
    eps = 1e-8  # Avoid log(0)

    # Build bins from the reference distribution
    _, bin_edges = np.histogram(reference.dropna(), bins=n_bins)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference.dropna(), bins=bin_edges)
    cmp_counts, _ = np.histogram(comparison.dropna(), bins=bin_edges)

    ref_pct = ref_counts / (ref_counts.sum() + eps)
    cmp_pct = cmp_counts / (cmp_counts.sum() + eps)

    ref_pct = np.clip(ref_pct, eps, None)
    cmp_pct = np.clip(cmp_pct, eps, None)

    bin_psi = (cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)
    psi_total = float(bin_psi.sum())

    if psi_total < 0.1:
        interpretation = "stable"
    elif psi_total < 0.25:
        interpretation = "moderate_shift"
    else:
        interpretation = "significant_shift"

    return PSIResult(
        psi_total=psi_total,
        bin_psi=bin_psi.tolist(),
        interpretation=interpretation,
    )


def compute_performance_metrics(
    df: pd.DataFrame,
    score_column: str,
    target_column: str = "default_flag",
    source: str = "model",
) -> PerformanceMetrics:
    """Compute KS, AUC, and Gini for a given score column.

    Parameters
    ----------
    df:
        DataFrame containing *score_column* and *target_column*.
    score_column:
        Name of the score / probability column.
    target_column:
        Binary default label column.
    source:
        Descriptive label for this metric set.

    Returns
    -------
    PerformanceMetrics
    """
    if score_column not in df.columns:
        raise KeyError(f"Score column '{score_column}' not found in DataFrame.")
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame.")

    clean = df[[score_column, target_column]].dropna()
    if len(clean) < 10:
        raise ValueError(
            f"Insufficient data to compute metrics ({len(clean)} rows after "
            "dropping NaN)."
        )

    y_true = clean[target_column].astype(int)
    scores = clean[score_column].astype(float)

    ks_stat, ks_thresh = compute_ks_statistic(y_true, scores)
    auc = compute_auc(y_true, scores)
    gini = compute_gini(auc)
    default_rate = float(y_true.mean())

    return PerformanceMetrics(
        ks_statistic=round(ks_stat, 4),
        ks_threshold=round(ks_thresh, 4),
        auc_roc=round(auc, 4),
        gini=round(gini, 4),
        default_rate=round(default_rate, 4),
        num_observations=len(clean),
        source=source,
    )


# ── Main entry point ───────────────────────────────────────────────────────────

def compute_baseline_metrics(
    approved: pd.DataFrame,
    config: BaselineMetricsConfig | None = None,
    comparison_df: pd.DataFrame | None = None,
) -> BaselineMetricsResult:
    """Compute baseline metrics for the legacy rules engine score.

    Parameters
    ----------
    approved:
        Approved applicants with observed default labels and (optionally) a
        legacy score column.
    config:
        Metric computation configuration.
    comparison_df:
        Optional second population for PSI calculation (e.g., out-of-time
        validation set).

    Returns
    -------
    BaselineMetricsResult
    """
    cfg = config or BaselineMetricsConfig()

    # ── Legacy engine metrics ──────────────────────────────────────────────────
    legacy_metrics: PerformanceMetrics | None = None
    meets_ks: bool | None = None
    meets_auc: bool | None = None

    if cfg.legacy_score_column in approved.columns:
        legacy_metrics = compute_performance_metrics(
            approved,
            score_column=cfg.legacy_score_column,
            target_column=cfg.target_column,
            source="legacy_rules_engine",
        )
        meets_ks = legacy_metrics.ks_statistic >= cfg.min_ks
        meets_auc = legacy_metrics.auc_roc >= cfg.min_auc

        logger.info(
            "Legacy engine — KS=%.4f (%s), AUC=%.4f (%s), Gini=%.4f",
            legacy_metrics.ks_statistic,
            "✓" if meets_ks else "✗",
            legacy_metrics.auc_roc,
            "✓" if meets_auc else "✗",
            legacy_metrics.gini,
        )
    else:
        logger.warning(
            "Legacy score column '%s' not found — skipping legacy baseline.",
            cfg.legacy_score_column,
        )

    # ── PSI (optional) ─────────────────────────────────────────────────────────
    psi_result: PSIResult | None = None
    if (
        comparison_df is not None
        and cfg.legacy_score_column in approved.columns
        and cfg.legacy_score_column in comparison_df.columns
    ):
        psi_result = compute_psi(
            approved[cfg.legacy_score_column].dropna(),
            comparison_df[cfg.legacy_score_column].dropna(),
            n_bins=cfg.psi_bins,
        )
        logger.info(
            "PSI = %.4f (%s)",
            psi_result.psi_total,
            psi_result.interpretation,
        )

    # ── Summary report ─────────────────────────────────────────────────────────
    report: dict = {
        "legacy_metrics": (
            {
                "ks_statistic": legacy_metrics.ks_statistic,
                "ks_threshold": legacy_metrics.ks_threshold,
                "auc_roc": legacy_metrics.auc_roc,
                "gini": legacy_metrics.gini,
                "default_rate": legacy_metrics.default_rate,
                "num_observations": legacy_metrics.num_observations,
            }
            if legacy_metrics
            else None
        ),
        "meets_minimum_ks": meets_ks,
        "meets_minimum_auc": meets_auc,
        "psi": (
            {
                "psi_total": psi_result.psi_total,
                "interpretation": psi_result.interpretation,
            }
            if psi_result
            else None
        ),
    }

    return BaselineMetricsResult(
        legacy_metrics=legacy_metrics,
        psi=psi_result,
        meets_minimum_ks=meets_ks,
        meets_minimum_auc=meets_auc,
        report=report,
    )
