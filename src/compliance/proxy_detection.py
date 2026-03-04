"""Proxy variable detection module for the AI Underwriting Engine.

Satisfies the Reg B / ECOA requirement to scrub impermissible proxy variables
from the feature set before model training.  A "proxy" is a feature that is
statistically associated with a protected characteristic (race, sex, national
origin, etc.) at a level that could lead to discriminatory outcomes.

Detection strategy
------------------
For every candidate feature, compute:

1. **Point-biserial / Pearson correlation** against each binary or continuous
   protected-class column.
2. **Cramér's V** for categorical feature × categorical protected-class pairs.
3. **Statistical significance test** (t-test / chi-squared) to determine whether
   the association is meaningful beyond sampling noise.

Features are flagged as proxies when their correlation with *any* protected-
class column exceeds ``correlation_threshold`` *and* the p-value is below
``significance_level``.

References
----------
* CFPB Circular 2022-03: Proxy variables and fair lending.
* Feldman, M. et al. (2015). Certifying and removing disparate impact.
  *KDD 2015*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ProxyDetectionConfig:
    """Configuration for proxy variable detection."""

    protected_class_columns: list[str] = field(default_factory=list)
    correlation_threshold: float = 0.4
    significance_level: float = 0.05


@dataclass
class ProxyFlag:
    """Record of a single detected proxy association."""

    feature: str
    protected_column: str
    correlation: float
    p_value: float
    test_type: str  # "pearson", "pointbiserial", or "cramers_v"


@dataclass
class ProxyDetectionResult:
    """Full output of the proxy-detection scan."""

    flagged_features: list[str]       # Features to exclude from modelling
    proxy_flags: list[ProxyFlag]      # Detailed per-association records
    clean_features: list[str]         # Features that passed the scan
    report: dict                      # Human-readable summary


# ── Statistical helpers ────────────────────────────────────────────────────────

def _cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Compute Cramér's V and its chi-squared p-value for two categorical series.

    Returns
    -------
    (cramers_v, p_value)
    """
    contingency = pd.crosstab(x, y)
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    n = contingency.sum().sum()
    k = min(contingency.shape) - 1
    if n == 0 or k == 0:
        return 0.0, 1.0
    v = float(np.sqrt(chi2 / (n * k)))
    return v, float(p_value)


def _correlation_and_pvalue(
    feature: pd.Series,
    protected: pd.Series,
) -> tuple[float, float, str]:
    """Compute the most appropriate correlation metric between *feature* and *protected*.

    - Both numeric → Pearson correlation + t-test p-value.
    - One binary → point-biserial correlation.
    - Both categorical → Cramér's V + chi-squared p-value.

    Returns
    -------
    (correlation, p_value, test_type)
    """
    feat_is_numeric = pd.api.types.is_numeric_dtype(feature)
    prot_is_numeric = pd.api.types.is_numeric_dtype(protected)

    # Drop rows where either value is NaN
    mask = feature.notna() & protected.notna()
    f = feature[mask]
    p = protected[mask]

    if len(f) < 5:
        return 0.0, 1.0, "insufficient_data"

    if feat_is_numeric and prot_is_numeric:
        corr, p_val = stats.pearsonr(f, p)
        return float(abs(corr)), float(p_val), "pearson"

    if feat_is_numeric and not prot_is_numeric:
        # Treat protected as grouping variable; use point-biserial
        unique_vals = p.unique()
        if len(unique_vals) == 2:
            binary = (p == unique_vals[0]).astype(int)
            corr, p_val = stats.pointbiserialr(binary, f)
            return float(abs(corr)), float(p_val), "pointbiserial"
        else:
            # Fallback: ANOVA F-test direction of association
            groups = [f[p == v].values for v in unique_vals]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) < 2:
                return 0.0, 1.0, "anova"
            f_stat, p_val = stats.f_oneway(*groups)
            # Normalise F-statistic to [0,1] range (approximate)
            eta_sq = f_stat / (f_stat + len(f) - len(unique_vals))
            return float(np.sqrt(max(0.0, eta_sq))), float(p_val), "anova"

    # Categorical feature vs. categorical or numeric protected
    v, p_val = _cramers_v(f.astype(str), p.astype(str))
    return v, p_val, "cramers_v"


# ── Main detection function ────────────────────────────────────────────────────

def detect_proxy_variables(
    df: pd.DataFrame,
    feature_columns: list[str],
    config: ProxyDetectionConfig | None = None,
) -> ProxyDetectionResult:
    """Scan *feature_columns* for statistical association with protected-class columns.

    Parameters
    ----------
    df:
        Full dataset (must contain both feature and protected-class columns).
    feature_columns:
        Candidate model features to evaluate.
    config:
        Detection configuration.

    Returns
    -------
    ProxyDetectionResult
        Lists of flagged and clean features, plus per-association detail.
    """
    cfg = config or ProxyDetectionConfig()

    present_protected = [c for c in cfg.protected_class_columns if c in df.columns]
    if not present_protected:
        logger.info(
            "No protected-class columns present in dataset — proxy scan skipped."
        )
        return ProxyDetectionResult(
            flagged_features=[],
            proxy_flags=[],
            clean_features=list(feature_columns),
            report={"message": "No protected-class columns present; scan skipped."},
        )

    logger.info(
        "Running proxy detection on %d feature(s) against %d protected column(s): %s",
        len(feature_columns),
        len(present_protected),
        present_protected,
    )

    proxy_flags: list[ProxyFlag] = []
    flagged_set: set[str] = set()

    for feat in feature_columns:
        if feat not in df.columns:
            continue
        for prot in present_protected:
            corr, p_val, test_type = _correlation_and_pvalue(df[feat], df[prot])

            if (
                corr >= cfg.correlation_threshold
                and p_val < cfg.significance_level
            ):
                logger.warning(
                    "PROXY DETECTED: '%s' ↔ '%s' | %s=%.3f, p=%.4f",
                    feat,
                    prot,
                    test_type,
                    corr,
                    p_val,
                )
                flagged_set.add(feat)
                proxy_flags.append(
                    ProxyFlag(
                        feature=feat,
                        protected_column=prot,
                        correlation=corr,
                        p_value=p_val,
                        test_type=test_type,
                    )
                )

    flagged = sorted(flagged_set)
    clean = [f for f in feature_columns if f not in flagged_set]

    report = {
        "total_features_scanned": len(feature_columns),
        "flagged_features": flagged,
        "clean_features": clean,
        "num_flagged": len(flagged),
        "num_clean": len(clean),
        "proxy_associations": [
            {
                "feature": pf.feature,
                "protected_column": pf.protected_column,
                "correlation": round(pf.correlation, 4),
                "p_value": round(pf.p_value, 6),
                "test_type": pf.test_type,
            }
            for pf in proxy_flags
        ],
    }

    if flagged:
        logger.warning(
            "%d feature(s) flagged as potential proxies and excluded from modelling: %s",
            len(flagged),
            flagged,
        )
    else:
        logger.info("Proxy scan complete — no proxy variables detected.")

    return ProxyDetectionResult(
        flagged_features=flagged,
        proxy_flags=proxy_flags,
        clean_features=clean,
        report=report,
    )
