"""Phase 1 pipeline orchestrator for the AI Underwriting Engine.

Ties together all Phase 1 components into a single, reproducible run:

  1. **Data Ingestion** — load, validate, and split historical lending data.
  2. **Reject Inference** — augment approved-only training data with inferred
     outcomes for the rejected population.
  3. **Proxy Detection** — scan feature columns for impermissible associations
     with protected-class variables.
  4. **Baseline Metrics** — evaluate legacy rules engine and record KS / AUC /
     Gini benchmarks that Phase 2 models must beat.

Usage
-----
    python -m src.training.pipeline \\
        --data-path data/historical_lending.parquet \\
        --config configs/phase1.yaml \\
        --output models/artifacts/

Or call directly from Python:

    from src.training.pipeline import run_phase1_pipeline
    result = run_phase1_pipeline(config_path="configs/phase1.yaml")
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from src.compliance.proxy_detection import (
    ProxyDetectionConfig,
    ProxyDetectionResult,
    detect_proxy_variables,
)
from src.training.baseline_metrics import (
    BaselineMetricsConfig,
    BaselineMetricsResult,
    compute_baseline_metrics,
)
from src.training.data_ingestion import (
    IngestionConfig,
    IngestionResult,
    generate_ingestion_report,
    ingest_data,
)
from src.training.reject_inference import (
    FuzzyConfig,
    IterativeConfig,
    RejectInferenceResult,
    apply_reject_inference,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Configuration helpers ──────────────────────────────────────────────────────

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to the ``.yaml`` configuration file.

    Returns
    -------
    dict
        Parsed YAML contents.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def _build_ingestion_config(cfg: dict[str, Any], data_path: str | None) -> IngestionConfig:
    data_cfg = cfg.get("data", {})
    return IngestionConfig(
        data_path=data_path or data_cfg.get("historical_data_path", ""),
        target_column=data_cfg.get("target_column", "default_flag"),
        approved_flag_column=data_cfg.get("approved_flag_column", "approved_flag"),
        application_id_column=data_cfg.get("application_id_column", "application_id"),
        protected_class_columns=data_cfg.get("protected_class_columns", []),
    )


def _build_reject_inference_args(cfg: dict[str, Any]) -> dict[str, Any]:
    ri_cfg = cfg.get("reject_inference", {})
    method = ri_cfg.get("method", "iterative_reclassification")

    iter_raw = ri_cfg.get("iterative", {})
    iterative_config = IterativeConfig(
        max_iterations=iter_raw.get("max_iterations", 10),
        convergence_threshold=iter_raw.get("convergence_threshold", 0.001),
        cutoff_percentile=iter_raw.get("cutoff_percentile", 30),
    )

    fuzzy_raw = ri_cfg.get("fuzzy", {})
    fuzzy_config = FuzzyConfig(
        default_weight=fuzzy_raw.get("default_weight", 0.5),
        non_default_weight=fuzzy_raw.get("non_default_weight", 0.5),
    )

    return {
        "method": method,
        "iterative_config": iterative_config,
        "fuzzy_config": fuzzy_config,
    }


def _build_proxy_config(cfg: dict[str, Any]) -> ProxyDetectionConfig:
    data_cfg = cfg.get("data", {})
    return ProxyDetectionConfig(
        protected_class_columns=data_cfg.get("protected_class_columns", []),
        correlation_threshold=data_cfg.get("proxy_correlation_threshold", 0.4),
        significance_level=data_cfg.get("proxy_significance_level", 0.05),
    )


def _build_baseline_config(cfg: dict[str, Any]) -> BaselineMetricsConfig:
    bm_cfg = cfg.get("baseline_metrics", {})
    data_cfg = cfg.get("data", {})
    return BaselineMetricsConfig(
        legacy_score_column=bm_cfg.get("legacy_score_column", "legacy_score"),
        target_column=data_cfg.get("target_column", "default_flag"),
        psi_bins=bm_cfg.get("psi_bins", 10),
        min_ks=bm_cfg.get("min_ks", 0.20),
        min_auc=bm_cfg.get("min_auc", 0.65),
    )


# ── Pipeline result ────────────────────────────────────────────────────────────

class Phase1PipelineResult:
    """Aggregated output of the Phase 1 pipeline run."""

    def __init__(
        self,
        ingestion: IngestionResult,
        reject_inference: RejectInferenceResult,
        proxy_detection: ProxyDetectionResult,
        baseline_metrics: BaselineMetricsResult,
        clean_feature_columns: list[str],
    ) -> None:
        self.ingestion = ingestion
        self.reject_inference = reject_inference
        self.proxy_detection = proxy_detection
        self.baseline_metrics = baseline_metrics
        self.clean_feature_columns = clean_feature_columns

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the full pipeline run."""
        return {
            "phase": 1,
            "ingestion": generate_ingestion_report(self.ingestion),
            "reject_inference": {
                "method": self.reject_inference.method,
                "augmented_rows": len(self.reject_inference.augmented_data),
                "iterations_run": self.reject_inference.iterations_run,
                "inferred_default_rate": (
                    float(self.reject_inference.reject_labels.mean())
                    if len(self.reject_inference.reject_labels) > 0
                    else None
                ),
            },
            "proxy_detection": self.proxy_detection.report,
            "baseline_metrics": self.baseline_metrics.report,
            "clean_feature_columns": self.clean_feature_columns,
        }


# ── Main pipeline function ─────────────────────────────────────────────────────

def run_phase1_pipeline(
    config_path: str | Path = "configs/phase1.yaml",
    data_path: str | None = None,
    output_dir: str | Path | None = None,
) -> Phase1PipelineResult:
    """Execute the full Phase 1: Foundation & Data Engineering pipeline.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.
    data_path:
        Override the data path from config (useful for CLI invocation).
    output_dir:
        Directory where artefacts (reports, augmented data) are saved.

    Returns
    -------
    Phase1PipelineResult
    """
    logger.info("=" * 70)
    logger.info("AI Underwriting Engine — Phase 1: Foundation & Data Engineering")
    logger.info("=" * 70)

    # ── Load configuration ─────────────────────────────────────────────────────
    cfg = load_config(config_path)
    out_dir = Path(output_dir or cfg.get("data", {}).get("output_dir", "models/artifacts/"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Data Ingestion ─────────────────────────────────────────────────
    logger.info("STEP 1 — Data Ingestion")
    ingestion_config = _build_ingestion_config(cfg, data_path)
    ingestion_result = ingest_data(ingestion_config)

    # ── Step 2: Reject Inference ───────────────────────────────────────────────
    logger.info("STEP 2 — Reject Inference")
    ri_args = _build_reject_inference_args(cfg)
    ri_result = apply_reject_inference(
        approved=ingestion_result.approved,
        rejected=ingestion_result.rejected,
        feature_columns=ingestion_result.feature_columns,
        method=ri_args["method"],
        target_column=ingestion_config.target_column,
        iterative_config=ri_args["iterative_config"],
        fuzzy_config=ri_args["fuzzy_config"],
    )

    # ── Step 3: Proxy Variable Detection ──────────────────────────────────────
    logger.info("STEP 3 — Proxy Variable Detection")
    proxy_config = _build_proxy_config(cfg)
    proxy_result = detect_proxy_variables(
        df=ri_result.augmented_data,
        feature_columns=ingestion_result.feature_columns,
        config=proxy_config,
    )

    # Remove any flagged proxies from the clean feature set
    clean_features = proxy_result.clean_features

    # ── Step 4: Baseline Metrics ───────────────────────────────────────────────
    logger.info("STEP 4 — Baseline Metrics")
    baseline_config = _build_baseline_config(cfg)
    baseline_result = compute_baseline_metrics(
        approved=ingestion_result.approved,
        config=baseline_config,
    )

    # ── Persist artefacts ──────────────────────────────────────────────────────
    pipeline_result = Phase1PipelineResult(
        ingestion=ingestion_result,
        reject_inference=ri_result,
        proxy_detection=proxy_result,
        baseline_metrics=baseline_result,
        clean_feature_columns=clean_features,
    )

    summary = pipeline_result.summary()
    report_path = out_dir / "phase1_report.json"
    with report_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Phase 1 report written to %s", report_path)

    # Save augmented dataset for Phase 2 model training
    augmented_path = out_dir / "phase1_augmented_data.parquet"
    ri_result.augmented_data.to_parquet(augmented_path, index=False)
    logger.info("Augmented dataset saved to %s", augmented_path)

    logger.info("=" * 70)
    logger.info("Phase 1 complete.")
    logger.info("=" * 70)

    return pipeline_result


# ── CLI entry point ────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Underwriting Engine — Phase 1: Foundation & Data Engineering"
    )
    parser.add_argument(
        "--config",
        default="configs/phase1.yaml",
        help="Path to the YAML configuration file (default: configs/phase1.yaml)",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override the data path from config.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for artefacts (default: from config).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the Phase 1 pipeline."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    result = run_phase1_pipeline(
        config_path=args.config,
        data_path=args.data_path,
        output_dir=args.output,
    )

    print("\n── Phase 1 Summary ──")
    print(json.dumps(result.summary(), indent=2, default=str))


if __name__ == "__main__":
    main()
