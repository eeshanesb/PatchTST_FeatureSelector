from __future__ import annotations

import argparse
import math
import sys
from typing import Sequence

from .core import (
    CONFIG_DEFAULT_PATH,
    FeatureSelectorSettings,
    derive_defaults,
    extract_config_path,
    load_yaml_config,
    run_feature_selection,
)


def build_parser(defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PatchTST feature selector via correlation screening and exhaustive search."
    )
    parser.add_argument(
        "--config",
        default=defaults.get("config_path", CONFIG_DEFAULT_PATH),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data-dir",
        default=defaults.get("data_dir"),
        help="Directory containing ticker CSV files (e.g., rdata or HFdata).",
    )
    parser.add_argument(
        "--target",
        default=defaults.get("target"),
        required=defaults.get("target") is None,
        help="Ticker symbol to forecast (case-insensitive).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=defaults.get("top_n"),
        help="Number of correlated tickers to keep for exhaustive search.",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=defaults.get("min_overlap"),
        help="Minimum overlapping return observations required when computing correlations.",
    )
    parser.add_argument(
        "--combo-max-size",
        type=int,
        default=defaults.get("combo_max_size"),
        help="Optional cap on feature combination size (defaults to top-N).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=defaults.get("horizon"),
        help="Validation horizon (forecast steps) and val_size during training.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=defaults.get("input_size"),
        help="PatchTST input window length.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=defaults.get("max_steps"),
        help="Maximum training steps for PatchTST.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=defaults.get("early_stop_patience"),
        help="Early stopping patience in steps (set 0 or negative to disable).",
    )
    parser.add_argument(
        "--val-check-steps",
        type=int,
        default=defaults.get("val_check_steps"),
        help="Validation frequency in steps.",
    )
    parser.add_argument(
        "--train-last-n",
        type=int,
        default=defaults.get("train_last_n"),
        help="If set, use only the last N observations per series for training.",
    )
    parser.add_argument(
        "--freq",
        default=defaults.get("freq"),
        help="Override inferred frequency (e.g., D, B, 15min).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults.get("batch_size"),
        help="Mini-batch size for PatchTST.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=defaults.get("learning_rate"),
        help="Learning rate for PatchTST optimizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.get("seed"),
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--export-summary",
        default=defaults.get("export_summary"),
        help="Optional CSV path to store evaluation summary.",
    )
    parser.add_argument(
        "--silent",
        dest="silent",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("silent", False),
        help="Suppress console output (still returns programmatic results).",
    )
    return parser


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    argv = list(argv) if argv is not None else None
    config_path = extract_config_path(argv)
    config = load_yaml_config(config_path)
    defaults = derive_defaults(config)
    defaults["config_path"] = config_path
    parser = build_parser(defaults)
    args = parser.parse_args(argv)
    if not args.target:
        parser.error("Target ticker must be specified.")
    return args


def to_settings(args: argparse.Namespace) -> FeatureSelectorSettings:
    export_summary = args.export_summary
    if isinstance(export_summary, str) and export_summary.lower() in {"", "none", "null"}:
        export_summary = None

    return FeatureSelectorSettings(
        data_dir=args.data_dir,
        target=args.target,
        top_n=args.top_n,
        min_overlap=args.min_overlap,
        combo_max_size=args.combo_max_size,
        horizon=args.horizon,
        input_size=args.input_size,
        max_steps=args.max_steps,
        early_stop_patience=args.early_stop_patience,
        val_check_steps=args.val_check_steps,
        train_last_n=args.train_last_n,
        freq=args.freq,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        export_summary=export_summary,
        silent=args.silent,
        config_path=args.config,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    settings = to_settings(args)

    if not settings.silent:
        print(f"Loaded configuration from {args.config}")

    result = run_feature_selection(settings)

    summary = result.summary
    best = result.best_result
    if not settings.silent:
        print("\nBest combination:")
        best_desc = ",".join(best.features) if best.features else "(target only)"
        print(f"  Features: {best_desc}")
        print(f"  MAPE: {best.mape:.4f}%")
        print(f"  Runtime (s): {best.runtime_sec:.1f}")
        print("\nFull summary:")
        print(
            summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
                if isinstance(x, float) and not math.isnan(x)
                else str(x),
            )
        )

    if settings.export_summary:
        summary.to_csv(settings.export_summary, index=False)
        if not settings.silent:
            print(f"\nSummary exported to {settings.export_summary}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user.")
