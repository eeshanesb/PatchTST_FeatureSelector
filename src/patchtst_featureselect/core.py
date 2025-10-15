from __future__ import annotations

import itertools
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "PyYAML is required to load configuration. Please install it via 'pip install pyyaml'."
    ) from exc

os.environ.setdefault("PYTORCH_LIGHTNING_DISABLE_PROGRESS_BAR", "1")


CONFIG_DEFAULT_PATH = "config.yaml"

DATE_CANDIDATES = (
    "ds",
    "DS",
    "date",
    "Date",
    "datetime",
    "Datetime",
    "timestamp",
    "Timestamp",
)
PRICE_CANDIDATES = (
    "Close",
    "close",
    "Adj Close",
    "AdjClose",
    "adj_close",
    "CLOSE",
    "PX_LAST",
    "price",
    "Price",
    "y",
)


def normalize_datetimes(series: pd.Series) -> pd.Series:
    """Return timezone-naive timestamps for robust merges."""
    values = pd.to_datetime(series, errors="coerce")
    try:
        values = values.dt.tz_convert(None)
    except (TypeError, AttributeError, ValueError):
        try:
            values = values.dt.tz_localize(None)
        except (TypeError, AttributeError, ValueError):
            pass
    return values


Logger = Callable[[str], None] | None


@dataclass
class ComboResult:
    """Summary of a PatchTST evaluation for one feature combination."""

    features: Tuple[str, ...]
    tickers: Tuple[str, ...]
    mape: float
    runtime_sec: float
    detail: pd.DataFrame | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class FeatureSelectorSettings:
    """Configuration for running the feature selector."""

    data_dir: str
    target: str
    top_n: int = 5
    min_overlap: int = 60
    combo_max_size: int | None = None
    horizon: int = 7
    input_size: int = 30
    max_steps: int = 50
    early_stop_patience: int = 20
    val_check_steps: int = 1
    train_last_n: int | None = None
    freq: str | None = None
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 42
    export_summary: str | None = None
    silent: bool = False
    config_path: str | None = None


@dataclass
class FeatureSelectionResult:
    """Container for the entire feature-selection sweep."""

    target: str
    frequency: str
    correlations: List[Tuple[str, float, int]]
    combinations: List[ComboResult]
    summary: pd.DataFrame
    best_result: ComboResult


def extract_config_path(argv: Sequence[str] | None = None) -> str:
    """Extract configuration path from CLI arguments without triggering full parse."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        default=CONFIG_DEFAULT_PATH,
        help="Path to YAML configuration file.",
    )
    args, _ = parser.parse_known_args(argv)
    return args.config


def load_yaml_config(path: str) -> Dict[str, object]:
    """Load YAML configuration as a dictionary."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Configuration file not found at {path}. Please create it using "
            "the settings from train_model.py and time_forecast.py."
        )
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Configuration file {path} must contain a YAML mapping.")
    return data


def derive_defaults(config: Dict[str, object]) -> Dict[str, object]:
    """Derive CLI defaults from a shared configuration mapping."""

    def _normalize(cfg: Dict[str, object] | None) -> Dict[str, object]:
        cfg = cfg or {}
        return {
            (k.upper() if isinstance(k, str) else k): v
            for k, v in cfg.items()
        }

    def _get(cfg: Dict[str, object], key: str, fallback: object = None) -> object:
        return cfg.get(key) if cfg.get(key) is not None else fallback

    train_cfg = _normalize(config.get("train_model"))
    forecast_cfg = _normalize(config.get("time_forecast"))
    feature_cfg = _normalize(config.get("featureselect"))

    defaults = {
        "data_dir": _get(train_cfg, "DATA_DIR", "rdata"),
        "top_n": int(_get(feature_cfg, "TOP_N", 5)),
        "min_overlap": int(_get(feature_cfg, "MIN_OVERLAP", 60)),
        "combo_max_size": (
            None
            if _get(feature_cfg, "COMBO_MAX_SIZE") in (None, "", "null")
            else int(_get(feature_cfg, "COMBO_MAX_SIZE"))
        ),
        "horizon": int(
            _get(train_cfg, "FORECAST_HORIZON", _get(forecast_cfg, "HORIZON", 7))
        ),
        "input_size": int(_get(train_cfg, "INPUT_SIZE", 30)),
        "max_steps": int(_get(train_cfg, "MAX_STEPS", 50)),
        "early_stop_patience": int(_get(feature_cfg, "EARLY_STOP_PATIENCE", 20)),
        "val_check_steps": int(_get(feature_cfg, "VAL_CHECK_STEPS", 1)),
        "train_last_n": (
            None
            if _get(train_cfg, "TRAIN_LAST_N_DAYS") in (None, "", "null")
            else int(_get(train_cfg, "TRAIN_LAST_N_DAYS"))
        ),
        "freq": _get(feature_cfg, "FREQ"),
        "batch_size": int(_get(feature_cfg, "BATCH_SIZE", 32)),
        "learning_rate": float(_get(feature_cfg, "LEARNING_RATE", 1e-3)),
        "seed": int(_get(feature_cfg, "SEED", 42)),
        "export_summary": _get(feature_cfg, "EXPORT_SUMMARY"),
        "silent": bool(_get(feature_cfg, "SILENT", False)),
        "target": _get(forecast_cfg, "TICKER"),
    }
    return defaults


def infer_ticker_from_filename(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    lowered = base.lower()
    for suffix in ("_daily", "_15m", "_30m", "_1h", "_hour", "_hourly"):
        if lowered.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.upper()


def read_close_series(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    date_col = next((col for col in DATE_CANDIDATES if col in raw.columns), None)
    if date_col is None:
        raise ValueError("missing recognizable date column")
    price_col = next((col for col in PRICE_CANDIDATES if col in raw.columns), None)
    if price_col is None:
        raise ValueError("missing recognizable close/price column")

    df = raw[[date_col, price_col]].copy()
    df["ds"] = pd.to_datetime(df[date_col], errors="coerce")
    df["y"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).drop_duplicates(subset="ds")
    df = df.sort_values("ds").reset_index(drop=True)[["ds", "y"]]
    if df.empty or df["y"].nunique() < 2:
        raise ValueError("insufficient variation after cleaning")
    return df


def load_close_series_map(data_dir: str, logger: Logger = None) -> Dict[str, pd.DataFrame]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    warn = logger or print
    series_map: Dict[str, pd.DataFrame] = {}
    for entry in os.listdir(data_dir):
        if not entry.lower().endswith(".csv"):
            continue
        path = os.path.join(data_dir, entry)
        ticker = infer_ticker_from_filename(entry)
        try:
            series = read_close_series(path)
        except Exception as exc:  # noqa: BLE001
            warn(f"⚠️  Skipping {entry}: {exc}")
            continue
        if ticker in series_map:
            warn(f"⚠️  Duplicate ticker detected ({ticker}); keeping first occurrence.")
            continue
        series_map[ticker] = series
    if not series_map:
        raise RuntimeError(f"No valid CSV files found in {data_dir}")
    return series_map


def infer_frequency(ds: pd.Series) -> str:
    ordered = ds.sort_values()
    freq = pd.infer_freq(ordered)
    if freq:
        return freq
    diffs = ordered.diff().dropna()
    if diffs.empty:
        return "D"
    try:
        mode_delta = diffs.mode().iloc[0]
    except IndexError:
        mode_delta = diffs.iloc[0]
    try:
        from pandas.tseries.frequencies import to_offset

        offset = to_offset(mode_delta)
        if offset is not None and offset.nanos > 0:
            return offset.freqstr
    except Exception:  # noqa: BLE001
        pass
    seconds = mode_delta.total_seconds() if hasattr(mode_delta, "total_seconds") else None
    if seconds is None or seconds <= 0:
        return "D"
    if seconds >= 43200:
        return "D"
    if seconds % 3600 == 0:
        hours = int(seconds / 3600)
        return f"{hours}H"
    if seconds % 60 == 0:
        minutes = int(seconds / 60)
        return f"{minutes}min"
    return "D"


def compute_return_correlations(
    target: str,
    series_map: Dict[str, pd.DataFrame],
    min_overlap: int,
) -> List[Tuple[str, float, int]]:
    target_df = series_map[target]
    target_returns = target_df.set_index("ds")["y"].pct_change().dropna()

    correlations: List[Tuple[str, float, int]] = []
    for ticker, df in series_map.items():
        if ticker == target:
            continue
        other_returns = df.set_index("ds")["y"].pct_change().dropna()
        joined = pd.concat([target_returns, other_returns], axis=1, join="inner").dropna()
        overlap = joined.shape[0]
        if overlap < min_overlap:
            continue
        corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
        if pd.isna(corr):
            continue
        correlations.append((ticker, corr, overlap))
    correlations.sort(key=lambda item: abs(item[1]), reverse=True)
    return correlations


def subset_series(series: pd.DataFrame, train_last_n: int | None) -> pd.DataFrame:
    if train_last_n is None:
        return series
    return series.tail(train_last_n).reset_index(drop=True)


def build_panel(
    tickers: Sequence[str],
    series_map: Dict[str, pd.DataFrame],
    train_last_n: int | None,
) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        frame = subset_series(series_map[ticker], train_last_n)
        tmp = frame.copy()
        tmp["unique_id"] = f"{ticker}_CLOSE"
        frames.append(tmp[["unique_id", "ds", "y"]])
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return panel


def ensure_enough_history(panel: pd.DataFrame, target_uid: str, input_size: int, horizon: int) -> None:
    target_history = panel[panel["unique_id"] == target_uid]
    if target_history.shape[0] < input_size + horizon:
        raise ValueError(
            f"Not enough history for target ({target_uid}); "
            f"need at least {input_size + horizon} rows, found {target_history.shape[0]}."
        )


def configure_patch_model(
    horizon: int,
    input_size: int,
    max_steps: int,
    early_stop_patience: int,
    val_check_steps: int,
    batch_size: int,
    learning_rate: float,
) -> PatchTST:
    model = PatchTST(
        h=horizon,
        input_size=input_size,
        max_steps=max_steps,
        early_stop_patience_steps=early_stop_patience,
        val_check_steps=val_check_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    trainer_kwargs = getattr(model, "trainer_kwargs", {}).copy()
    trainer_kwargs.update(
        {
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "accelerator": "cpu",
            "devices": 1,
        }
    )
    model.trainer_kwargs = trainer_kwargs
    if hasattr(model, "early_stop_patience_steps") and early_stop_patience <= 0:
        model.early_stop_patience_steps = -1
    return model


def detect_prediction_column(preds: pd.DataFrame) -> str:
    candidates = [col for col in preds.columns if col not in {"unique_id", "ds", "cutoff"}]
    if not candidates:
        raise RuntimeError("Prediction output missing expected columns.")
    return candidates[0]


def compute_mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_arr = np.asarray(list(actual), dtype=float)
    pred_arr = np.asarray(list(predicted), dtype=float)
    mask = ~np.isnan(actual_arr) & ~np.isnan(pred_arr)
    if not mask.any():
        return float("inf")
    actual_used = actual_arr[mask]
    pred_used = pred_arr[mask]
    denom = np.where(np.abs(actual_used) < 1e-8, 1e-8, actual_used)
    absolute_percentage_errors = np.abs((actual_used - pred_used) / denom) * 100.0
    return float(np.mean(absolute_percentage_errors))


def evaluate_combination(
    target: str,
    features: Sequence[str],
    series_map: Dict[str, pd.DataFrame],
    freq: str,
    settings: FeatureSelectorSettings,
) -> ComboResult:
    start = time.time()
    tickers = tuple([target, *[feature.upper() for feature in features]])

    try:
        panel = build_panel(tickers, series_map, settings.train_last_n)
        target_uid = f"{target}_CLOSE"
        ensure_enough_history(panel, target_uid, settings.input_size, settings.horizon)

        model = configure_patch_model(
            horizon=settings.horizon,
            input_size=settings.input_size,
            max_steps=settings.max_steps,
            early_stop_patience=settings.early_stop_patience,
            val_check_steps=settings.val_check_steps,
            batch_size=settings.batch_size,
            learning_rate=settings.learning_rate,
        )

        nf = NeuralForecast(models=[model], freq=freq)
        val_size = max(1, settings.horizon)
        nf.fit(panel, val_size=val_size)

        preds = nf.predict()
        target_preds = preds[preds["unique_id"] == target_uid]
        if target_preds.shape[0] < settings.horizon:
            # Fallback to fitted (in-sample) predictions when out-of-sample horizon is unavailable.
            preds = nf.predict(fitted=True)
            target_preds = preds[preds["unique_id"] == target_uid]

        pred_col = detect_prediction_column(preds)
        tail_actual_full = panel[panel["unique_id"] == target_uid].copy()
        tail_actual_full["ds"] = normalize_datetimes(tail_actual_full["ds"])
        tail_actual_full = tail_actual_full.sort_values("ds")

        tail_pred_full = target_preds.copy()
        tail_pred_full["ds"] = normalize_datetimes(tail_pred_full["ds"])
        tail_pred_full = tail_pred_full.sort_values("ds").rename(columns={pred_col: "yhat"})

        merged = tail_actual_full.merge(tail_pred_full[["ds", "yhat"]], on="ds", how="inner")
        merged = merged.dropna(subset=["y", "yhat"])
        merged = merged.sort_values("ds").tail(settings.horizon)
        eval_horizon = merged.shape[0]

        fallback_detail = None
        if eval_horizon <= 0:
            actual_series = tail_actual_full.dropna(subset=["y"]).tail(settings.horizon)
            pred_series = tail_pred_full.dropna(subset=["yhat"]).tail(settings.horizon)
            align = min(actual_series.shape[0], pred_series.shape[0])
            if align > 0:
                actual_series = actual_series.tail(align)
                pred_series = pred_series.tail(align)
                merged = pd.DataFrame(
                    {
                        "ds": actual_series["ds"].reset_index(drop=True),
                        "y": actual_series["y"].reset_index(drop=True),
                        "yhat": pred_series["yhat"].reset_index(drop=True),
                    }
                )
                eval_horizon = align
                fallback_detail = "Aligned predictions by order due to timestamp mismatch."
            else:
                return ComboResult(
                    features=tuple(features),
                    tickers=tickers,
                    mape=float("inf"),
                    runtime_sec=time.time() - start,
                    detail=merged,
                    error="Predictions could not be aligned with actual values.",
                )

        mape = compute_mape(merged["y"].values, merged["yhat"].values)
        runtime = time.time() - start
        detail = merged
        if eval_horizon < settings.horizon:
            detail = merged.copy()
            detail.attrs["warning"] = (
                f"Evaluated on {eval_horizon} of requested {settings.horizon} steps "
                "due to limited predictions."
            )
        if fallback_detail:
            detail = detail.copy()
            detail.attrs["alignment"] = fallback_detail

        return ComboResult(
            features=tuple(features),
            tickers=tickers,
            mape=mape,
            runtime_sec=runtime,
            detail=detail,
            error=None,
        )

    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        return ComboResult(
            features=tuple(features),
            tickers=tickers,
            mape=float("inf"),
            runtime_sec=time.time() - start,
            detail=None,
            error=str(exc),
        )


def build_feature_combinations(
    candidates: Sequence[str],
    max_size: int | None,
) -> List[Tuple[str, ...]]:
    if not candidates:
        return [tuple()]
    cap = max_size if max_size is not None else len(candidates)
    cap = min(cap, len(candidates))
    combos: List[Tuple[str, ...]] = [tuple()]
    for size in range(1, cap + 1):
        combos.extend(itertools.combinations(candidates, size))
    return combos


def format_correlation_summary(
    target: str,
    correlations: List[Tuple[str, float, int]],
    top_n: int,
) -> List[str]:
    if not correlations:
        return [
            "⚠️  No correlated tickers met the overlap requirement; continuing with target alone."
        ]
    lines = [f"Top {min(top_n, len(correlations))} correlations for {target}:"]
    for rank, (ticker, corr, overlap) in enumerate(correlations[:top_n], start=1):
        lines.append(f"  {rank:>2}. {ticker:<15} corr={corr:>7.4f} | overlap={overlap}")
    return lines


def summarise_results(results: List[ComboResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        rows.append(
            {
                "features": ",".join(res.features) if res.features else "(target only)",
                "num_features": len(res.features),
                "tickers": ",".join(res.tickers),
                "mape": res.mape,
                "runtime_sec": res.runtime_sec,
                "status": "ok" if res.ok else f"error: {res.error}",
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("mape", na_position="last").reset_index(drop=True)
    return df


def _resolve_logger(logger: Logger, silent: bool) -> Callable[[str], None]:
    if silent:
        return lambda *_: None
    return logger or print


def _score_result(result: ComboResult) -> float:
    if result.error is not None:
        return float("inf")
    if math.isnan(result.mape):
        return float("inf")
    return result.mape


def run_feature_selection(
    settings: FeatureSelectorSettings,
    logger: Logger = print,
) -> FeatureSelectionResult:
    """Execute the feature selection sweep and return detailed results."""

    log = _resolve_logger(logger, settings.silent)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)

    target = settings.target.upper()
    series_map = load_close_series_map(settings.data_dir, logger=log)
    if target not in series_map:
        raise KeyError(f"Target ticker {target} not found in {settings.data_dir}.")

    freq = settings.freq or infer_frequency(series_map[target]["ds"])
    log(f"Using frequency '{freq}' for NeuralForecast.")

    correlations = compute_return_correlations(target, series_map, settings.min_overlap)
    for line in format_correlation_summary(target, correlations, settings.top_n):
        log(line)
    candidates = [ticker for ticker, _, _ in correlations[: settings.top_n]]

    combos = build_feature_combinations(candidates, settings.combo_max_size)
    log(f"Evaluating {len(combos)} combinations (including baseline target-only run).")

    results: List[ComboResult] = []
    for idx, combo in enumerate(combos, start=1):
        feature_list = list(combo)
        label = " + ".join([target, *feature_list]) if feature_list else target
        log(f"[{idx}/{len(combos)}] Training PatchTST with tickers: {label}")
        res = evaluate_combination(target, feature_list, series_map, freq, settings)
        status = "OK" if res.ok else f"FAIL ({res.error})"
        log(f"    -> MAPE: {res.mape:.4f}% | runtime {res.runtime_sec:.1f}s | {status}")
        results.append(res)

    summary = summarise_results(results)
    if not results:
        raise RuntimeError("No combinations evaluated.")
    best_result = min(results, key=_score_result)

    return FeatureSelectionResult(
        target=target,
        frequency=freq,
        correlations=correlations,
        combinations=results,
        summary=summary,
        best_result=best_result,
    )
