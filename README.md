# PatchTST Feature Selection

Utilities for screening correlated tickers and exhaustively testing feature combinations with [PatchTST](https://arxiv.org/abs/2211.13299) models. The package wraps a workflow that:

- Scans a directory of CSV price data and identifies the tickers most correlated with a target symbol.
- Trains a PatchTST model for every subset of the top correlated tickers (including the baseline target-only model).
- Reports validation MAPE scores and exports a summary for further analysis.

The code is packaged so it can be imported and re-used programmatically, or invoked from the command line.

## Installation

```bash
pip install .
```

The package depends on heavy numerical libraries (`torch`, `neuralforecast`, `numpy`, `pandas`, `scipy`, `pyyaml`). Ensure a matching Python wheel is available for your platform; using a fresh virtual environment is recommended.

After installation you can call the CLI via `patchtst-featureselect` or import the Python API:

```python
from patchtst_featureselect import FeatureSelectorSettings, run_feature_selection
```

## Usage

```bash
patchtst-featureselect --config config.yaml --target ERX --data-dir HFdata
```

The CLI reads defaults from a shared YAML configuration (see `config.yaml` for an example) and supports overriding parameters such as `--top-n`, `--combo-max-size`, `--input-size`, and more. Use `--help` to list all options.

Set `--silent` to suppress console output while still running the exhaustive search. The `--export-summary` option writes the evaluation table to CSV.

## Programmatic API

```python
from patchtst_featureselect import FeatureSelectorSettings, run_feature_selection

settings = FeatureSelectorSettings(
    data_dir="HFdata",
    target="ERX",
    top_n=5,
    min_overlap=60,
    horizon=7,
)

result = run_feature_selection(settings)
print(result.summary.head())
```

`FeatureSelectionResult.summary` contains the sorted evaluation DataFrame, while `FeatureSelectionResult.best_result` exposes the best combination with detailed metrics.

## Configuration Expectations

Ticker CSV files must contain a recognizable datetime column (e.g. `date`, `ds`, `timestamp`) and a price column (e.g. `Close`, `Adj Close`, `price`). The loader automatically infers the sampling frequency if one is not provided.

The YAML configuration mirrors the original PatchTST tooling and provides sensible defaults for the CLI. Update paths and hyperparameters there before running the selector.

## Development

- Format: the codebase follows standard `black`/`isort` compatible formatting.
- Packaging: update `pyproject.toml` with your preferred metadata and bump `__version__` in `patchtst_featureselect/__init__.py` when releasing.
- Tests: consider adding integration tests once you have deterministic training stubs or fixtures.

Contributions and suggestions are welcome—fork the repository and open a PR on GitHub.
