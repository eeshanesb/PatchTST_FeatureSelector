from __future__ import annotations

import pathlib
import sys


def _bootstrap_local_package() -> pathlib.Path:
    """Ensure the PatchTST feature selector package is importable."""
    base_dir = pathlib.Path(__file__).resolve().parents[1]
    src_dir = base_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return base_dir


REPO_ROOT = _bootstrap_local_package()

from patchtst_featureselect import FeatureSelectorSettings, run_feature_selection  # noqa: E402


def main() -> None:
    data_dir = REPO_ROOT / "HFdata"
    settings = FeatureSelectorSettings(
        data_dir=str(data_dir),
        target="ERX",
        top_n=5,
        combo_max_size=None,
        horizon=7,
        train_last_n=90,
    )

    result = run_feature_selection(settings)
    print(result.summary.head())


if __name__ == "__main__":
    main()
