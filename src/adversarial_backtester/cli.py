"""CLI adapter for a reproducible demo or a headered price CSV."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .engine import BacktestConfig, run_backtest
from .synthetic import synthetic_prices


def load_prices(path: Path, column: str = "close") -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"CSV is missing price column {column!r}")
        try:
            return np.array([float(row[column]) for row in reader], dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"CSV column {column!r} must contain numeric prices on every row"
            ) from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, help="ordered, equally spaced prices; defaults to synthetic"
    )
    parser.add_argument("--price-column", default="close")
    parser.add_argument("--output", type=Path, help="write a complete JSON research artifact")
    parser.add_argument("--n-prices", type=int, default=600, help="synthetic dataset length")
    defaults = BacktestConfig()
    for name in (
        "momentum_window",
        "volatility_window",
        "n_splits",
        "embargo",
        "min_train_size",
        "periods_per_year",
        "n_permutations",
        "block_size",
        "n_trials",
        "seed",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name)
        )
    for name in ("threshold", "cost_bps", "alpha"):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name)
        )
    args = parser.parse_args(argv)
    try:
        config = BacktestConfig(
            **{
                name: getattr(args, name)
                for name in BacktestConfig.__dataclass_fields__
                if name != "liquidate"
            }
        )
        prices = (
            load_prices(args.csv, args.price_column)
            if args.csv
            else synthetic_prices(args.n_prices, args.seed)
        )
        result = run_backtest(prices, config)
        artifact = result.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {key: value for key, value in artifact.items() if key != "observations"},
            indent=2,
            allow_nan=False,
        )
    )
    return 0
