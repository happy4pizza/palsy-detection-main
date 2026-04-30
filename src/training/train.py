from __future__ import annotations

import argparse
from pathlib import Path

from src.training.config import DEFAULT_CONFIG_PATH, load_config
from src.training.engine import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model from a TOML experiment config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to a TOML config file. Default: {DEFAULT_CONFIG_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_training(config, config_path=args.config)


if __name__ == "__main__":
    main()
