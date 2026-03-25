from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("Python 3.11+ is required to read TOML configs.") from exc

from src.training.engine import run_experiment
from src.training.experiment_config import config_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more training experiments from a TOML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "experiments.toml",
        help="Path to TOML experiments config file.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of run_name values to run.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failed experiment.",
    )
    return parser.parse_args()


def _expand_repeats(experiment: dict) -> list[dict]:
    repeats = int(experiment.get("repeats", 1))
    if repeats < 1:
        raise ValueError(f"Invalid repeats={repeats} for run '{experiment.get('run_name')}'.")

    base_name = experiment.get("run_name")
    expanded: list[dict] = []
    for idx in range(1, repeats + 1):
        item = dict(experiment)
        item.pop("repeats", None)
        if repeats > 1:
            item["run_name"] = f"{base_name}_r{idx:02d}"
            if item.get("seed") is not None:
                item["seed"] = int(item["seed"]) + (idx - 1)
        expanded.append(item)
    return expanded


def load_experiments(config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    base_dir = config_path.parent.resolve()
    defaults = raw.get("defaults", {})
    experiments = raw.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments found. Add one or more [[experiments]] entries.")

    expanded_items: list[dict] = []
    for item in experiments:
        expanded_items.extend(_expand_repeats(item))

    return [config_from_dict(item, base_dir=base_dir, defaults=defaults) for item in expanded_items]


def main() -> None:
    args = parse_args()
    configs = load_experiments(args.config.resolve())

    if args.only:
        selected = set(args.only)
        configs = [cfg for cfg in configs if cfg.run_name in selected]
        missing = selected - {cfg.run_name for cfg in configs}
        if missing:
            raise ValueError(f"Unknown run_name values in --only: {sorted(missing)}")

    print(f"Loaded {len(configs)} experiment(s) from {args.config}.", flush=True)
    results = []
    failed = []

    for idx, config in enumerate(configs, start=1):
        print(f"\n[{idx}/{len(configs)}] Starting '{config.run_name}'", flush=True)
        try:
            result = run_experiment(config)
            results.append(result)
        except Exception as exc:  # pragma: no cover - depends on runtime/data
            failed.append({"run_name": config.run_name, "error": str(exc)})
            print(f"Experiment '{config.run_name}' failed: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            if args.stop_on_error:
                break

    print("\nBatch run summary:", flush=True)
    print(f"Successful: {len(results)}", flush=True)
    print(f"Failed: {len(failed)}", flush=True)
    for item in failed:
        print(f"- {item['run_name']}: {item['error']}", flush=True)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
