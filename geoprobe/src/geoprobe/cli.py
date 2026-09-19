from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .config import load_config


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _common(parser: argparse.ArgumentParser, *, dataset: bool = False, proposer: bool = False, model: bool = False, device: bool = False) -> None:
    parser.add_argument("--config", type=_path, required=True)
    if dataset:
        parser.add_argument("--dataset-root", type=_path)
    if proposer:
        parser.add_argument("--wedetect-model", type=_path)
    if model:
        parser.add_argument("--geoclip-weights", type=_path)
        parser.add_argument("--gallery", type=_path)
    if device:
        parser.add_argument("--device")
    parser.add_argument("--output-dir", type=_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geoprobe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache-proposals", help="build the immutable WeDetect proposal cache")
    _common(cache, dataset=True, proposer=True)

    search = subparsers.add_parser("search", help="search interventions on the discovery split")
    _common(search, dataset=True, model=True, device=True)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a frozen search selection")
    _common(evaluate, dataset=True, model=True, device=True)
    evaluate.add_argument("--selection", type=_path, required=True)
    evaluate.add_argument("--split", choices=("holdout", "all"), required=True)

    export = subparsers.add_parser("export", help="recompute tables from paired rows")
    _common(export)
    return parser


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "dataset_root", None) is not None:
        config["dataset"]["root"] = str(args.dataset_root)
    if getattr(args, "wedetect_model", None) is not None:
        config["proposer"]["model_path"] = str(args.wedetect_model)
    if getattr(args, "geoclip_weights", None) is not None:
        config["geoclip"]["weights_path"] = str(args.geoclip_weights)
    if getattr(args, "gallery", None) is not None:
        config["geoclip"]["gallery_path"] = str(args.gallery)
    if getattr(args, "device", None) is not None:
        config["runtime"]["device"] = args.device
    if getattr(args, "output_dir", None) is not None:
        config["runtime"]["output_dir"] = str(args.output_dir)
    return config


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _apply_overrides(load_config(args.config), args)
    from .pipeline import build_pipeline_block

    block = build_pipeline_block(args.command, config)
    if args.command == "evaluate":
        block.run(selection_path=args.selection, split=args.split)
    else:
        block.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
