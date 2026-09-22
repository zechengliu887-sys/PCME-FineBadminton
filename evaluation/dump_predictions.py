#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner
from mmaction.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Dump MMAction2 predictions for the frozen "
            "validation or test split."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
    )
    parser.add_argument(
        "--expected-n",
        type=int,
        default=None,
    )
    return parser.parse_args()


def get_head_type(cfg: Config) -> str:
    model = cfg.get("model", {})
    if not hasattr(model, "get"):
        return ""

    cls_head = model.get("cls_head", {})
    if not hasattr(cls_head, "get"):
        return ""

    return str(cls_head.get("type", ""))


def main():
    args = parse_args()

    register_all_modules(init_default_scope=True)

    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = Config.fromfile(str(config_path))

    head_type = get_head_type(cfg)
    pair_only = (
        "PairOnly" in head_type
        or "paironly" in config_path.name.lower()
        or "pair_only" in config_path.name.lower()
    )

    if args.split == "test" and pair_only:
        raise RuntimeError(
            "Pair-only is a validation-only control in the frozen "
            "experimental protocol. Test inference is intentionally "
            "disabled for Pair-only configs."
        )

    if args.split == "val":
        if "val_dataloader" not in cfg:
            raise KeyError(
                "The config does not define val_dataloader."
            )

        cfg.test_dataloader = cfg.val_dataloader
    else:
        if "test_dataloader" not in cfg:
            raise KeyError(
                "The config does not define test_dataloader."
            )
    cfg.test_evaluator = [
        dict(
            type="DumpResults",
            out_file_path=str(output_path),
        )
    ]

    cfg.load_from = str(checkpoint_path)
    cfg.resume = False

    cfg.work_dir = str(
        output_path.parent
        / f"_runner_{args.split}_{output_path.stem}"
    )

    print("=" * 88)
    print("MMAction2 frozen-split inference")
    print("=" * 88)
    print("split      :", args.split)
    print("config     :", config_path)
    print("checkpoint :", checkpoint_path)
    print("head       :", head_type or "<unknown>")
    print("dump       :", output_path)
    print("=" * 88)

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    if not output_path.is_file():
        raise RuntimeError(
            "Prediction dump was not created."
        )

    if args.expected_n is not None:
        with output_path.open("rb") as f:
            items = pickle.load(f)

        if len(items) != args.expected_n:
            raise RuntimeError(
                f"Expected N={args.expected_n}, "
                f"got {len(items)}."
            )

    print()
    print("=" * 88)
    print(
        f"{args.split.upper()} INFERENCE: PASS"
    )
    print("metrics:", metrics)
    print("dump   :", output_path)

    if args.expected_n is not None:
        print("N      :", args.expected_n)

    print("=" * 88)


if __name__ == "__main__":
    main()
