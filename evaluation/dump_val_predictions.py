#!/usr/bin/env python3

import argparse
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner
from mmaction.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    register_all_modules(init_default_scope=True)

    config_path = Path(
        args.config
    ).resolve()

    checkpoint_path = Path(
        args.checkpoint
    ).resolve()

    output_path = Path(
        args.output
    ).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            config_path
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = Config.fromfile(
        str(config_path)
    )

    # -----------------------------------------
    # Validation-only inference.
    #
    # IMPORTANT:
    # Use validation dataloader as test loop
    # input. Do not touch the original config.
    # -----------------------------------------

    cfg.test_dataloader = (
        cfg.val_dataloader
    )

    base_evaluator = cfg.val_evaluator

    dump_evaluator = dict(
        type="DumpResults",
        out_file_path=str(
            output_path
        ),
    )

    if isinstance(
        base_evaluator,
        (list, tuple),
    ):
        cfg.test_evaluator = (
            list(base_evaluator)
            + [dump_evaluator]
        )
    else:
        cfg.test_evaluator = [
            base_evaluator,
            dump_evaluator,
        ]

    cfg.load_from = str(
        checkpoint_path
    )

    cfg.resume = False

    cfg.work_dir = str(
        output_path.parent
        / (
            "_runner_"
            + output_path.stem
        )
    )

    print("=" * 88)
    print(
        "MMAction2 Validation-only inference"
    )
    print("=" * 88)
    print(
        "config     :",
        config_path,
    )
    print(
        "checkpoint :",
        checkpoint_path,
    )
    print(
        "dump       :",
        output_path,
    )
    print("=" * 88)

    runner = Runner.from_cfg(cfg)

    metrics = runner.test()

    if not output_path.is_file():
        raise RuntimeError(
            "Prediction dump was not created."
        )

    print()
    print("=" * 88)
    print("VALIDATION INFERENCE: PASS")
    print("metrics:", metrics)
    print("dump   :", output_path)
    print("=" * 88)


if __name__ == "__main__":
    main()
