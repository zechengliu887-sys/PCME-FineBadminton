#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


LABELS = [
    "block",
    "clear",
    "cross-court net shot",
    "drive",
    "drop shot",
    "kill",
    "net kill",
    "net lift",
    "net shot",
    "push shot",
    "serve",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fixed 1:1 probability late fusion "
            "for joint and bone streams."
        )
    )

    parser.add_argument(
        "--joint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--bone",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--expected-n",
        type=int,
        default=None,
    )

    return parser.parse_args()


def get_field(item, name):
    try:
        value = item[name]
    except Exception:
        value = getattr(item, name)

    if hasattr(value, "cpu"):
        value = value.cpu().numpy()

    return np.asarray(value)


def load_dump(path: Path):
    with path.open("rb") as f:
        items = pickle.load(f)

    scores = []
    labels = []

    for item in items:
        score = get_field(
            item,
            "pred_score",
        ).reshape(-1)

        gt = get_field(
            item,
            "gt_label",
        ).reshape(-1)

        if score.shape != (11,):
            raise RuntimeError(
                f"{path}: score shape "
                f"{score.shape} != (11,)"
            )

        if gt.size != 1:
            raise RuntimeError(
                f"{path}: invalid gt shape "
                f"{gt.shape}"
            )

        scores.append(score)
        labels.append(int(gt[0]))

    return (
        np.stack(scores),
        np.asarray(labels),
    )


def save_matrix(
    path: Path,
    matrix: np.ndarray,
):
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            ["true/pred"] + LABELS
        )

        for name, row in zip(
            LABELS,
            matrix,
        ):
            writer.writerow(
                [name] + list(row)
            )


def main():
    args = parse_args()

    joint_path = (
        args.joint.expanduser().resolve()
    )

    bone_path = (
        args.bone.expanduser().resolve()
    )

    out = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    joint_score, joint_gt = (
        load_dump(joint_path)
    )

    bone_score, bone_gt = (
        load_dump(bone_path)
    )

    if joint_score.shape != bone_score.shape:
        raise RuntimeError(
            "Joint/bone score shapes differ: "
            f"{joint_score.shape} vs "
            f"{bone_score.shape}"
        )

    if not np.array_equal(
        joint_gt,
        bone_gt,
    ):
        raise RuntimeError(
            "Joint/bone ground-truth "
            "orders differ."
        )

    if args.expected_n is not None:
        if len(joint_gt) != args.expected_n:
            raise RuntimeError(
                f"Expected N={args.expected_n}, "
                f"got {len(joint_gt)}"
            )

    if not np.isfinite(
        joint_score
    ).all():
        raise RuntimeError(
            "Joint scores contain NaN/Inf."
        )

    if not np.isfinite(
        bone_score
    ).all():
        raise RuntimeError(
            "Bone scores contain NaN/Inf."
        )

    # Frozen paper protocol:
    # fixed 1:1 probability late fusion.
    fused = (
        joint_score
        + bone_score
    ) / 2.0

    y = joint_gt
    pred = fused.argmax(axis=1)

    acc = accuracy_score(
        y,
        pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y,
            pred,
            labels=list(range(11)),
            average="macro",
            zero_division=0,
        )
    )

    top5 = np.mean([
        y[i]
        in np.argsort(
            fused[i]
        )[-5:]
        for i in range(len(y))
    ])

    cm = confusion_matrix(
        y,
        pred,
        labels=list(range(11)),
    )

    row_sum = cm.sum(
        axis=1,
        keepdims=True,
    )

    cm_norm = np.divide(
        cm,
        row_sum,
        out=np.zeros_like(
            cm,
            dtype=float,
        ),
        where=row_sum != 0,
    )

    with (
        out / "predictions.pkl"
    ).open("wb") as f:
        pickle.dump(
            {
                "gt": y,
                "prob": fused,
                "pred": pred,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    save_matrix(
        out / "confusion_matrix.csv",
        cm,
    )

    save_matrix(
        out
        / "normalized_confusion_matrix.csv",
        cm_norm,
    )

    metrics = {
        "N": int(len(y)),
        "accuracy": float(acc),
        "top5_accuracy": float(top5),
        "macro_precision": float(
            precision
        ),
        "macro_recall": float(
            recall
        ),
        "macro_f1": float(f1),
        "fusion": "fixed_1_to_1_probability",
    }

    (
        out / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("N =", len(y))
    print(
        "Accuracy =",
        f"{acc * 100:.6f}%"
    )
    print(
        "Macro-F1 =",
        f"{f1 * 100:.6f}%"
    )
    print(
        "Top-5 =",
        f"{top5 * 100:.6f}%"
    )
    print(
        "TWO-STREAM FUSION = PASS"
    )


if __name__ == "__main__":
    main()
