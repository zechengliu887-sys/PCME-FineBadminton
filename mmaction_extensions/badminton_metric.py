from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from mmengine.evaluator import BaseMetric

from mmaction.registry import METRICS


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


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(
        numerator,
        dtype=np.float64,
    )

    np.divide(
        numerator,
        denominator,
        out=output,
        where=denominator != 0,
    )

    return output


@METRICS.register_module()
class BadmintonClassificationMetric(BaseMetric):
    """Single-label badminton classification metrics.

    Returns accuracy, top-5 accuracy, macro precision, macro recall,
    and macro F1. Optionally writes per-class metrics, a confusion
    matrix, and predictions to an output directory.
    """

    default_prefix = "badminton"

    def __init__(
        self,
        labels: Sequence[str] = LABELS,
        detailed: bool = False,
        output_dir: Optional[str] = None,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(
            collect_device=collect_device,
            prefix=prefix,
        )

        self.labels = list(labels)
        self.num_classes = len(self.labels)
        self.detailed = detailed
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else None
        )

    def process(
        self,
        data_batch: Sequence[Tuple[Any, Dict]],
        data_samples: Sequence[Dict],
    ) -> None:
        for data_sample in data_samples:
            pred_score = (
                data_sample["pred_score"]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=False,
                )
            )

            gt_label = int(
                data_sample["gt_label"]
                .detach()
                .cpu()
                .item()
            )

            self.results.append(
                {
                    "pred_score":
                        pred_score,
                    "gt_label":
                        gt_label,
                }
            )

    def compute_metrics(
        self,
        results: List[Dict],
    ) -> Dict[str, float]:
        scores = np.stack(
            [
                item["pred_score"]
                for item in results
            ],
            axis=0,
        )

        targets = np.asarray(
            [
                item["gt_label"]
                for item in results
            ],
            dtype=np.int64,
        )

        predictions = scores.argmax(
            axis=1
        ).astype(
            np.int64,
            copy=False,
        )

        if scores.shape[1] != self.num_classes:
            raise RuntimeError(
                "类别数不一致："
                f"scores={scores.shape[1]}, "
                f"labels={self.num_classes}"
            )

        confusion = np.zeros(
            (
                self.num_classes,
                self.num_classes,
            ),
            dtype=np.int64,
        )

        np.add.at(
            confusion,
            (
                targets,
                predictions,
            ),
            1,
        )

        true_positive = np.diag(
            confusion
        ).astype(
            np.float64
        )

        support = confusion.sum(
            axis=1
        ).astype(
            np.float64
        )

        predicted_count = confusion.sum(
            axis=0
        ).astype(
            np.float64
        )

        precision = safe_divide(
            true_positive,
            predicted_count,
        )

        recall = safe_divide(
            true_positive,
            support,
        )

        f1 = safe_divide(
            2.0
            * precision
            * recall,
            precision
            + recall,
        )

        accuracy = float(
            true_positive.sum()
            / len(targets)
        )

        top_k = min(
            5,
            self.num_classes,
        )

        top_indices = np.argpartition(
            scores,
            kth=scores.shape[1] - top_k,
            axis=1,
        )[:, -top_k:]

        top5_accuracy = float(
            np.mean(
                np.any(
                    top_indices
                    == targets[:, None],
                    axis=1,
                )
            )
        )

        metrics: Dict[str, float] = {
            "accuracy":
                accuracy,
            "top5_accuracy":
                top5_accuracy,
            "macro_precision":
                float(
                    precision.mean()
                ),
            "macro_recall":
                float(
                    recall.mean()
                ),
            "macro_f1":
                float(
                    f1.mean()
                ),
        }

        if self.detailed:
            for index, label in enumerate(
                self.labels
            ):
                safe_label = (
                    label
                    .replace("-", "_")
                    .replace(" ", "_")
                )

                metrics[
                    f"f1_{index:02d}_{safe_label}"
                ] = float(
                    f1[index]
                )

        if self.output_dir is not None:
            self._write_outputs(
                scores=scores,
                targets=targets,
                predictions=predictions,
                confusion=confusion,
                precision=precision,
                recall=recall,
                f1=f1,
                metrics=metrics,
            )

        return metrics

    def _write_outputs(
        self,
        scores: np.ndarray,
        targets: np.ndarray,
        predictions: np.ndarray,
        confusion: np.ndarray,
        precision: np.ndarray,
        recall: np.ndarray,
        f1: np.ndarray,
        metrics: Dict[str, float],
    ) -> None:
        output_dir = self.output_dir
        assert output_dir is not None

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary = {
            "version":
                "badminton_classification_metric_v1",
            "sample_count":
                int(
                    len(targets)
                ),
            "labels":
                self.labels,
            "metrics":
                metrics,
        }

        (
            output_dir
            / "summary.json"
        ).write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        with (
            output_dir
            / "per_class_metrics.csv"
        ).open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "class_index",
                    "label",
                    "support",
                    "precision",
                    "recall",
                    "f1",
                ],
            )

            writer.writeheader()

            for index, label in enumerate(
                self.labels
            ):
                writer.writerow(
                    {
                        "class_index":
                            index,
                        "label":
                            label,
                        "support":
                            int(
                                confusion[
                                    index
                                ].sum()
                            ),
                        "precision":
                            float(
                                precision[
                                    index
                                ]
                            ),
                        "recall":
                            float(
                                recall[
                                    index
                                ]
                            ),
                        "f1":
                            float(
                                f1[
                                    index
                                ]
                            ),
                    }
                )

        with (
            output_dir
            / "confusion_matrix.csv"
        ).open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    "true_label",
                    *self.labels,
                ]
            )

            for index, label in enumerate(
                self.labels
            ):
                writer.writerow(
                    [
                        label,
                        *confusion[
                            index
                        ].tolist(),
                    ]
                )

        with (
            output_dir
            / "predictions.csv"
        ).open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "sample_index",
                    "true_index",
                    "true_label",
                    "predicted_index",
                    "predicted_label",
                    "confidence",
                    "correct",
                ],
            )

            writer.writeheader()

            for index, (
                target,
                prediction,
            ) in enumerate(
                zip(
                    targets,
                    predictions,
                )
            ):
                writer.writerow(
                    {
                        "sample_index":
                            index,
                        "true_index":
                            int(target),
                        "true_label":
                            self.labels[
                                int(target)
                            ],
                        "predicted_index":
                            int(
                                prediction
                            ),
                        "predicted_label":
                            self.labels[
                                int(
                                    prediction
                                )
                            ],
                        "confidence":
                            float(
                                scores[
                                    index,
                                    prediction,
                                ]
                            ),
                        "correct":
                            int(
                                target
                                == prediction
                            ),
                    }
                )
