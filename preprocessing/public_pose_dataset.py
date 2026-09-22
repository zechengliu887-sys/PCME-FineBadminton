from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import Dataset

from pose_temporal_resample48 import resample_temporal_array


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

LABEL_TO_INDEX = {
    label: index
    for index, label in enumerate(LABELS)
}


def _text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _fill_coordinate_series(
    coordinates: np.ndarray,
    observed_joint_mask: np.ndarray,
) -> np.ndarray:
    """Fill missing joint coordinates along time for resampling.

    Missing joints keep score=0. Coordinates are filled only to prevent
    NaN propagation during temporal interpolation:
    - >=2 valid observations: linear interpolation with nearest endpoints;
    - 1 valid observation: repeat that position;
    - 0 valid observations: use zero.
    """

    output = np.asarray(
        coordinates,
        dtype=np.float32,
    ).copy()

    people, frames, joints, dimensions = (
        output.shape
    )

    timeline = np.arange(
        frames,
        dtype=np.float64,
    )

    for person in range(people):
        for joint in range(joints):
            valid_times = np.flatnonzero(
                observed_joint_mask[
                    person,
                    :,
                    joint,
                ]
            )

            for dimension in range(
                dimensions
            ):
                series = output[
                    person,
                    :,
                    joint,
                    dimension,
                ]

                finite_valid_times = valid_times[
                    np.isfinite(
                        series[
                            valid_times
                        ]
                    )
                ]

                if (
                    len(
                        finite_valid_times
                    )
                    >= 2
                ):
                    output[
                        person,
                        :,
                        joint,
                        dimension,
                    ] = np.interp(
                        timeline,
                        finite_valid_times,
                        series[
                            finite_valid_times
                        ].astype(
                            np.float64
                        ),
                    ).astype(
                        np.float32
                    )

                elif (
                    len(
                        finite_valid_times
                    )
                    == 1
                ):
                    output[
                        person,
                        :,
                        joint,
                        dimension,
                    ] = series[
                        finite_valid_times[
                            0
                        ]
                    ]

                else:
                    output[
                        person,
                        :,
                        joint,
                        dimension,
                    ] = 0.0

    return output


class PublicPoseDataset(Dataset):
    """Read Clean v3 NPZ samples and create ST-GCN-style tensors.

    Returned tensors:
      pose_dual:   [3, 48, 17, 2]  -> x, y, keypoint score for both players
      pose_target: [3, 48, 17, 1]  -> target hitter only
      valid_mask:  [2, 48]

    Missing/non-finite source joints are handled explicitly:
      1. count and report them;
      2. fill coordinates only for safe temporal interpolation;
      3. keep their keypoint score at zero;
      4. zero invalid player frames after resampling.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        target_frames: int = 48,
        minimum_source_frames: int = 16,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.target_frames = int(target_frames)
        self.minimum_source_frames = int(minimum_source_frames)

        with self.manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            self.rows = list(csv.DictReader(file))

        if not self.rows:
            raise RuntimeError(
                f"清单为空：{self.manifest_path}"
            )

        required_columns = {
            "clip_id",
            "label_clean",
            "split_match_v1",
            "clean_action",
            "effective_frame_count",
            "final_npz_path",
            "length_tier",
        }

        missing = (
            required_columns
            - set(self.rows[0])
        )

        if missing:
            raise RuntimeError(
                "主清单缺少字段："
                f"{sorted(missing)}"
            )

        self.clip_ids = [
            _text(row["clip_id"])
            for row in self.rows
        ]

        self.labels = [
            _text(row["label_clean"])
            for row in self.rows
        ]

        self.splits = [
            _text(row["split_match_v1"])
            for row in self.rows
        ]

        self.actions = [
            _text(row["clean_action"])
            for row in self.rows
        ]

        self.paths = [
            Path(_text(row["final_npz_path"]))
            for row in self.rows
        ]

        self.source_lengths = [
            int(row["effective_frame_count"])
            for row in self.rows
        ]

        for index, label in enumerate(
            self.labels
        ):
            if label not in LABEL_TO_INDEX:
                raise RuntimeError(
                    f"未知标签：{label}，"
                    f"clip={self.clip_ids[index]}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def _load_pose_arrays(
        self,
        index: int,
    ) -> dict[str, np.ndarray | int]:
        clip_id = self.clip_ids[index]
        npz_path = self.paths[index]
        expected_source_length = (
            self.source_lengths[index]
        )

        with np.load(
            npz_path,
            allow_pickle=False,
        ) as data:
            required = {
                "frame_indices",
                "keypoints_image_norm",
                "keypoint_scores",
                "valid_mask",
                "target_player_index",
            }

            missing = (
                required
                - set(data.files)
            )

            if missing:
                raise RuntimeError(
                    f"{clip_id}缺少NPZ字段："
                    f"{sorted(missing)}"
                )

            frame_indices = np.asarray(
                data["frame_indices"]
            )

            coordinates = np.asarray(
                data[
                    "keypoints_image_norm"
                ],
                dtype=np.float32,
            )

            scores = np.asarray(
                data["keypoint_scores"],
                dtype=np.float32,
            )

            valid_mask = np.asarray(
                data["valid_mask"],
                dtype=np.uint8,
            )

            target_player_index = int(
                np.asarray(
                    data[
                        "target_player_index"
                    ]
                ).reshape(-1)[0]
            )

        source_length = len(
            frame_indices
        )

        if (
            source_length
            != expected_source_length
        ):
            raise RuntimeError(
                f"{clip_id}清单长度"
                f"{expected_source_length}，"
                f"NPZ长度{source_length}"
            )

        if (
            source_length
            < self.minimum_source_frames
        ):
            raise RuntimeError(
                f"{clip_id}仅{source_length}帧，"
                f"低于主训练阈值"
                f"{self.minimum_source_frames}"
            )

        expected_coordinate_shape = (
            2,
            source_length,
            17,
            2,
        )

        expected_score_shape = (
            2,
            source_length,
            17,
        )

        expected_valid_shape = (
            2,
            source_length,
        )

        if (
            coordinates.shape
            != expected_coordinate_shape
        ):
            raise RuntimeError(
                f"{clip_id}源坐标shape错误："
                f"{coordinates.shape}"
            )

        if scores.shape != expected_score_shape:
            raise RuntimeError(
                f"{clip_id}源score shape错误："
                f"{scores.shape}"
            )

        if (
            valid_mask.shape
            != expected_valid_shape
        ):
            raise RuntimeError(
                f"{clip_id}源valid_mask shape错误："
                f"{valid_mask.shape}"
            )

        if target_player_index not in {
            0,
            1,
        }:
            raise RuntimeError(
                f"{clip_id}目标球员索引错误："
                f"{target_player_index}"
            )

        if not np.isin(
            valid_mask,
            [0, 1],
        ).all():
            raise RuntimeError(
                f"{clip_id}valid_mask含非0/1值"
            )

        coordinate_finite = np.isfinite(
            coordinates
        )

        score_finite = np.isfinite(
            scores
        )

        joint_coordinate_finite = (
            coordinate_finite.all(
                axis=-1
            )
        )

        player_frame_valid = (
            valid_mask.astype(bool)
        )

        observed_joint_mask = (
            player_frame_valid[
                ...,
                None,
            ]
            & joint_coordinate_finite
            & score_finite
        )

        source_nonfinite_coordinate_value_count = int(
            (~coordinate_finite).sum()
        )

        source_nonfinite_score_value_count = int(
            (~score_finite).sum()
        )

        source_nonfinite_joint_count = int(
            (
                ~(
                    joint_coordinate_finite
                    & score_finite
                )
            ).sum()
        )

        source_invalid_player_frame_count = int(
            (~player_frame_valid).sum()
        )

        coordinates_filled = (
            _fill_coordinate_series(
                coordinates=coordinates,
                observed_joint_mask=(
                    observed_joint_mask
                ),
            )
        )

        scores_clean = np.where(
            score_finite,
            scores,
            0.0,
        ).astype(
            np.float32,
            copy=False,
        )

        scores_clean[
            ~observed_joint_mask
        ] = 0.0

        coordinates_48 = (
            resample_temporal_array(
                coordinates_filled,
                axis=1,
                target_frames=(
                    self.target_frames
                ),
            )
            .astype(
                np.float32,
                copy=False,
            )
        )

        scores_48 = (
            resample_temporal_array(
                scores_clean,
                axis=1,
                target_frames=(
                    self.target_frames
                ),
            )
            .astype(
                np.float32,
                copy=False,
            )
        )

        valid_mask_48 = (
            resample_temporal_array(
                valid_mask,
                axis=1,
                target_frames=(
                    self.target_frames
                ),
            )
            .astype(
                np.uint8,
                copy=False,
            )
        )

        invalid_player_frames_48 = (
            valid_mask_48 == 0
        )

        coordinates_48[
            np.broadcast_to(
                invalid_player_frames_48[
                    ...,
                    None,
                    None,
                ],
                coordinates_48.shape,
            )
        ] = 0.0

        scores_48[
            np.broadcast_to(
                invalid_player_frames_48[
                    ...,
                    None,
                ],
                scores_48.shape,
            )
        ] = 0.0

        if not np.isfinite(
            coordinates_48
        ).all():
            raise RuntimeError(
                f"{clip_id}清洗后坐标仍含NaN或Inf"
            )

        if not np.isfinite(
            scores_48
        ).all():
            raise RuntimeError(
                f"{clip_id}清洗后score仍含NaN或Inf"
            )

        return {
            "coordinates":
                coordinates_48,
            "scores":
                scores_48,
            "valid_mask":
                valid_mask_48,
            "target_player_index":
                target_player_index,
            "source_length":
                source_length,
            "source_nonfinite_coordinate_value_count":
                source_nonfinite_coordinate_value_count,
            "source_nonfinite_score_value_count":
                source_nonfinite_score_value_count,
            "source_nonfinite_joint_count":
                source_nonfinite_joint_count,
            "source_invalid_player_frame_count":
                source_invalid_player_frame_count,
        }

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        arrays = self._load_pose_arrays(
            index
        )

        coordinates = arrays[
            "coordinates"
        ]

        scores = arrays["scores"]

        valid_mask = arrays[
            "valid_mask"
        ]

        target_player_index = int(
            arrays[
                "target_player_index"
            ]
        )

        pose = np.concatenate(
            [
                coordinates,
                scores[
                    ...,
                    None,
                ],
            ],
            axis=-1,
        )

        # [M,T,V,C] -> [C,T,V,M]
        pose_dual = np.transpose(
            pose,
            (3, 1, 2, 0),
        ).astype(
            np.float32,
            copy=False,
        )

        pose_target = (
            pose_dual[
                ...,
                target_player_index:
                target_player_index + 1,
            ]
        )

        return {
            "pose_dual":
                torch.from_numpy(
                    np.ascontiguousarray(
                        pose_dual
                    )
                ),
            "pose_target":
                torch.from_numpy(
                    np.ascontiguousarray(
                        pose_target
                    )
                ),
            "valid_mask":
                torch.from_numpy(
                    np.ascontiguousarray(
                        valid_mask
                    )
                ),
            "label":
                torch.tensor(
                    LABEL_TO_INDEX[
                        self.labels[index]
                    ],
                    dtype=torch.long,
                ),
            "target_player_index":
                torch.tensor(
                    target_player_index,
                    dtype=torch.long,
                ),
            "source_length":
                torch.tensor(
                    int(
                        arrays[
                            "source_length"
                        ]
                    ),
                    dtype=torch.long,
                ),
            "source_nonfinite_coordinate_value_count":
                torch.tensor(
                    int(
                        arrays[
                            "source_nonfinite_coordinate_value_count"
                        ]
                    ),
                    dtype=torch.long,
                ),
            "source_nonfinite_score_value_count":
                torch.tensor(
                    int(
                        arrays[
                            "source_nonfinite_score_value_count"
                        ]
                    ),
                    dtype=torch.long,
                ),
            "source_nonfinite_joint_count":
                torch.tensor(
                    int(
                        arrays[
                            "source_nonfinite_joint_count"
                        ]
                    ),
                    dtype=torch.long,
                ),
            "source_invalid_player_frame_count":
                torch.tensor(
                    int(
                        arrays[
                            "source_invalid_player_frame_count"
                        ]
                    ),
                    dtype=torch.long,
                ),
            "sample_index":
                torch.tensor(
                    index,
                    dtype=torch.long,
                ),
        }
