from __future__ import annotations

from pathlib import Path
import numpy as np


TARGET_FRAMES = 48

KNOWN_TIME_AXES = {
    "frame_indices": 0,
    "keypoints_px": 1,
    "keypoints": 1,
    "keypoints_norm": 1,
    "keypoints_image_norm": 1,
    "keypoint_scores": 1,
    "bboxes": 1,
    "bbox_scores": 1,
    "valid_mask": 1,
    "player_valid_mask": 1,
    "court_xy": 1,
    "foot_points": 1,
}


def _linear_resample(
    array: np.ndarray,
    axis: int,
    target_frames: int,
) -> np.ndarray:
    source_frames = array.shape[axis]

    if source_frames == target_frames:
        return np.array(
            array,
            copy=True,
        )

    source_positions = np.linspace(
        0.0,
        1.0,
        source_frames,
        dtype=np.float64,
    )

    target_positions = np.linspace(
        0.0,
        1.0,
        target_frames,
        dtype=np.float64,
    )

    moved = np.moveaxis(
        array,
        axis,
        0,
    )

    flat = moved.reshape(
        source_frames,
        -1,
    )

    output = np.empty(
        (
            target_frames,
            flat.shape[1],
        ),
        dtype=np.float32,
    )

    for column in range(
        flat.shape[1]
    ):
        output[
            :,
            column,
        ] = np.interp(
            target_positions,
            source_positions,
            flat[
                :,
                column,
            ].astype(
                np.float64
            ),
        )

    output = output.reshape(
        (
            target_frames,
            *moved.shape[1:],
        )
    )

    return np.moveaxis(
        output,
        0,
        axis,
    )


def _nearest_resample(
    array: np.ndarray,
    axis: int,
    target_frames: int,
) -> np.ndarray:
    source_frames = array.shape[axis]

    indices = np.rint(
        np.linspace(
            0,
            source_frames - 1,
            target_frames,
        )
    ).astype(
        np.int64
    )

    return np.take(
        array,
        indices,
        axis=axis,
    )


def resample_temporal_array(
    array: np.ndarray,
    axis: int,
    target_frames: int = TARGET_FRAMES,
) -> np.ndarray:
    if array.shape[axis] <= 0:
        raise ValueError(
            "时间轴长度必须大于0"
        )

    if (
        np.issubdtype(
            array.dtype,
            np.floating,
        )
    ):
        return _linear_resample(
            array,
            axis,
            target_frames,
        )

    return _nearest_resample(
        array,
        axis,
        target_frames,
    )


def load_npz_resampled_to_48(
    npz_path: str | Path,
    minimum_source_frames: int = 16,
    target_frames: int = TARGET_FRAMES,
) -> dict[str, np.ndarray]:
    path = Path(
        npz_path
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        arrays = {
            key: np.asarray(
                data[
                    key
                ]
            )
            for key in data.files
        }

    if "frame_indices" not in arrays:
        raise KeyError(
            f"{path}缺少frame_indices"
        )

    source_frames = len(
        arrays[
            "frame_indices"
        ]
    )

    if (
        source_frames
        < minimum_source_frames
    ):
        raise ValueError(
            f"{path}仅{source_frames}帧，"
            f"低于主训练阈值{minimum_source_frames}"
        )

    output = {}

    for key, array in arrays.items():
        if key in KNOWN_TIME_AXES:
            axis = KNOWN_TIME_AXES[
                key
            ]

            if (
                axis >= array.ndim
                or array.shape[
                    axis
                ]
                != source_frames
            ):
                raise ValueError(
                    f"{key}时间轴不匹配："
                    f"shape={array.shape}, "
                    f"axis={axis}, "
                    f"T={source_frames}"
                )

            output[
                key
            ] = resample_temporal_array(
                array,
                axis,
                target_frames,
            )

            continue

        matching_axes = [
            axis
            for axis, size in enumerate(
                array.shape
            )
            if size
            == source_frames
        ]

        if len(
            matching_axes
        ) == 0:
            output[
                key
            ] = np.array(
                array,
                copy=True,
            )

        elif len(
            matching_axes
        ) == 1:
            output[
                key
            ] = resample_temporal_array(
                array,
                matching_axes[
                    0
                ],
                target_frames,
            )

        else:
            raise ValueError(
                f"未知数组{key}有多个可能时间轴："
                f"shape={array.shape}, "
                f"axes={matching_axes}"
            )

    output[
        "source_effective_frame_count"
    ] = np.asarray(
        source_frames,
        dtype=np.int64,
    )

    output[
        "model_target_frame_count"
    ] = np.asarray(
        target_frames,
        dtype=np.int64,
    )

    return output
