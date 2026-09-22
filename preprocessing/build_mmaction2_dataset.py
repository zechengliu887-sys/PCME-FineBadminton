from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from public_pose_dataset import LABEL_TO_INDEX, PublicPoseDataset


EXPECTED_TOTAL = 19866

EXPECTED_SPLITS = {
    "train": 14279,
    "val": 2894,
    "test": 2693,
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(
            f"Empty manifest: {path}"
        )

    required = {
        "sample_id",
        "stroke_class",
        "match_id",
        "split",
        "target_side",
        "source_start_frame",
        "source_end_frame",
        "retained_source_length",
        "model_target_frames",
        "temporal_policy",
    }

    missing = required - set(rows[0])

    if missing:
        raise RuntimeError(
            "Manifest missing columns: "
            + ", ".join(sorted(missing))
        )

    return rows


def crop_npz(
    source: Path,
    destination: Path,
    start_frame: int,
    end_frame: int,
    expected_length: int,
) -> None:

    with np.load(
        source,
        allow_pickle=False,
    ) as z:

        required = {
            "frame_indices",
            "keypoints_image_norm",
            "keypoint_scores",
            "valid_mask",
            "target_player_index",
        }

        missing = required - set(z.files)

        if missing:
            raise RuntimeError(
                f"{source.name} missing NPZ keys: "
                f"{sorted(missing)}"
            )

        frame_indices = np.asarray(
            z["frame_indices"]
        )

        selected = np.flatnonzero(
            (frame_indices >= start_frame)
            & (frame_indices <= end_frame)
        )

        if len(selected) != expected_length:
            raise RuntimeError(
                f"{source.name}: expected "
                f"{expected_length} retained frames, "
                f"found {len(selected)}"
            )

        payload = {
            "frame_indices":
                np.asarray(
                    z["frame_indices"]
                )[selected],

            "keypoints_image_norm":
                np.asarray(
                    z["keypoints_image_norm"]
                )[:, selected, ...],

            "keypoint_scores":
                np.asarray(
                    z["keypoint_scores"]
                )[:, selected, ...],

            "valid_mask":
                np.asarray(
                    z["valid_mask"]
                )[:, selected, ...],

            "target_player_index":
                np.asarray(
                    z["target_player_index"]
                ),
        }

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        destination,
        **payload,
    )


def build_compat_manifest(
    public_rows: list[dict[str, str]],
    pose_root: Path,
    work_dir: Path,
) -> tuple[Path, int]:

    trim_dir = work_dir / "trimmed_npz"
    compat_path = work_dir / "compat_manifest.csv"

    compat_rows = []
    cropped_count = 0

    for index, row in enumerate(public_rows):

        sample_id = row["sample_id"]

        source = (
            pose_root
            / f"{sample_id}.npz"
        )

        if not source.is_file():
            raise RuntimeError(
                f"Missing pose NPZ: {source}"
            )

        start_frame = int(
            row["source_start_frame"]
        )

        end_frame = int(
            row["source_end_frame"]
        )

        expected_length = int(
            row["retained_source_length"]
        )

        with np.load(
            source,
            allow_pickle=False,
        ) as z:

            frame_indices = np.asarray(
                z["frame_indices"]
            )

            selected = np.flatnonzero(
                (frame_indices >= start_frame)
                & (frame_indices <= end_frame)
            )

            if len(selected) != expected_length:
                raise RuntimeError(
                    f"{sample_id}: retained length "
                    f"{len(selected)} != "
                    f"{expected_length}"
                )

            use_original = (
                len(selected)
                == len(frame_indices)
                and np.array_equal(
                    selected,
                    np.arange(
                        len(frame_indices)
                    ),
                )
            )

        if use_original:
            final_path = source
        else:
            final_path = (
                trim_dir
                / f"{sample_id}.npz"
            )

            crop_npz(
                source=source,
                destination=final_path,
                start_frame=start_frame,
                end_frame=end_frame,
                expected_length=expected_length,
            )

            cropped_count += 1

        compat_rows.append(
            {
                "clip_id":
                    sample_id,

                "label_clean":
                    row["stroke_class"],

                "split_match_v1":
                    row["split"],

                "clean_action":
                    "public_manifest",

                "effective_frame_count":
                    expected_length,

                "final_npz_path":
                    str(final_path.resolve()),

                "length_tier":
                    "main_ge16",
            }
        )

        if (
            (index + 1) % 2000 == 0
            or index + 1 == len(public_rows)
        ):
            print(
                "prepared",
                index + 1,
                "/",
                len(public_rows),
            )

    fields = list(
        compat_rows[0].keys()
    )

    with compat_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            compat_rows
        )

    return compat_path, cropped_count


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--pose-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
    )

    args = parser.parse_args()

    manifest = (
        args.manifest
        .expanduser()
        .resolve()
    )

    pose_root = (
        args.pose_root
        .expanduser()
        .resolve()
    )

    output = (
        args.output
        .expanduser()
        .resolve()
    )

    rows = read_manifest(
        manifest
    )

    print("manifest rows =", len(rows))

    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} rows, "
            f"got {len(rows)}"
        )

    split_counts = Counter(
        row["split"]
        for row in rows
    )

    print(
        "split counts =",
        dict(split_counts),
    )

    if dict(split_counts) != EXPECTED_SPLITS:
        raise RuntimeError(
            "Unexpected split counts: "
            f"{dict(split_counts)}"
        )

    if any(
        int(row["model_target_frames"])
        != 48
        for row in rows
    ):
        raise RuntimeError(
            "All samples must use "
            "model_target_frames=48"
        )

    with tempfile.TemporaryDirectory(
        prefix="pcme_public_build_"
    ) as tmp:

        work_dir = Path(tmp)

        compat_manifest, cropped_count = (
            build_compat_manifest(
                public_rows=rows,
                pose_root=pose_root,
                work_dir=work_dir,
            )
        )

        print(
            "cropped samples =",
            cropped_count,
        )

        if args.verify_only:
            print(
                "PUBLIC DATA CONTRACT "
                "VERIFICATION = PASS"
            )
            return 0

        dataset = PublicPoseDataset(
            manifest_path=compat_manifest,
            target_frames=48,
            minimum_source_frames=16,
        )

        annotations = []

        split = {
            "train": [],
            "val": [],
            "test": [],
        }

        for index in range(len(dataset)):

            sample = dataset[index]
            row = rows[index]

            pose = (
                sample["pose_target"]
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            keypoint = np.transpose(
                pose[0:2],
                (3, 1, 2, 0),
            ).astype(
                np.float32,
                copy=False,
            )

            keypoint_score = np.transpose(
                pose[2],
                (2, 0, 1),
            ).astype(
                np.float32,
                copy=False,
            )

            if keypoint.shape != (
                1,
                48,
                17,
                2,
            ):
                raise RuntimeError(
                    f"{row['sample_id']}: "
                    f"bad keypoint shape "
                    f"{keypoint.shape}"
                )

            if keypoint_score.shape != (
                1,
                48,
                17,
            ):
                raise RuntimeError(
                    f"{row['sample_id']}: "
                    f"bad score shape "
                    f"{keypoint_score.shape}"
                )

            label_name = row[
                "stroke_class"
            ]

            split_name = row[
                "split"
            ]

            sample_id = row[
                "sample_id"
            ]

            annotation = {
                "frame_dir":
                    sample_id,

                "label":
                    LABEL_TO_INDEX[
                        label_name
                    ],

                "img_shape":
                    (1, 1),

                "original_shape":
                    (1, 1),

                "total_frames":
                    48,

                "keypoint":
                    keypoint,

                "keypoint_score":
                    keypoint_score,

                "clip_id":
                    sample_id,

                "label_name":
                    label_name,

                "split_match_v1":
                    split_name,

                "source_effective_frame_count":
                    int(
                        row[
                            "retained_source_length"
                        ]
                    ),
            }

            annotations.append(
                annotation
            )

            split[
                split_name
            ].append(
                sample_id
            )

            if (
                (index + 1) % 2000 == 0
                or index + 1 == len(dataset)
            ):
                print(
                    "built",
                    index + 1,
                    "/",
                    len(dataset),
                )

        package = {
            "split": split,
            "annotations": annotations,
        }

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open(
            "wb"
        ) as f:
            pickle.dump(
                package,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        report = {
            "manifest":
                str(manifest),

            "pose_root":
                str(pose_root),

            "output":
                str(output),

            "total":
                len(annotations),

            "split_counts":
                dict(split_counts),

            "cropped_samples":
                cropped_count,

            "target_frames":
                48,
        }

        report_path = (
            args.report
            if args.report is not None
            else output.with_suffix(
                ".summary.json"
            )
        )

        report_path = (
            report_path
            .expanduser()
            .resolve()
        )

        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print()
    print("OUTPUT =", output)
    print("TOTAL =", len(annotations))
    print(
        "CROPPED =",
        cropped_count,
    )
    print(
        "PUBLIC MMACTION2 BUILD = PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
