# Dataset and benchmark protocol

This document describes the frozen data contract used by the PCME experiments.

## Source dataset

The benchmark is derived from **Finebadminton-20K**, the public subset of FineBadminton.

Official sources:

- Dataset: https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K
- Project page: https://finebadminton.github.io/FineBadminton/
- Paper: https://arxiv.org/abs/2508.07554

This repository does **not** redistribute the original FineBadminton videos. Users should obtain the source dataset from its official provider and follow the license and terms published there.

## Frozen PCME benchmark

The benchmark contains **19,866** target-player stroke samples from **11** stroke classes.

Split sizes:

- Train: **14,279**
- Validation: **2,894**
- Test: **2,693**

The split is **match-disjoint**: no `match_id` appears in more than one split.

The released benchmark manifest is:

```text
manifests/finebadminton_pcme_match_v1_manifest.csv
```

The manifest contains exactly 19,866 unique `sample_id` values.

## Public manifest schema

Each row contains:

| Column | Meaning |
| --- | --- |
| `sample_id` | Stable sample identifier |
| `stroke_class` | One of the 11 stroke labels |
| `match_id` | Source match identifier |
| `split` | `train`, `val`, or `test` |
| `target_side` | Target player side (`top` or `bottom`) |
| `source_start_frame` | First retained source-frame index, inclusive |
| `source_end_frame` | Last retained source-frame index, inclusive |
| `retained_source_length` | Number of source frames retained before resampling |
| `model_target_frames` | Temporal length expected by the recognition model; fixed to 48 |
| `temporal_policy` | Frozen temporal-resampling policy |

The public manifest contains no machine-specific absolute paths.

## Pose-NPZ input contract

The released preprocessing code starts from one full two-player pose NPZ per sample:

```text
<pose-root>/<sample_id>.npz
```

Required arrays:

```text
frame_indices
keypoints_image_norm
keypoint_scores
valid_mask
target_player_index
```

Expected source shapes are:

```text
keypoints_image_norm : [2, T, 17, 2]
keypoint_scores      : [2, T, 17]
valid_mask           : [2, T]
target_player_index  : scalar selecting player 0 or 1
```

`frame_indices` identifies the source-video frame represented by each temporal position.

### Coordinate convention

`keypoints_image_norm` stores image-normalized pose coordinates:

```text
x_image_norm = x_pixel / image_width
y_image_norm = y_pixel / image_height
```

The public dataset builder preserves these normalized coordinates in the generated MMAction2 annotation and assigns `img_shape=(1, 1)`. The formal recognition configurations then apply MMAction2 `PreNormalize2D`. Therefore, the coordinates actually presented to the subsequent skeleton feature pipeline are:

```text
x_model = 2 * x_image_norm - 1
y_model = 2 * y_image_norm - 1
```

This transformation centers the image coordinate system but does not perform root-joint centering or bounding-box-based spatial normalization.

The repository does not assume machine-specific paths. The pose root is supplied explicitly through `--pose-root`.

## Frozen temporal protocol

For each public-manifest row, the builder:

1. loads `<pose-root>/<sample_id>.npz`;
2. retains the interval from `source_start_frame` through `source_end_frame`, inclusive;
3. verifies that the retained count equals `retained_source_length`;
4. applies the frozen missing-coordinate and score handling implemented in `preprocessing/public_pose_dataset.py`;
5. linearly resamples the sequence to **48 frames**;
6. preserves the frozen validity handling;
7. selects the annotated target player;
8. writes an MMAction2 skeleton-recognition annotation.

The frozen benchmark contains **216** samples that require temporal cropping from the full NPZ representation before resampling. All 216 were independently verified to reconstruct exactly from the full NPZ files using the released frame bounds.

## MMAction2 output contract

The public builder is:

```text
preprocessing/build_mmaction2_dataset.py
```

Example:

```bash
python preprocessing/build_mmaction2_dataset.py \
  --manifest manifests/finebadminton_pcme_match_v1_manifest.csv \
  --pose-root /path/to/full_pose_npz \
  --output data/finebadminton_pcme_match_v1.pkl
```

For each sample, model-relevant arrays are emitted with shapes:

```text
keypoint       : [1, 48, 17, 2]
keypoint_score : [1, 48, 17]
```

The generated package contains MMAction2 `annotations` and the frozen `train` / `val` / `test` split lists.

## Reproducibility audit

Starting from the frozen two-player pose NPZ representation, the released preprocessing pipeline was compared against the frozen MMAction2 annotation PKL used in the paper.

The audit verified:

- **19,866 / 19,866** samples reproduced from the frozen two-player pose NPZ inputs;
- Train / Validation / Test sizes matched exactly;
- split order matched exactly;
- sample order matched exactly;
- labels and model-relevant scalar metadata matched;
- `keypoint` arrays matched exactly;
- `keypoint_score` arrays matched exactly;
- maximum absolute difference for both arrays was **0.0**.

The old internal annotation file contained one additional QC-only field, `clean_action`; it is intentionally omitted from the public reconstruction because it is not used as model input.

## What is and is not distributed

Distributed in this repository:

- frozen match-disjoint benchmark manifest;
- public preprocessing code;
- MMAction2 experiment configurations;
- PCME and Pair-only heads;
- evaluation and fixed 1:1 two-stream fusion code.

Not redistributed in this repository:

- original FineBadminton videos;
- private/local filesystem paths;
- training checkpoints;
- generated prediction dumps;
- internal experiment logs;
- large intermediate PKL / NPZ / NPY artifacts.

Users are responsible for obtaining the underlying dataset from the official FineBadminton source and for providing pose NPZ files that satisfy the documented input contract.
