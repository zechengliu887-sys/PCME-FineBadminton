# Dataset and benchmark protocol

This repository does not redistribute the original FineBadminton videos.

The frozen PCME benchmark contains 19,866 target-player stroke samples from 11 stroke classes.

Split sizes:

- Train: 14,279
- Validation: 2,894
- Test: 2,693

The split is match-disjoint.

The public benchmark manifest is:

`manifests/finebadminton_pcme_match_v1_manifest.csv`

Each row provides:

- sample_id
- stroke_class
- match_id
- split
- target_side
- source_start_frame
- source_end_frame
- retained_source_length
- model_target_frames
- temporal_policy

No local filesystem paths are stored in the public manifest.

## Pose input

The public preprocessing pipeline expects one full two-player pose NPZ per sample:

`<pose-root>/<sample_id>.npz`

Required NPZ arrays are:

- frame_indices
- keypoints_image_norm
- keypoint_scores
- valid_mask
- target_player_index

The public manifest determines the retained temporal interval. Samples that require trimming are reconstructed from the full NPZ before temporal resampling.

All model inputs are resampled to 48 frames.

The released preprocessing pipeline was verified against the frozen dataset used in the paper. All 19,866 samples, split order, labels, keypoint arrays, and keypoint-score arrays matched exactly.

Original FineBadminton data must be obtained separately from its official source and used according to its license and terms.
