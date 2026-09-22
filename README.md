# Pair-Conditioned Motion Evidence for Fine-Grained Skeleton-Based Badminton Stroke Recognition

Official reproducibility package for PCME, a pair-conditioned refinement method for fine-grained skeleton-based badminton stroke recognition.

## Repository contents

- `mmaction_extensions/`: PCME heads and badminton evaluation metric
- `configs/`: formal experiment configurations
- `manifests/`: frozen match-disjoint benchmark manifest
- `preprocessing/`: public pose preprocessing and MMAction2 dataset builder
- `evaluation/`: prediction dumping and fixed two-stream fusion
- `environment/`: environment versions
- `docs/`: dataset and reproducibility notes

## Benchmark

The frozen benchmark contains 19,866 samples from 11 badminton stroke classes:

- Train: 14,279
- Validation: 2,894
- Test: 2,693

The split is match-disjoint.

## Data preparation

The preprocessing pipeline starts from extracted two-player pose NPZ files.

Example:

```bash
python preprocessing/build_mmaction2_dataset.py \
  --manifest manifests/finebadminton_pcme_match_v1_manifest.csv \
  --pose-root /path/to/full_pose_npz \
  --output data/finebadminton_pcme_match_v1.pkl
```

Each pose file must follow:

```text
<pose-root>/<sample_id>.npz
```

See `docs/DATASET.md` for the required NPZ schema.

## Training

This project uses MMAction2 1.2.0.

Example joint-stream PCME configuration:

```text
configs/research/2s_agcn_joint_pcme_v12_seed0_formal.py
```

Example bone-stream PCME configuration:

````text
configs/research/2s_agcn_bone_pcme_v12_seed0_formal.py
```

Three formal random seeds are provided for both streams.

## Two-stream fusion

Joint and bone streams use fixed 1:1 probability late fusion.

```bash
python evaluation/fuse_two_stream.py \
  --joint /path/to/joint_predictions.pkl \
  --bone /path/to/bone_predictions.pkl \
  --output-dir /path/to/output
```

Fusion weights are not tuned on the test set.

## Pair-only control

The repository also contains the parameter-matched Pair-only calibration control used in the validation-only ablation.

Pair-only configurations are explicitly marked `valonly`.

They should not be interpreted as test-set experiments.

## Reproducibility validation

The public preprocessing pipeline was checked against the frozen dataset used for the paper:

- 19,866 / 19,866 samples reproduced
- Train / Validation / Test ordering matched exactly
- keypoint arrays matched exactly
- keypoint-score arrays matched exactly
- maximum absolute array difference: 0.0

The public fixed 1:1 two-stream fusion script was also verified against the original experiment pipeline for all three seeds with exactly matching metrics.

## Data availability

Original FineBadminton videos are not redistributed in this repository.

Users must obtain the dataset separately from its original provider and comply with its license and terms of use.

## License

This repository is released under the Apache License 2.0. See `LICENSE` for details.

## Citation

Citation information will be added after publication.
