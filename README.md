# Pair-Conditioned Motion Evidence for Fine-Grained Skeleton-Based Badminton Stroke Recognition

Reproducibility package for **PCME (Pair-Conditioned Motion Evidence)**, a pair-conditioned refinement method for fine-grained skeleton-based badminton stroke recognition.

## Repository contents

- `mmaction_extensions/`: PCME heads and badminton evaluation metric
- `configs/`: formal experiment configurations
- `manifests/`: frozen match-disjoint benchmark manifest
- `preprocessing/`: public pose preprocessing and MMAction2 dataset builder
- `evaluation/`: prediction dumping and fixed two-stream fusion
- `environment/`: frozen software-version record
- `docs/`: dataset and reproducibility notes
- `third_party/`: minimal third-party configuration dependency retained for reproducibility

## Benchmark

The frozen benchmark contains **19,866** target-player stroke samples from **11** badminton stroke classes:

- Train: **14,279**
- Validation: **2,894**
- Test: **2,693**

The split is **match-disjoint**.

The public benchmark manifest is:

```text
manifests/finebadminton_pcme_match_v1_manifest.csv
```

## Source dataset

The benchmark is derived from **Finebadminton-20K**, the public subset of FineBadminton.

Official sources:

- Dataset: https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K
- Project page: https://finebadminton.github.io/FineBadminton/
- Paper: https://arxiv.org/abs/2508.07554

The original Finebadminton-20K release contains 70 full-match videos, 2,066 annotated rallies, and 20,757 annotated hit instances. This repository does **not** redistribute the original videos.

## Scope of this release

The released preprocessing pipeline starts from extracted **two-player pose NPZ files** rather than raw videos.

Expected layout:

```text
<pose-root>/<sample_id>.npz
```

Required NPZ arrays:

```text
frame_indices
keypoints_image_norm
keypoint_scores
valid_mask
target_player_index
```

Coordinate convention:

- `keypoints_image_norm` stores image-normalized coordinates `(x/W, y/H)`.
- The generated MMAction2 annotations retain these normalized coordinates and use `img_shape=(1, 1)`.
- The formal recognition configs subsequently apply MMAction2 `PreNormalize2D`, resulting in centered image coordinates `(2x/W - 1, 2y/H - 1)` before skeleton recognition.
- No additional root-centering or bounding-box-based spatial normalization is applied.

The public manifest specifies the retained temporal interval for each sample. All model inputs are resampled to **48 frames**, and only the target player is retained for recognition.

See `docs/DATASET.md` for the detailed data contract.

## Data preparation

Example:

```bash
python preprocessing/build_mmaction2_dataset.py \
  --manifest manifests/finebadminton_pcme_match_v1_manifest.csv \
  --pose-root /path/to/full_pose_npz \
  --output data/mmaction2/finebadminton_pose_clean_v3_target48_match_v1.pkl
```

The generated annotation PKL follows the MMAction2 skeleton-recognition format used by the formal experiments.

## Environment

The formal experiments used:

```text
Python 3.10
PyTorch 2.1.2+cu118
Torchvision 0.16.2+cu118
NumPy 1.26.4
OpenCV 4.11.0
MMCV 2.1.0
MMEngine 0.10.7
MMDetection 3.3.0
MMPose 1.3.2
MMAction2 1.2.0
```

The same information is recorded in `environment/versions.txt`. A compact recognition-side dependency list is provided in `environment/requirements-recognition.txt`. See `docs/INSTALL.md` for the CUDA-specific PyTorch installation, MMAction2 setup, dataset construction, training, and validation commands.

## Formal configurations

### Joint-stream PCME

```text
configs/research/2s_agcn_joint_pcme_v12_seed0_formal.py
configs/research/2s_agcn_joint_pcme_v12_seed1_formal.py
configs/research/2s_agcn_joint_pcme_v12_seed2_formal.py
```

### Bone-stream PCME

```text
configs/research/2s_agcn_bone_pcme_v12_seed0_formal.py
configs/research/2s_agcn_bone_pcme_v12_seed1_formal.py
configs/research/2s_agcn_bone_pcme_v12_seed2_formal.py
```

Formal 2s-AGCN baseline configurations for the same three seeds are provided under `configs/badminton/`.

## Frozen-split prediction dumping

`evaluation/dump_predictions.py` performs inference on either the frozen validation or test split and writes MMAction2 prediction dumps.

Validation example:

```bash
python evaluation/dump_predictions.py \
  --config configs/research/2s_agcn_joint_pcme_v12_seed0_formal.py \
  --checkpoint /path/to/joint_checkpoint.pth \
  --output outputs/predictions/pcme_joint_seed0_val.pkl \
  --split val \
  --expected-n 2894
```

Test example:

```bash
python evaluation/dump_predictions.py \
  --config configs/research/2s_agcn_joint_pcme_v12_seed0_formal.py \
  --checkpoint /path/to/joint_checkpoint.pth \
  --output outputs/predictions/pcme_joint_seed0_test.pkl \
  --split test \
  --expected-n 2693
```

Run the same commands with the corresponding bone config/checkpoint to obtain bone-stream dumps. Formal checkpoints are selected by validation Macro-F1; the test split is used only for final evaluation, not for checkpoint selection or tuning.

## Fixed two-stream fusion

Joint and bone streams use fixed **1:1 probability late fusion**.

Validation fusion:

```bash
python evaluation/fuse_two_stream.py \
  --joint outputs/predictions/pcme_joint_seed0_val.pkl \
  --bone outputs/predictions/pcme_bone_seed0_val.pkl \
  --output-dir outputs/fused/pcme_seed0_val \
  --expected-n 2894
```

Test fusion:

```bash
python evaluation/fuse_two_stream.py \
  --joint outputs/predictions/pcme_joint_seed0_test.pkl \
  --bone outputs/predictions/pcme_bone_seed0_test.pkl \
  --output-dir outputs/fused/pcme_seed0_test \
  --expected-n 2693
```

The fusion weight is frozen at 1:1 and is not searched or tuned on the test set.

## Pair-only control

The repository also contains the parameter-matched **Pair-only** calibration control used in the validation-only ablation.

Pair-only configurations are explicitly marked `valonly`.

They should **not** be interpreted as test-set experiments.

## Reproducibility validation

The public preprocessing pipeline was checked against the frozen dataset used for the paper, starting from the frozen two-player pose NPZ representation:

- **19,866 / 19,866** samples reproduced from the frozen two-player pose NPZ inputs
- Train / Validation / Test ordering matched exactly
- scalar training metadata matched
- keypoint arrays matched exactly
- keypoint-score arrays matched exactly
- maximum absolute array difference: **0.0**

The public fixed 1:1 two-stream fusion script was also checked against the original experiment pipeline for all three seeds, with exactly matching metrics.

The public PCME and Pair-only heads were additionally verified in a clean-room registry/model-build audit.

## Data availability

Original FineBadminton / Finebadminton-20K videos are not redistributed in this repository.

Users should obtain the dataset from the official Finebadminton-20K release and comply with its license and terms of use:

https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K

The machine-independent benchmark manifest used by this work is released in this repository.

## FineBadminton citation

```bibtex
@inproceedings{he2025finebadminton,
  title={Finebadminton: A Multi-Level Dataset for Fine-Grained Badminton Video Understanding},
  author={He, Xusheng and Liu, Wei and Ma, Shanshan and Liu, Qian and Ma, Chenghao and Wu, Jianlong},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={12776--12783},
  year={2025}
}
```

## License

This repository is released under the Apache License 2.0. See `LICENSE` for details.

Third-party notices are provided in `THIRD_PARTY_NOTICES.md`.

## Citation

Citation information for the PCME paper will be added after publication.
