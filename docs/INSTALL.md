# Installation and training

This document describes the recognition-side environment used by the released PCME reproducibility package.

The public repository starts from the documented two-player pose NPZ representation. RTMPose/MMDetection/MMPose are therefore not required to rebuild the released MMAction2 annotation PKL or to train the released recognition models.

## 1. Frozen recognition environment

The formal experiments used:

```text
Python 3.10
PyTorch 2.1.2+cu118
Torchvision 0.16.2+cu118
NumPy 1.26.4
MMCV 2.1.0
MMEngine 0.10.7
MMAction2 1.2.0
```

The original research environment also contained MMDetection 3.3.0, MMPose 1.3.2, and OpenCV 4.11.0 for pose-related work. They are not required by the released pose-NPZ-to-recognition pipeline itself.

## 2. Create the environment

A Linux environment with an NVIDIA GPU is recommended.

```bash
conda create -n pcme python=3.10 -y
conda activate pcme

pip install torch==2.1.2 torchvision==0.16.2   --index-url https://download.pytorch.org/whl/cu118

pip install numpy==1.26.4
pip install -U openmim
mim install "mmengine==0.10.7"
mim install "mmcv==2.1.0"
pip install scikit-learn
```

Clone and install the MMAction2 version used by the experiments:

```bash
git clone --branch v1.2.0 --depth 1   https://github.com/open-mmlab/mmaction2.git ../mmaction2

pip install -v -e ../mmaction2
```

Verify the core versions:

```bash
python -c "import torch, numpy, mmcv, mmengine, mmaction; print('torch', torch.__version__); print('numpy', numpy.__version__); print('mmcv', mmcv.__version__); print('mmengine', mmengine.__version__); print('mmaction2', mmaction.__version__)"
```

## 3. Build the frozen MMAction2 annotation PKL

Run the following commands from the root of this repository.

Create the expected output directory:

```bash
mkdir -p data/mmaction2
```

Build the annotation file:

```bash
python preprocessing/build_mmaction2_dataset.py   --manifest manifests/finebadminton_pcme_match_v1_manifest.csv   --pose-root /path/to/full_pose_npz   --output data/mmaction2/finebadminton_pose_clean_v3_target48_match_v1.pkl
```

The formal configs expect exactly:

```text
data/mmaction2/finebadminton_pose_clean_v3_target48_match_v1.pkl
```

## 4. Training setup

Run training commands from the repository root so that relative dataset and output paths resolve consistently.

Expose the repository's custom MMAction2 modules:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

For deterministic CUDA execution used by the formal runs:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## 5. Train PCME

### Seed 0

Joint stream:

```bash
python ../mmaction2/tools/train.py   configs/research/2s_agcn_joint_pcme_v12_seed0_formal.py
```

Bone stream:

```bash
python ../mmaction2/tools/train.py   configs/research/2s_agcn_bone_pcme_v12_seed0_formal.py
```

### Seeds 1 and 2

Replace `seed0` in the two commands above with `seed1` or `seed2`.

The released formal configs use 80 epochs and validation-based best-checkpoint selection.

## 6. Train the 2s-AGCN baseline

The corresponding joint and bone baseline configs are provided under:

```text
configs/badminton/
```

For example, seed 0:

```bash
python ../mmaction2/tools/train.py   configs/badminton/2s_agcn_joint_target_match_v1_seed0.py

python ../mmaction2/tools/train.py   configs/badminton/2s_agcn_bone_target_match_v1_seed0.py
```

## 7. Frozen-split prediction dumps

Use `evaluation/dump_predictions.py` for formal validation or test inference. The released benchmark contains 2,894 validation samples and 2,693 test samples.

Joint-stream validation:

```bash
python evaluation/dump_predictions.py \
  --config configs/research/2s_agcn_joint_pcme_v12_seed0_formal.py \
  --checkpoint /path/to/joint_checkpoint.pth \
  --output outputs/predictions/pcme_joint_seed0_val.pkl \
  --split val \
  --expected-n 2894
```

Joint-stream test:

```bash
python evaluation/dump_predictions.py \
  --config configs/research/2s_agcn_joint_pcme_v12_seed0_formal.py \
  --checkpoint /path/to/joint_checkpoint.pth \
  --output outputs/predictions/pcme_joint_seed0_test.pkl \
  --split test \
  --expected-n 2693
```

Run the same commands with the corresponding bone config/checkpoint to obtain the bone-stream dumps. Formal checkpoints are selected by validation Macro-F1; the test split is used only for final evaluation.

## 8. Fixed two-stream fusion

Fuse joint and bone class probabilities using the frozen 1:1 protocol.

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

The fusion weight is fixed. It is not searched or tuned on the test set.

## 9. Pair-only control

Pair-only configs are provided under `configs/research/` and are explicitly named `valonly`.

They reproduce the parameter-matched validation-only control used in the paper. They should not be interpreted as test-set experiments.

## 10. Scope

This release reproduces the benchmark construction and recognition experiments from the extracted two-player pose NPZ representation onward.

It does not claim an end-to-end raw-video-to-pose reproduction pipeline.
