# Auto-generated and frozen by P5c.
# Reviewer-motivated Pair-only calibration control.
#
# IMPORTANT:
# - Validation-only mechanism experiment.
# - Held-out Test must not be evaluated.
# - H=716 was selected only by parameter matching.
#
_base_ = ['./2s_agcn_joint_pcme_v12_seed0_formal.py']

custom_imports = dict(
    imports=[
        'mmaction_extensions.badminton_metric',
        'mmaction_extensions.pcme_cosine_head',
        'mmaction_extensions.pcme_paironly_head'
    ],
    allow_failed_imports=False,
)

model = dict(
    cls_head=dict(
        _delete_=True,
        type='PairOnlyTop2CalibrationHead',
        num_classes=11,
        in_channels=256,
        pair_dim=64,
        hidden_dim=716,
        dropout=0.0,
    )
)

work_dir = 'work_dirs/2s_agcn_joint_paironly_calibration_seed0_valonly'