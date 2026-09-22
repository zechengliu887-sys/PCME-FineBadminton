_base_ = (
    '../badminton/'
    '2s_agcn_joint_target_match_v1_seed0.py'
)

custom_imports = dict(
    imports=[
        'mmaction_extensions.badminton_metric',
        'mmaction_extensions.pcme_cosine_head',
    ],
    allow_failed_imports=False,
)

model = dict(
    cls_head=dict(
        _delete_=True,
        type='CosineBidirectionalPairConditionedEvidenceHead',
        num_classes=11,
        in_channels=256,
        pair_dim=64,
        hidden_dim=128,
        temperature=8.0,
        dropout=0.0,
    )
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1,
    val_begin=1,
    val_interval=1,
)

work_dir = (
    'outputs/mmaction2/'
    '2s_agcn_joint_pcme_v12_seed0_smoke'
)

test_evaluator = dict(
    output_dir=(
        'outputs/mmaction2/'
        '2s_agcn_joint_pcme_v12_seed0_smoke/'
        'test_reports'
    )
)
