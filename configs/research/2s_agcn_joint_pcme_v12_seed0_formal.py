_base_ = (
    './2s_agcn_joint_pcme_v12_seed0_smoke.py'
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=80,
    val_begin=1,
    val_interval=1,
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='badminton/macro_f1',
        rule='greater',
    )
)

work_dir = (
    'outputs/mmaction2/'
    '2s_agcn_joint_pcme_v12_seed0_formal'
)

test_evaluator = dict(
    output_dir=(
        'outputs/mmaction2/'
        '2s_agcn_joint_pcme_v12_seed0_formal/'
        'test_reports'
    )
)
