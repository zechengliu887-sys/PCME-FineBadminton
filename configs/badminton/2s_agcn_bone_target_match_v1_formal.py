_base_ = './2s_agcn_joint_target_match_v1_formal.py'

bone_train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(
        type='GenSkeFeat',
        dataset='coco',
        feats=['b']
    ),
    dict(
        type='UniformSampleFrames',
        clip_len=48
    ),
    dict(type='PoseDecode'),
    dict(
        type='FormatGCNInput',
        num_person=1
    ),
    dict(type='PackActionInputs')
]

bone_eval_pipeline = [
    dict(type='PreNormalize2D'),
    dict(
        type='GenSkeFeat',
        dataset='coco',
        feats=['b']
    ),
    dict(
        type='UniformSampleFrames',
        clip_len=48,
        num_clips=1,
        test_mode=True
    ),
    dict(type='PoseDecode'),
    dict(
        type='FormatGCNInput',
        num_person=1
    ),
    dict(type='PackActionInputs')
]

train_pipeline = bone_train_pipeline
val_pipeline = bone_eval_pipeline
test_pipeline = bone_eval_pipeline

train_dataloader = dict(
    dataset=dict(
        pipeline=bone_train_pipeline
    )
)

val_dataloader = dict(
    dataset=dict(
        pipeline=bone_eval_pipeline
    )
)

test_dataloader = dict(
    dataset=dict(
        pipeline=bone_eval_pipeline
    )
)

work_dir = (
    'outputs/mmaction2/'
    '2s_agcn_bone_target_match_v1_formal'
)

test_evaluator = dict(
    output_dir=(
        'outputs/mmaction2/'
        '2s_agcn_bone_target_match_v1_formal/'
        'test_reports'
    )
)
