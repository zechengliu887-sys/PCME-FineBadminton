_base_ = '../../third_party/mmaction2/configs/_base_/default_runtime.py'

custom_imports = dict(
    imports=[
        'mmaction_extensions.badminton_metric',
    ],
    allow_failed_imports=False,
)

labels = [
    'block',
    'clear',
    'cross-court net shot',
    'drive',
    'drop shot',
    'kill',
    'net kill',
    'net lift',
    'net shot',
    'push shot',
    'serve',
]

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        graph_cfg=dict(
            layout='coco',
            mode='stgcn_spatial')),
    cls_head=dict(
        type='GCNHead',
        num_classes=11,
        in_channels=256))

dataset_type = 'PoseDataset'

ann_file = (
    'data/mmaction2/'
    'finebadminton_pose_clean_v3_'
    'target48_match_v1.pkl'
)

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(
        type='GenSkeFeat',
        dataset='coco',
        feats=['j']),
    dict(
        type='UniformSampleFrames',
        clip_len=48),
    dict(type='PoseDecode'),
    dict(
        type='FormatGCNInput',
        num_person=1),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(
        type='GenSkeFeat',
        dataset='coco',
        feats=['j']),
    dict(
        type='UniformSampleFrames',
        clip_len=48,
        num_clips=1,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(
        type='FormatGCNInput',
        num_person=1),
    dict(type='PackActionInputs')
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=train_pipeline,
        split='train'))

val_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='val',
        test_mode=True))

test_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='test',
        test_mode=True))

val_evaluator = dict(
    type='BadmintonClassificationMetric',
    labels=labels,
    detailed=False)

test_evaluator = dict(
    type='BadmintonClassificationMetric',
    labels=labels,
    detailed=True,
    output_dir=(
        'outputs/mmaction2/'
        'stgcn_standard_target_match_v1_seed0/'
        'test_reports'
    ))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=80,
    val_begin=1,
    val_interval=1)

val_cfg = dict(
    type='ValLoop')

test_cfg = dict(
    type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=80,
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True))

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        save_best='badminton/macro_f1',
        rule='greater',
        max_keep_ckpts=3),
    logger=dict(
        interval=20))

randomness = dict(
    seed=0,
    deterministic=True)

work_dir = (
    'outputs/mmaction2/'
    'stgcn_standard_target_match_v1_seed0'
)

auto_scale_lr = dict(
    enable=False,
    base_batch_size=64)
