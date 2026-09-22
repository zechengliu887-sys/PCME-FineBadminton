_base_ = './stgcn_standard_target_match_v1_formal.py'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='AAGCN',
        graph_cfg=dict(
            layout='coco',
            mode='spatial'
        ),
        gcn_attention=False,
        num_person=1
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=11,
        in_channels=256
    )
)

work_dir = (
    'outputs/mmaction2/'
    '2s_agcn_joint_target_match_v1_formal'
)

test_evaluator = dict(
    output_dir=(
        'outputs/mmaction2/'
        '2s_agcn_joint_target_match_v1_formal/'
        'test_reports'
    )
)
