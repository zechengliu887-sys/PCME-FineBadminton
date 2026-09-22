_base_ = (
    '../badminton/'
    '2s_agcn_bone_target_match_v1_formal.py'
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
