_base_ = './2s_agcn_bone_pcme_v12_formal.py'

randomness = dict(
    seed=0,
    deterministic=True,
)

work_dir = (
    'outputs/mmaction2/'
    '2s_agcn_bone_pcme_v12_seed0_formal'
)

test_evaluator = dict(
    output_dir=(
        'outputs/mmaction2/'
        '2s_agcn_bone_pcme_v12_seed0_formal/'
        'test_reports'
    )
)
