_base_ = './2s_agcn_joint_pcme_v12_seed0_formal.py'

randomness = dict(
    seed=1,
    deterministic=True,
)

work_dir = (
    'outputs/mmaction2/'
    '2s_agcn_joint_pcme_v12_seed1_formal'
)

test_evaluator = dict(
    output_dir=(
        'outputs/mmaction2/'
        '2s_agcn_joint_pcme_v12_seed1_formal/'
        'test_reports'
    )
)
