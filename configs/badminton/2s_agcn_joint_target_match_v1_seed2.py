
_base_ = './2s_agcn_joint_target_match_v1_formal.py'

randomness = dict(
    seed=2,
    deterministic=True
)

work_dir = (
    'outputs/mmaction2/'
    '2s_agcn_joint_target_match_v1_seed2'
)

test_evaluator = dict(
    output_dir=(
        'outputs/mmaction2/'
        '2s_agcn_joint_target_match_v1_seed2/'
        'test_reports'
    )
)
