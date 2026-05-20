# ---------------------------------------------------------------
# Full-memory cycle DAFormer: all cycle levels, light losses.
# ---------------------------------------------------------------

_base_ = ['gta2cs_uda_cycle_dacs_daformer_mitb5_s0.py']

uda = dict(
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    cycle_feature_indices=[0, 1, 2, 3],
    lambda_cycle_feat=0.1,
    lambda_source_cycle_seg=0.05,
    lambda_source_trans_seg=0.0,
    lambda_target_consistency=0.1,
    target_consistency_threshold=0.9,
    enable_feature_adversarial=False,
    lambda_feature_adv=0.0,
)

evaluation = dict(
    interval=4000,
    metric='mIoU',
    first_eval_min_miou=0.5,
    first_eval_metric='mIoU',
    first_eval_kill=True,
)

name = 'gta2cs_uda_cycle_dacs_daformer_mitb5_s0_ablation_alllevels_light'
name_uda = 'cycle_dacs_a999_fd_things_alllevels_lcf01_lsc005_ltc01_thr09_guard'

