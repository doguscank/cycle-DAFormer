# ---------------------------------------------------------------
# Full-memory cycle DAFormer: cycle training, original inference.
# ---------------------------------------------------------------

_base_ = ['gta2cs_uda_cycle_dacs_daformer_mitb5_s0.py']

uda = dict(
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    inference_mode='original',
    cycle_feature_indices=[2, 3],
    lambda_cycle_feat=1.0,
    lambda_source_cycle_seg=0.25,
    lambda_source_trans_seg=0.0,
    lambda_target_consistency=0.25,
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

name = 'gta2cs_uda_cycle_dacs_daformer_mitb5_s0_ablation_original_infer_levels23_balanced'
name_uda = 'cycle_dacs_a999_fd_things_original_infer_levels23_lcf1_lsc025_ltc025_guard'

