# ---------------------------------------------------------------
# Feature-cycle UDA on top of DAFormer/DACS.
# ---------------------------------------------------------------

uda = dict(
    type='CycleDACS',
    alpha=0.99,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,
    print_grad_magnitude=False,
    cycle_feature_indices=[0, 1, 2, 3],
    translator_hidden_channels=None,
    lambda_cycle_feat=1.0,
    lambda_source_cycle_seg=1.0,
    lambda_source_trans_seg=0.0,
    lambda_target_consistency=1.0,
    cycle_feature_loss_type='l1',
    normalize_cycle_features=True,
    target_consistency_loss_type='kl',
    target_consistency_threshold=None,
    target_consistency_detach_teacher=True,
    target_consistency_teacher_no_grad=True,
    enable_feature_adversarial=False,
    lambda_feature_adv=0.0,
    feature_adv_loss_type='bce',
    feature_disc_base_channels=64,
    inference_mode='target2source',
)
use_ddp_wrapper = True
