# ---------------------------------------------------------------
# 16 GB VRAM-friendly feature-cycle DAFormer on GTA5 -> Cityscapes.
# ---------------------------------------------------------------

_base_ = ['gta2cs_uda_cycle_dacs_daformer_mitb5_s0.py']

data = dict(samples_per_gpu=1, workers_per_gpu=2)

uda = dict(
    # Feature distance keeps an extra ImageNet model and source feature graph.
    # Disable it for 16 GB fine-tuning; re-enable only if VRAM allows.
    imnet_feature_dist_lambda=0.0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    # The deepest two levels carry the most semantic signal with much smaller
    # spatial maps than levels 0/1.
    cycle_feature_indices=[2, 3],
    target_consistency_teacher_no_grad=True,
    # Rebalanced from early training logs: source-cycle CE was already a large
    # auxiliary term, while feature-cycle and target consistency were too weak.
    lambda_cycle_feat=3.0,
    lambda_source_cycle_seg=0.15,
    lambda_source_trans_seg=0.0,
    lambda_target_consistency=1.0,
    enable_feature_adversarial=True,
    lambda_feature_adv=0.01,
)

name = 'gta2cs_uda_cycle_dacs_daformer_mitb5_s0_lowmem'
name_uda = 'cycle_dacs_lowmem_no_fd_levels23_fadv001'
