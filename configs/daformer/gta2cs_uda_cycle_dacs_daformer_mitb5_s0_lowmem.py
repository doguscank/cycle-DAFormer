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
    # Low-memory training cannot keep the ImageNet feature-distance branch, so
    # keep cycle active but use it as a light regularizer. Stronger cycle/fadv
    # settings over-regularized early runs and kept validation mIoU near 16-17.
    inference_mode='target2source',
    cycle_feature_indices=[2, 3],
    target_consistency_teacher_no_grad=True,
    lambda_cycle_feat=0.1,
    lambda_source_cycle_seg=0.05,
    lambda_source_trans_seg=0.0,
    lambda_target_consistency=0.1,
    target_consistency_threshold=0.9,
    enable_feature_adversarial=False,
    lambda_feature_adv=0.0,
)

name = 'gta2cs_uda_cycle_dacs_daformer_mitb5_s0_lowmem'
name_uda = 'cycle_dacs_lowmem_no_fd_levels23_lcf01_lsc005_ltc01_thr09'
