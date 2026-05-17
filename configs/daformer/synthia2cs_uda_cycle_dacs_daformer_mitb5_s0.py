# ---------------------------------------------------------------
# Feature-cycle DAFormer on SYNTHIA -> Cityscapes.
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/uda_synthia_to_cityscapes_512x512.py',
    '../_base_/uda/cycle_dacs.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

seed = 0
uda = dict(
    alpha=0.999,
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15],
    imnet_feature_dist_scale_min_ratio=0.75,
)
data = dict(
    train=dict(
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')

name = 'synthia2cs_uda_cycle_dacs_daformer_mitb5_s0'
exp = 'cycle'
name_dataset = 'synthia2cityscapes'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'cycle_dacs_a999_fd_things_rcs0.01'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
