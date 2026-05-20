# ---------------------------------------------------------------
# Full-memory cycle DAFormer baseline with first-validation guard.
# ---------------------------------------------------------------

_base_ = ['gta2cs_uda_cycle_dacs_daformer_mitb5_s0.py']

evaluation = dict(
    interval=4000,
    metric='mIoU',
    first_eval_min_miou=0.5,
    first_eval_metric='mIoU',
    first_eval_kill=True,
)

name = 'gta2cs_uda_cycle_dacs_daformer_mitb5_s0_ablation_baseline_guard'
name_uda = 'cycle_dacs_a999_fd_things_rcs0.01_cpl_fadv001_guard'

