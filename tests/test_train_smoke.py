"""Short DACS training smoke test (10 iterations)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from mmseg.apis import train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models.builder import build_train_model
from mmseg.utils.mmcv_shim.config import Config


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.requires_gpu
def test_dacs_smoke_train_10_iters(encoder_ckpt_path, cityscapes_val_available):
    data_root = Path(os.environ.get('DAFORMER_DATA_ROOT', ''))
    gta_img = data_root / 'gta' / 'images'
    if not gta_img.is_dir() or not cityscapes_val_available:
        pytest.skip('GTA + Cityscapes required for smoke train')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')

    cfg = Config.fromfile(str(
        Path(__file__).resolve().parents[1]
        / 'configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py'))
    cfg.model.pretrained = str(encoder_ckpt_path)
    cfg.runner.max_iters = 10
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0
    cfg.evaluation = dict(interval=1000, metric='mIoU')
    cfg.checkpoint_config = dict(by_epoch=False, interval=10, max_keep_ckpts=1)
    import tempfile
    cfg.work_dir = tempfile.mkdtemp(prefix='daformer_smoke_')
    cfg.model.train_cfg.work_dir = cfg.work_dir
    cfg.gpu_ids = [0]

    model = build_train_model(cfg)
    model.init_weights()
    dataset = build_dataset(cfg.data.train)
    model.CLASSES = dataset.CLASSES
    train_segmentor(model, [dataset], cfg, distributed=False, validate=False)
