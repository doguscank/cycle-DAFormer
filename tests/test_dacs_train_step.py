"""DACS train_step smoke test with synthetic batch (no datasets)."""

from __future__ import annotations

import torch

from mmseg.models.builder import build_train_model
from mmseg.utils.mmcv_shim.config import Config


def _synthetic_uda_batch(device, batch_size=1):
    h, w = 512, 512
    return dict(
        img=torch.randn(batch_size, 3, h, w, device=device),
        img_metas=[
            {
                "img_shape": (h, w, 3),
                "ori_shape": (h, w, 3),
                "pad_shape": (h, w, 3),
                "scale_factor": 1.0,
                "flip": False,
                "flip_direction": None,
                "img_norm_cfg": {
                    "mean": [123.675, 116.28, 103.53],
                    "std": [58.395, 57.12, 57.375],
                    "to_rgb": True,
                },
            }
        ]
        * batch_size,
        gt_semantic_seg=torch.randint(0, 19, (batch_size, 1, h, w), device=device),
        target_img=torch.randn(batch_size, 3, h, w, device=device),
        target_img_metas=[
            {
                "img_shape": (h, w, 3),
                "ori_shape": (h, w, 3),
                "pad_shape": (h, w, 3),
                "scale_factor": 1.0,
                "flip": False,
                "flip_direction": None,
                "img_norm_cfg": {
                    "mean": [123.675, 116.28, 103.53],
                    "std": [58.395, 57.12, 57.375],
                    "to_rgb": True,
                },
            }
        ]
        * batch_size,
    )


def test_dacs_train_step_cpu(gta2cs_model_cfg, encoder_ckpt_path, tmp_path):
    cfg = gta2cs_model_cfg
    cfg.model.pretrained = str(encoder_ckpt_path)
    cfg.model.train_cfg["work_dir"] = str(tmp_path)
    cfg.uda["debug_img_interval"] = 10**9
    cfg.uda["print_grad_magnitude"] = False
    model = build_train_model(cfg)
    model.init_weights()
    model.train()
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5)
    batch = _synthetic_uda_batch(device)
    log_vars = model.train_step(batch, optimizer)
    assert isinstance(log_vars, dict)
    optimizer.step()
