"""Load published GTA2CS checkpoint into eval segmentor."""

from __future__ import annotations

import torch

from mmseg.models import build_segmentor
from mmseg.utils.checkpoint import load_checkpoint
from mmseg.utils.legacy_cfg import update_legacy_cfg
from mmseg.utils.mmcv_shim.config import Config


def test_load_gta2cs_latest_into_model(gta2cs_config_path, gta2cs_checkpoint_path):
    cfg = update_legacy_cfg(Config.fromfile(str(gta2cs_config_path)))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(
        model,
        str(gta2cs_checkpoint_path),
        revise_keys=[(r"^module\.", ""), (r"^model\.", "")],
        strict=False,
    )
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    img_metas = [
        dict(
            ori_shape=(512, 512, 3),
            img_shape=(512, 512, 3),
            pad_shape=(512, 512, 3),
            scale_factor=1.0,
            flip=False,
            flip_direction=None,
        )
    ]
    with torch.no_grad():
        out = model.encode_decode(x, img_metas)
    assert out.shape[1] == 19
