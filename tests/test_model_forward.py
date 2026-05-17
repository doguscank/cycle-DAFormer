"""EncoderDecoder + DAFormerHead forward shape test."""

from __future__ import annotations

import torch

from mmseg.models import build_segmentor


def test_encoder_decoder_forward_shape(gta2cs_model_cfg, encoder_ckpt_path):
    cfg = gta2cs_model_cfg
    cfg.model.pretrained = str(encoder_ckpt_path)
    model = build_segmentor(cfg.model)
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    img_metas = [dict(
        ori_shape=(512, 512, 3),
        img_shape=(512, 512, 3),
        pad_shape=(512, 512, 3),
        scale_factor=1.0,
        flip=False,
        flip_direction=None,
    )]
    with torch.no_grad():
        out = model.encode_decode(x, img_metas)
    assert out.shape == (1, 19, 512, 512)
