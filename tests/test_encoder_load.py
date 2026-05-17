"""MiT-B5 encoder load and forward smoke test."""

from __future__ import annotations

import torch

from mmseg.models.backbones.mix_transformer import mit_b5


def test_mit_b5_loads_encoder_and_forward(encoder_ckpt_path):
    backbone = mit_b5(pretrained=str(encoder_ckpt_path))
    backbone.eval()
    x = torch.randn(1, 3, 512, 512)
    feats = backbone(x)
    assert len(feats) == 4
    assert feats[-1].shape[1] == 512
    assert feats[-1].shape[-1] == 16
