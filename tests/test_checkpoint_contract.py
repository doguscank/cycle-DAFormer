"""Checkpoint structure contracts (torch only)."""

from __future__ import annotations

import torch


def test_encoder_checkpoint_keys(encoder_ckpt_path):
    ckpt = torch.load(encoder_ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    keys = list(state.keys())
    assert any("patch_embed" in k for k in keys)
    assert any("block" in k for k in keys)
    assert len(keys) > 100


def test_gta2cs_full_checkpoint_prefixes(gta2cs_checkpoint_path):
    ckpt = torch.load(gta2cs_checkpoint_path, map_location="cpu", weights_only=False)
    assert "state_dict" in ckpt
    sd = ckpt["state_dict"]
    prefixes = {k.split(".")[0] for k in sd}
    assert "model" in prefixes
    # Published checkpoints may strip EMA / imnet keys (see tools/publish_model.py).
    model_keys = [k for k in sd if k.startswith("model.decode_head")]
    assert len(model_keys) > 10
