"""Compatibility contracts for mmcv-free shims."""

from __future__ import annotations

import torch

from mmseg.utils.mmcv_shim.cnn import DepthwiseSeparableConvModule
from mmseg.utils.mmcv_shim.parallel import MMDataParallel


def test_depthwise_separable_conv_activates_depthwise_stage():
    block = DepthwiseSeparableConvModule(
        3,
        5,
        3,
        padding=1,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="ReLU"),
    )

    assert block.depthwise_conv.with_norm
    assert block.depthwise_conv.with_activation
    assert isinstance(block.depthwise_conv.activate, torch.nn.ReLU)
    assert block.pointwise_conv.with_activation


def test_mm_data_parallel_normalizes_cpu_test_batch():
    class Echo(torch.nn.Module):
        def forward(self, img, img_metas, return_loss=True):
            return img, img_metas, return_loss

    data = dict(
        img=[torch.zeros(1, 3, 4, 4)],
        img_metas=[[[dict(ori_shape=(4, 4, 3))]]],
    )

    imgs, img_metas, return_loss = MMDataParallel(Echo(), device_ids=[])(
        return_loss=False, **data
    )

    assert imgs[0].shape == (1, 3, 4, 4)
    assert img_metas == [[dict(ori_shape=(4, 4, 3))]]
    assert return_loss is False
