# CNN building blocks.

from __future__ import annotations

import inspect
from typing import Optional

import torch
import torch.nn as nn

from .registry import MODELS


def _build_group_norm(num_features, **kwargs):
    num_groups = kwargs.pop("num_groups", 32)
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_features, **kwargs)


NORM_LAYERS = {
    "BN": nn.BatchNorm2d,
    "SyncBN": nn.SyncBatchNorm,
    "GN": _build_group_norm,
}
# mmcv-compatible module names for norm layers (used in published checkpoints).
NORM_ABBR = {
    "BN": "bn",
    "SyncBN": "bn",
    "GN": "gn",
}
ACT_LAYERS = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "GELU": nn.GELU,
    "Sigmoid": nn.Sigmoid,
    None: None,
}


def build_conv_layer(cfg: Optional[dict], *args, **kwargs):
    if cfg is None:
        cfg = dict(type="Conv2d")
    cfg = cfg.copy()
    layer_type = cfg.pop("type")
    if layer_type == "Conv2d":
        return nn.Conv2d(*args, **kwargs, **cfg)
    raise NotImplementedError(layer_type)


def build_norm_layer(cfg: dict, num_features: int, postfix: str = ""):
    cfg = cfg.copy()
    layer_type = cfg.pop("type")
    requires_grad = cfg.pop("requires_grad", True)
    norm_cls = NORM_LAYERS.get(layer_type, nn.BatchNorm2d)
    layer = norm_cls(num_features, **cfg)
    for p in layer.parameters():
        p.requires_grad = requires_grad
    abbr = NORM_ABBR.get(layer_type, layer_type.lower())
    name = abbr + postfix
    return name, layer


def build_plugin_layer(cfg, in_channels, postfix=""):
    raise NotImplementedError("plugin layers not used in DAFormer path")


def constant_init(module, val=0, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class ConvModule(nn.Module):
    """Conv-Norm-Act block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        inplace=True,
        order=("conv", "norm", "act"),
    ):
        super().__init__()
        self.order = order
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == "auto":
            bias = norm_cfg is None
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm_name = None
        if self.with_norm:
            norm_name, norm = build_norm_layer(norm_cfg, out_channels)
            self.add_module(norm_name, norm)
            self.norm_name = norm_name
        if self.with_activation:
            act_cfg = act_cfg.copy()
            act_type = act_cfg.pop("type")
            act_cls = ACT_LAYERS.get(act_type, nn.ReLU)
            if "inplace" in inspect.signature(act_cls).parameters:
                self.activate = act_cls(inplace=inplace, **act_cfg)
            else:
                self.activate = act_cls(**act_cfg)
        else:
            self.activate = None

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        return None

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x


class DepthwiseSeparableConvModule(nn.Module):
    """mmcv-compatible depthwise separable conv block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        dw_norm_cfg="default",
        dw_act_cfg="default",
        pw_norm_cfg="default",
        pw_act_cfg="default",
    ):
        super().__init__()
        if dw_norm_cfg == "default":
            dw_norm_cfg = norm_cfg
        if dw_act_cfg == "default":
            dw_act_cfg = act_cfg
        if pw_norm_cfg == "default":
            pw_norm_cfg = norm_cfg
        if pw_act_cfg == "default":
            pw_act_cfg = act_cfg

        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            conv_cfg=conv_cfg,
            norm_cfg=dw_norm_cfg,
            act_cfg=dw_act_cfg,
        )
        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            bias=bias,
            conv_cfg=conv_cfg,
            norm_cfg=pw_norm_cfg,
            act_cfg=pw_act_cfg,
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        return self.pointwise_conv(x)
