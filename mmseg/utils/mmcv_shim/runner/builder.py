from __future__ import annotations

import copy

import torch

from mmseg.utils.mmcv_shim.utils import Registry, build_from_cfg
from .iter_based_runner import IterBasedRunner

RUNNERS = Registry("runner")
RUNNERS.register_module(module=IterBasedRunner)

OPTIMIZERS = Registry("optimizer")


def build_optimizer(model, cfg):
    cfg = cfg.copy()
    opt_type = cfg.pop("type")
    paramwise_cfg = cfg.pop("paramwise_cfg", None)
    base_lr = cfg.get("lr", 1e-4)
    base_wd = cfg.get("weight_decay", 0.0)
    params = _build_params(model, paramwise_cfg, base_lr, base_wd)
    if opt_type == "AdamW":
        return torch.optim.AdamW(params, **cfg)
    if opt_type == "SGD":
        return torch.optim.SGD(params, **cfg)
    if opt_type == "Adam":
        return torch.optim.Adam(params, **cfg)
    raise ValueError(f"Unsupported optimizer: {opt_type}")


def _build_params(model, paramwise_cfg, base_lr, base_wd):
    if paramwise_cfg is None:
        return model.parameters()
    custom_keys = paramwise_cfg.get("custom_keys", {})
    param_groups = []
    base_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        matched = False
        for key, spec in sorted(custom_keys.items(), key=lambda x: -len(x[0])):
            if key in name:
                lr_mult = spec.get("lr_mult", 1.0)
                decay_mult = spec.get("decay_mult", 1.0)
                param_groups.append(
                    {
                        "params": [p],
                        "lr": base_lr * lr_mult,
                        "weight_decay": base_wd * decay_mult,
                        "lr_mult": lr_mult,
                    }
                )
                matched = True
                break
        if not matched:
            base_params.append(p)
    if base_params:
        param_groups.insert(
            0,
            {
                "params": base_params,
                "lr": base_lr,
                "lr_mult": 1.0,
            },
        )
    return param_groups


def build_runner(cfg, default_args=None):
    return build_from_cfg(cfg, RUNNERS, default_args)
