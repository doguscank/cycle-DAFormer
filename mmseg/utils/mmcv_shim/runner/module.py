# BaseModule and mixed-precision helpers.

from __future__ import annotations

import functools
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg

    def init_weights(self):
        if self.init_cfg is None:
            return
        cfgs = self.init_cfg if isinstance(self.init_cfg, list) else [self.init_cfg]
        for cfg in cfgs:
            cfg = cfg.copy()
            init_type = cfg.pop('type', None)
            override = cfg.pop('override', None)
            if override:
                name = override['name']
                module = dict(self.named_modules())[name]
                _apply_init(module, init_type, cfg)
            else:
                _apply_init(self, init_type, cfg)


def _apply_init(module, init_type, cfg):
    if init_type == 'Normal':
        std = cfg.get('std', 0.01)
        for m in module.modules() if hasattr(module, 'modules') else [module]:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, std=std)
    elif init_type == 'Constant':
        val = cfg.get('val', 0)
        for m in module.modules():
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, val)


def auto_fp16(apply_to=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, 'fp16_enabled', False):
                with torch.cuda.amp.autocast(enabled=True):
                    return func(self, *args, **kwargs)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def force_fp32(apply_to=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def wrap_fp16_model(model):
    model.fp16_enabled = True
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True


class Sequential(nn.Sequential, BaseModule):
    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)
