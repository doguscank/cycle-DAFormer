# Iteration-based training runner.

from __future__ import annotations

import os
import os.path as osp
from typing import Any, List, Optional

import torch

from mmseg.utils.checkpoint import load_checkpoint_file, unwrap_state_dict
from mmseg.utils.mmcv_shim.runner.dist import get_dist_info
from mmseg.utils.mmcv_shim.runner.hooks import (
    CheckpointHook,
    LrUpdaterHook,
    OptimizerHook,
    TextLoggerHook,
)


class LogBuffer:
    def __init__(self):
        self.val_history = {}
        self.n_history = {}
        self.output = {}

    def clear(self):
        self.output = {}

    def update(self, vars_dict, count=1):
        for k, v in vars_dict.items():
            if k not in self.val_history:
                self.val_history[k] = v
                self.n_history[k] = count
            else:
                self.val_history[k] += v
                self.n_history[k] += count

    def average(self, n):
        for k in self.val_history:
            self.output[k] = self.val_history[k] / max(self.n_history[k], 1)
        self.val_history = {}
        self.n_history = {}


class IterBasedRunner:
    def __init__(
        self,
        model,
        batch_processor=None,
        optimizer=None,
        work_dir=None,
        logger=None,
        meta=None,
        max_iters=None,
        **kwargs,
    ):
        self.model = model
        self.optimizer = optimizer
        self.work_dir = work_dir
        self.logger = logger
        self.meta = meta or {}
        self.max_iters = max_iters or kwargs.get("max_iters", 0)
        self.iter = 0
        self.epoch = 0
        self.timestamp = None
        self.log_buffer = LogBuffer()
        self.rank, self.world_size = get_dist_info()
        self._hooks: List[Any] = []

    def register_hook(self, hook, priority="NORMAL"):
        hook.runner = self
        self._hooks.append(hook)

    def register_training_hooks(
        self,
        lr_config,
        optimizer_config,
        checkpoint_config,
        log_config,
        momentum_config=None,
    ):
        if lr_config is not None:
            self.register_hook(LrUpdaterHook(**lr_config))
        if optimizer_config is not None:
            self.register_hook(OptimizerHook(**optimizer_config))
        if checkpoint_config is not None:
            self.register_hook(CheckpointHook(**checkpoint_config))
        if log_config is not None:
            for hook_cfg in log_config.get("hooks", []):
                cfg = dict(hook_cfg)
                hook_type = cfg.pop("type", None)
                if hook_type == "TextLoggerHook":
                    self.register_hook(TextLoggerHook(**cfg))

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def train(self, data_loader, max_iters):
        self.model.train()
        data_iter = iter(data_loader)
        while self.iter < max_iters:
            try:
                data_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                data_batch = next(data_iter)
            self.call_hook("before_train_iter")
            outputs = self.model.train_step(data_batch, self.optimizer)
            if outputs is not None:
                log_vars = outputs.get("log_vars", outputs)
                self.log_buffer.update(log_vars)
            self.call_hook("after_train_iter")
            self.iter += 1

    def run(self, data_loaders, workflow):
        max_iters = self.max_iters
        for phase, _ in workflow:
            if phase == "train":
                self.train(data_loaders[0], max_iters)

    def load_checkpoint(self, filename, map_location="cpu"):
        ckpt = load_checkpoint_file(filename, map_location=map_location)
        state_dict = unwrap_state_dict(ckpt)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict, strict=False)
        if "optimizer" in ckpt and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "meta" in ckpt and "iter" in ckpt["meta"]:
            self.iter = ckpt["meta"]["iter"]

    def resume(self, filename):
        self.load_checkpoint(filename)
