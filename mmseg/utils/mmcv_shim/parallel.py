# Data parallel and collate utilities.

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from mmseg.utils.mmcv_shim.utils import Registry

MODULE_WRAPPERS = Registry("module_wrapper")


class DataContainer:
    def __init__(self, data, stack=False, padding_value=0, cpu_only=False):
        self.data = data
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only


def collate(batch, samples_per_gpu=1):
    """Collate function compatible with mmseg pipelines."""
    if not batch:
        return batch
    elem = batch[0]
    if isinstance(elem, DataContainer):
        stacked = []
        for i in range(0, len(batch), samples_per_gpu):
            group = batch[i : i + samples_per_gpu]
            if group[0].cpu_only:
                stacked.append([g.data for g in group])
            elif group[0].stack:
                if isinstance(group[0].data, torch.Tensor):
                    stacked.append(torch.stack([g.data for g in group], 0))
                else:
                    stacked.append([g.data for g in group])
            else:
                stacked.append([g.data for g in group])
        return stacked
    if isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)):
        return [collate(samples, samples_per_gpu) for samples in zip(*batch)]
    if isinstance(elem, Mapping):
        return {key: collate([d[key] for d in batch], samples_per_gpu) for key in elem}
    return default_collate(batch)


def default_collate(batch):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    if isinstance(elem, (int, float)):
        return torch.tensor(batch)
    if isinstance(elem, str):
        return batch
    if isinstance(elem, Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    if isinstance(elem, tuple) and hasattr(elem, "_fields"):
        return elem.__class__(*(default_collate(samples) for samples in zip(*batch)))
    if isinstance(elem, Sequence):
        return [default_collate(samples) for samples in zip(*batch)]
    return batch


def scatter(inputs, target_gpus, dim=0):
    if isinstance(inputs, torch.Tensor):
        return inputs.cuda(target_gpus[0], non_blocking=True)
    if isinstance(inputs, (list, tuple)):
        return type(inputs)(scatter(x, target_gpus, dim) for x in inputs)
    if isinstance(inputs, dict):
        return {k: scatter(v, target_gpus, dim) for k, v in inputs.items()}
    return inputs


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    inputs = scatter(inputs, target_gpus, dim) if inputs is not None else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else {}
    inputs = _ensure_list(inputs)
    kwargs = _ensure_list(kwargs)
    return inputs, kwargs


def _ensure_list(obj):
    if isinstance(obj, list):
        return obj
    return [obj]


def _normalize_img_metas(img_metas):
    """Convert collated img_metas to List[List[dict]] for forward_test."""
    if not isinstance(img_metas, list) or not img_metas:
        return img_metas
    inner = img_metas[0]
    while isinstance(inner, list) and len(inner) == 1 and isinstance(inner[0], list):
        inner = inner[0]
    if isinstance(inner, list) and inner and isinstance(inner[0], dict):
        return [inner]
    return img_metas


def _normalize_collated_batch(data: dict, device=None) -> dict:
    """Normalize collated mmseg batches and optionally move tensors."""
    out = {}
    for key, val in data.items():
        if key == "img_metas":
            out[key] = _normalize_img_metas(val)
            continue
        if isinstance(val, list) and val:
            item = val[0]
            if isinstance(item, torch.Tensor):
                out[key] = [item.to(device) if device is not None else item]
            elif isinstance(item, DataContainer) and isinstance(
                item.data, torch.Tensor
            ):
                tensor = item.data.to(device) if device is not None else item.data
                out[key] = [tensor]
            else:
                out[key] = val
        else:
            out[key] = val
    return out


def unwrap_train_batch(data: dict, device=None) -> dict:
    """Flatten collated batch for training (UDA / segmentor train_step)."""
    out = {}
    for key, val in data.items():
        if not isinstance(val, list) or not val:
            out[key] = val
            continue
        item = val[0]
        if isinstance(item, DataContainer):
            payload = item.data
            if isinstance(payload, torch.Tensor):
                out[key] = payload.to(device) if device is not None else payload
            else:
                out[key] = payload
        elif isinstance(item, torch.Tensor):
            out[key] = item.to(device) if device is not None else item
        else:
            out[key] = item
    return out


class MMDataParallel(DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids)

    def forward(self, *inputs, return_loss=True, **kwargs):
        device = self.device_ids[0] if self.device_ids else None
        if device is not None and kwargs:
            kwargs = _normalize_collated_batch(kwargs, device)
        elif kwargs:
            kwargs = _normalize_collated_batch(kwargs)
        return self.module(*inputs, return_loss=return_loss, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        device = self.device_ids[0] if self.device_ids else None
        data_batch = unwrap_train_batch(data_batch, device)
        return self.module.train_step(data_batch, optimizer, **kwargs)


class MMDistributedDataParallel(DistributedDataParallel):
    def train_step(self, *inputs, **kwargs):
        return self.module.train_step(*inputs, **kwargs)

    def val_step(self, *inputs, **kwargs):
        return self.module.val_step(*inputs, **kwargs)
