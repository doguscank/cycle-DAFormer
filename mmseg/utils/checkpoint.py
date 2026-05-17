# Copyright (c) cycle-DAFormer contributors.
"""Checkpoint loading utilities (mmcv-free)."""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


def load_checkpoint_file(path: str, map_location: str = 'cpu') -> dict:
    """Load a checkpoint file and return the raw dict."""
    return torch.load(path, map_location=map_location, weights_only=False)


def unwrap_state_dict(checkpoint: dict) -> dict:
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        return checkpoint['model']
    return checkpoint


def revise_state_dict(state_dict: dict,
                      revise_keys: Optional[Sequence[Tuple[str, str]]] = None
                      ) -> OrderedDict:
    if revise_keys is None:
        revise_keys = []
    new_sd = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        for pattern, repl in revise_keys:
            new_key = re.sub(pattern, repl, new_key)
        new_sd[new_key] = value
    return new_sd


def load_state_dict(model: nn.Module,
                    checkpoint: Union[str, dict],
                    revise_keys: Optional[Sequence[Tuple[str, str]]] = None,
                    strict: bool = False) -> dict:
    if isinstance(checkpoint, str):
        checkpoint = load_checkpoint_file(checkpoint, map_location='cpu')
    state_dict = unwrap_state_dict(checkpoint)
    if revise_keys:
        state_dict = revise_state_dict(state_dict, revise_keys)
    return model.load_state_dict(state_dict, strict=strict)


def load_checkpoint(model: nn.Module,
                    filename: str,
                    map_location: str = 'cpu',
                    strict: bool = False,
                    revise_keys: Optional[Sequence[Tuple[str, str]]] = None
                    ) -> dict:
    """Load weights into *model*; returns checkpoint dict (with meta)."""
    checkpoint = load_checkpoint_file(filename, map_location=map_location)
    load_state_dict(
        model,
        checkpoint,
        revise_keys=revise_keys,
        strict=strict,
    )
    return checkpoint


def _load_checkpoint(path: str, logger=None, map_location: str = 'cpu') -> dict:
    """Backbone init helper (mmcv-compatible name)."""
    if logger is not None:
        logger.info(f'load checkpoint from {path}')
    return load_checkpoint_file(path, map_location=map_location)
