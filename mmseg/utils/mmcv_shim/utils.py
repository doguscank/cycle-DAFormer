# Minimal mmcv.utils replacements.

from __future__ import annotations

import argparse
import copy
import inspect
import logging
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Callable, Optional

import torch


class Registry:
    """Module registry (mmcv-compatible subset)."""

    def __init__(self, name: str, parent: Optional["Registry"] = None):
        self._name = name
        self._module_dict: dict = {}
        self.parent = parent

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, item):
        return self.get(item) is not None

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key: str):
        if key in self._module_dict:
            return self._module_dict[key]
        if self.parent is not None:
            return self.parent.get(key)
        return None

    def _register_module(self, module_class, module_name=None, force=False):
        if module_name is None:
            module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError(f"{module_name} is already registered in {self._name}")
        self._module_dict[module_name] = module_class
        return module_class

    def register_module(self, name=None, force=False, module=None):
        def _register(cls):
            self._register_module(cls, module_name=name, force=force)
            return cls

        if module is not None:
            return _register(module)
        return _register

    def build(self, cfg, default_args=None):
        return build_from_cfg(cfg, self, default_args)


def build_from_cfg(cfg: dict, registry: Registry, default_args: Optional[dict] = None):
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be dict, got {type(cfg)}")
    if "type" not in cfg:
        raise KeyError(f"`type` missing in cfg: {cfg}")
    args = cfg.copy()
    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in {registry._name}")
    else:
        obj_cls = obj_type
    if default_args is not None:
        for k, v in default_args.items():
            args.setdefault(k, v)
    return obj_cls(**args)


def is_str(x) -> bool:
    return isinstance(x, str)


def is_list_of(seq, expected_type) -> bool:
    if not isinstance(seq, (list, tuple)):
        return False
    return all(isinstance(item, expected_type) for item in seq)


def is_tuple_of(seq, expected_type) -> bool:
    if not isinstance(seq, tuple):
        return False
    return all(isinstance(item, expected_type) for item in seq)


def deprecated_api_warning(name_dict, cls_name=None):
    def decorator(func):
        return func

    return decorator


class DictAction(argparse.Action):
    """Parse KEY=VALUE pairs from CLI."""

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split("=", 1)
            options[key] = _parse_value(val)
        setattr(namespace, self.dest, options)


def _parse_value(val: str):
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    if val.startswith("[") and val.endswith("]"):
        return [_parse_value(x.strip()) for x in val[1:-1].split(",") if x.strip()]
    return val


def get_logger(
    name: str, log_file: Optional[str] = None, log_level: int = logging.INFO
):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    if log_file:
        abs_log_file = os.path.abspath(log_file)
        has_file_handler = any(
            isinstance(h, logging.FileHandler)
            and os.path.abspath(getattr(h, "baseFilename", "")) == abs_log_file
            for h in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


def print_log(msg, logger=None, level=logging.INFO):
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif isinstance(logger, str):
        if logger == "silent":
            pass
        else:
            logging.getLogger(logger).log(level, msg)
    else:
        raise TypeError(
            f"logger must be a logging.Logger, str, or None, got {type(logger)}"
        )


def get_git_hash(fallback: str = "") -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return fallback


def collect_env() -> dict:
    env = {
        "sys.platform": sys.platform,
        "Python": sys.version.replace("\n", ""),
        "CUDA available": str(torch.cuda.is_available()),
        "PyTorch": torch.__version__,
        "PyTorch compiling details": getattr(
            getattr(torch, "__config__", None), "show", lambda: ""
        )(),
        "TorchVision": import_module("torchvision").__version__,
        "OpenCV": import_module("cv2").__version__,
        "MMCV": "shim (none)",
    }
    if torch.cuda.is_available():
        env["GPU 0"] = torch.cuda.get_device_name(0)
    return env


# parrots_wrapper compatibility
_BatchNorm = torch.nn.modules.batchnorm._BatchNorm
