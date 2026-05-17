# mmcv.Config replacement with _base_ inheritance.

from __future__ import annotations

import copy
import json
import os
import os.path as osp
import sys
import tempfile
from importlib import import_module
from typing import Any

import yaml


class ConfigDict(dict):
    def __missing__(self, key):
        raise KeyError(key)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def _dict_to_configdict(d: dict) -> ConfigDict:
    cfg = ConfigDict()
    for k, v in d.items():
        if isinstance(v, dict):
            cfg[k] = _dict_to_configdict(v)
        elif isinstance(v, (list, tuple)):
            cfg[k] = type(v)(
                _dict_to_configdict(x) if isinstance(x, dict) else x for x in v
            )
        else:
            cfg[k] = v
    return cfg


def _merge_dict(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if k == "_delete_":
            continue
        if isinstance(v, dict) and v.get("_delete_") is True:
            new_v = {ik: iv for ik, iv in v.items() if ik != "_delete_"}
            a[k] = copy.deepcopy(new_v)
        elif k in a and isinstance(a[k], dict) and isinstance(v, dict):
            _merge_dict(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a


def _load_py_config(filename: str) -> dict:
    filename = osp.abspath(filename)
    cfg_dir = osp.dirname(filename)
    module_name = osp.splitext(osp.basename(filename))[0]
    if cfg_dir not in sys.path:
        sys.path.insert(0, cfg_dir)
    spec = import_module(module_name)
    cfg_dict = {k: v for k, v in spec.__dict__.items() if not k.startswith("__")}
    if "_base_" in cfg_dict:
        base_paths = cfg_dict.pop("_base_")
        if isinstance(base_paths, str):
            base_paths = [base_paths]
        base_cfg = {}
        for base in base_paths:
            base_path = base if osp.isabs(base) else osp.join(cfg_dir, base)
            base_cfg = _merge_dict(base_cfg, Config.fromfile(base_path)._cfg_dict)
        cfg_dict = _merge_dict(base_cfg, cfg_dict)
    return cfg_dict


class Config:
    def __init__(self, cfg_dict=None, filename=None):
        if cfg_dict is None:
            cfg_dict = {}
        elif not isinstance(cfg_dict, dict):
            raise TypeError("cfg_dict must be dict")
        self._cfg_dict = _dict_to_configdict(cfg_dict)
        self.filename = filename

    @property
    def pretty_text(self) -> str:
        return yaml.dump(self._cfg_dict_to_plain(), default_flow_style=False)

    def _cfg_dict_to_plain(self):
        out = {}
        for k, v in self._cfg_dict.items():
            if isinstance(v, ConfigDict):
                out[k] = self._cfg_dict_to_plain_helper(v)
            else:
                out[k] = v
        return out

    def _cfg_dict_to_plain_helper(self, d: ConfigDict):
        out = {}
        for k, v in d.items():
            if isinstance(v, ConfigDict):
                out[k] = self._cfg_dict_to_plain_helper(v)
            else:
                out[k] = v
        return out

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict[name]

    def __setitem__(self, name, value):
        self._cfg_dict[name] = value

    def __contains__(self, name):
        return name in self._cfg_dict

    def get(self, key, default=None):
        return self._cfg_dict.get(key, default)

    def merge_from_dict(self, options: dict):
        _merge_dict(self._cfg_dict, options)

    def dump(self, file=None):
        plain = self._cfg_dict_to_plain()
        if file is None:
            return yaml.dump(plain)
        if file.endswith(".json"):
            json.dump(plain, open(file, "w"), indent=2)
        else:
            yaml.dump(plain, open(file, "w"), default_flow_style=False)

    @classmethod
    def fromfile(cls, filename: str) -> "Config":
        filename = osp.abspath(filename)
        if filename.endswith(".json"):
            with open(filename) as f:
                cfg_dict = json.load(f)
        elif filename.endswith((".yml", ".yaml")):
            with open(filename) as f:
                cfg_dict = yaml.safe_load(f)
        elif filename.endswith(".py"):
            cfg_dict = _load_py_config(filename)
        else:
            raise ValueError(f"Unsupported config: {filename}")
        return cls(cfg_dict, filename=filename)
