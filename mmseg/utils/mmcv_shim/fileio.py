# File I/O and progress utilities.

from __future__ import annotations

import json
import os
import os.path as osp
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Optional

import numpy as np
import yaml
from tqdm import tqdm


def mkdir_or_exist(path: str):
    os.makedirs(path, exist_ok=True)


def scandir(dir_path: str, suffix: Optional[str] = None, recursive: bool = False):
    """Yield paths relative to *dir_path* (mmcv-compatible)."""
    dir_path = osp.abspath(dir_path)
    if recursive:
        for root, _, files in os.walk(dir_path):
            for f in sorted(files):
                full = osp.join(root, f)
                if suffix is None or full.endswith(suffix):
                    yield osp.relpath(full, dir_path)
    else:
        for f in sorted(os.listdir(dir_path)):
            full = osp.join(dir_path, f)
            if osp.isfile(full) and (suffix is None or full.endswith(suffix)):
                yield f


def list_from_file(filename: str, prefix: str = "", offset: int = 0, max_num: int = 0):
    with open(filename) as f:
        lines = f.read().splitlines()
    lines = [prefix + x for x in lines[offset:]]
    if max_num > 0:
        lines = lines[:max_num]
    return lines


def load(file_path: str):
    if file_path.endswith((".yml", ".yaml")):
        with open(file_path) as f:
            return yaml.safe_load(f)
    if file_path.endswith(".json"):
        with open(file_path) as f:
            return json.load(f)
    if file_path.endswith((".pkl", ".pickle")):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return np.load(file_path, allow_pickle=True)


def dump(obj: Any, file_path: str):
    mkdir_or_exist(osp.dirname(file_path) or ".")
    if file_path.endswith((".pkl", ".pickle")):
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    elif file_path.endswith(".json"):
        with open(file_path, "w") as f:
            json.dump(obj, f, indent=2)
    else:
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)


class FileClient:
    """Disk-only file client."""

    def __init__(self, backend="disk", **kwargs):
        self.backend = backend

    def get(self, filepath: str) -> bytes:
        with open(filepath, "rb") as f:
            return f.read()


class ProgressBar:
    def __init__(self, task_num: int):
        self.task_num = task_num
        self.completed = 0
        self._pbar = tqdm(total=task_num)

    def update(self, n: int = 1):
        self.completed += n
        self._pbar.update(n)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._pbar.close()


def track_progress(func: Callable, tasks: Iterable, nproc: int = 1):
    tasks = list(tasks)
    if nproc <= 1:
        return [func(t) for t in tqdm(tasks)]
    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        futures = {ex.submit(func, t): i for i, t in enumerate(tasks)}
        for fut in tqdm(as_completed(futures), total=len(tasks)):
            i = futures[fut]
            results[i] = fut.result()
    return results


def track_parallel_progress(func, tasks, nproc):
    return track_progress(func, tasks, nproc=nproc)
