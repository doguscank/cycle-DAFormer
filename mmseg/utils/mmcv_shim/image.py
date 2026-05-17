# Image utilities (cv2-backed).

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import numpy as np


def imread(img_or_path, flag="color", backend=None) -> np.ndarray:
    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    if flag == "unchanged":
        imflag = cv2.IMREAD_UNCHANGED
    elif flag == "grayscale":
        imflag = cv2.IMREAD_GRAYSCALE
    else:
        imflag = cv2.IMREAD_COLOR
    img = cv2.imread(img_or_path, imflag)
    if img is None:
        raise IOError(f"Cannot read image: {img_or_path}")
    return img


def imwrite(img: np.ndarray, file_path: str, params=None):
    cv2.imwrite(file_path, img)


def imfrombytes(content: bytes, flag="color", backend=None) -> np.ndarray:
    arr = np.frombuffer(content, dtype=np.uint8)
    if flag == "unchanged":
        imflag = cv2.IMREAD_UNCHANGED
    elif flag == "grayscale":
        imflag = cv2.IMREAD_GRAYSCALE
    else:
        imflag = cv2.IMREAD_COLOR
    img = cv2.imdecode(arr, imflag)
    return img


def imdecode(content, flag="color", backend=None):
    return imfrombytes(content, flag=flag, backend=backend)


def imresize(img, size, return_scale=False, interpolation="bilinear"):
    if isinstance(size, (list, tuple)) and len(size) == 2:
        w, h = size
    else:
        raise ValueError("size must be (w, h)")
    interp = cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_NEAREST
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interp)
    if return_scale:
        h_scale = h / img.shape[0]
        w_scale = w / img.shape[1]
        return resized, w_scale, h_scale
    return resized


def _rescale_size(img_size, scale):
    """Compute new (w, h) and uniform scale factor (mmcv-compatible)."""
    w, h = img_size
    if isinstance(scale, (float, int)):
        scale_factor = float(scale)
        return (int(w * scale_factor + 0.5), int(h * scale_factor + 0.5)), scale_factor
    max_long_edge = max(scale)
    max_short_edge = min(scale)
    scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    new_w = int(w * scale_factor + 0.5)
    new_h = int(h * scale_factor + 0.5)
    return (new_w, new_h), scale_factor


def imrescale(img, scale, return_scale=False, interpolation="bilinear", backend=None):
    h, w = img.shape[:2]
    new_size, scale_factor = _rescale_size((w, h), scale)
    interp = cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_NEAREST
    resized = cv2.resize(img, new_size, interpolation=interp)
    if return_scale:
        return resized, scale_factor
    return resized


def impad(img, shape=None, pad_val=0, padding_mode="constant"):
    if shape is None:
        return img
    h, w = img.shape[:2]
    target_h, target_w = shape
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_h == 0 and pad_w == 0:
        return img
    return cv2.copyMakeBorder(
        img,
        0,
        pad_h,
        0,
        pad_w,
        cv2.BORDER_CONSTANT,
        value=pad_val if isinstance(pad_val, (int, float)) else 0,
    )


def impad_to_multiple(img, divisor, pad_val=0):
    h, w = img.shape[:2]
    pad_h = int(np.ceil(h / divisor) * divisor) - h
    pad_w = int(np.ceil(w / divisor) * divisor) - w
    return impad(img, (h + pad_h, w + pad_w), pad_val=pad_val)


def imflip(img, direction="horizontal"):
    if direction == "horizontal":
        return np.flip(img, axis=1).copy()
    return np.flip(img, axis=0).copy()


def imnormalize(img, mean, std, to_rgb=False):
    img = img.astype(np.float32)
    if to_rgb and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (img - mean) / std


def imshow(img, win_name="", wait_time=0):
    cv2.imshow(win_name, img)
    cv2.waitKey(wait_time)


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def bgr2hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def hsv2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if img.ndim == 2:
        return clahe_op.apply(img)
    channels = [clahe_op.apply(c) for c in cv2.split(img)]
    return cv2.merge(channels)


def imrotate(img, angle, center=None, scale=1.0, border_value=0):
    h, w = img.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, matrix, (w, h), borderValue=border_value)


def lut_transform(img, lut_table):
    return cv2.LUT(img, lut_table)


def tensor2imgs(tensor, mean=None, std=None, to_rgb=True):
    """Convert 4D tensor to list of uint8 images."""
    imgs = []
    for i in range(tensor.size(0)):
        img = tensor[i].detach().cpu().numpy()
        if mean is not None and std is not None:
            mean_arr = np.array(mean).reshape(-1, 1, 1)
            std_arr = np.array(std).reshape(-1, 1, 1)
            img = img * std_arr + mean_arr
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        if to_rgb and img.shape[2] == 3:
            pass  # already CHW denorm; caller handles
        imgs.append(img)
    return imgs
