"""Shared pytest fixtures for cycle-DAFormer migration tests."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
WEIGHTS_ROOT = Path(
    os.environ.get("DAFORMER_WEIGHTS_ROOT", WORKSPACE_ROOT / "pretrained_weights")
).resolve()
ENCODER_CKPT = Path(
    os.environ.get(
        "DAFORMER_ENCODER_CKPT", WEIGHTS_ROOT / "mit-imagenet-pretrained.pth"
    )
).resolve()
GTA2CS_TAR = WEIGHTS_ROOT / "daformer-gtav2cityscapes.tar.gz"
GTA2CS_EXTRACT_ROOT = Path(
    os.environ.get("DAFORMER_GTA2CS_EXTRACT_ROOT", WEIGHTS_ROOT / "gta2cs")
).resolve()
DATA_ROOT = os.environ.get("DAFORMER_DATA_ROOT", str(REPO_ROOT / "data"))


def _gta2cs_run_dir() -> Path | None:
    explicit = os.environ.get("DAFORMER_GTA2CS_CKPT_DIR")
    if explicit:
        return Path(explicit).resolve()
    run_name = "211108_1622_gta2cs_daformer_s0_7f24c"
    candidate = GTA2CS_EXTRACT_ROOT / run_name
    if (candidate / "latest.pth").is_file():
        return candidate
    nested = GTA2CS_EXTRACT_ROOT / run_name / run_name
    if (nested / "latest.pth").is_file():
        return nested
    return candidate if candidate.exists() else None


@pytest.fixture(scope="session")
def encoder_ckpt_path() -> Path:
    if not ENCODER_CKPT.is_file():
        pytest.skip(f"Encoder checkpoint not found: {ENCODER_CKPT}")
    return ENCODER_CKPT


@pytest.fixture(scope="session")
def gta2cs_run_dir() -> Path:
    if (
        GTA2CS_TAR.is_file()
        and not (GTA2CS_EXTRACT_ROOT / "211108_1622_gta2cs_daformer_s0_7f24c").exists()
    ):
        GTA2CS_EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(GTA2CS_TAR) as tf:
            tf.extractall(GTA2CS_EXTRACT_ROOT)
    run_dir = _gta2cs_run_dir()
    if run_dir is None or not (run_dir / "latest.pth").is_file():
        pytest.skip(
            "GTA2CS checkpoint not found; extract daformer-gtav2cityscapes.tar.gz"
        )
    return run_dir


@pytest.fixture(scope="session")
def gta2cs_config_path(gta2cs_run_dir) -> Path:
    jsons = [p for p in gta2cs_run_dir.glob("*.json") if "log" not in p.name.lower()]
    if not jsons:
        pytest.skip(f"No config JSON in {gta2cs_run_dir}")
    preferred = gta2cs_run_dir / f"{gta2cs_run_dir.name}.json"
    if preferred.is_file():
        return preferred
    return jsons[0]


@pytest.fixture(scope="session")
def gta2cs_checkpoint_path(gta2cs_run_dir) -> Path:
    return gta2cs_run_dir / "latest.pth"


@pytest.fixture(scope="session")
def cityscapes_val_available() -> bool:
    cs_root = Path(DATA_ROOT) / "cityscapes"
    return (cs_root / "leftImg8bit" / "val").is_dir() and (
        cs_root / "gtFine" / "val"
    ).is_dir()


@pytest.fixture
def gta2cs_model_cfg():
    from mmseg.utils.mmcv_shim.config import Config

    cfg_path = (
        REPO_ROOT
        / "configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py"
    )
    cfg = Config.fromfile(str(cfg_path))
    cfg.model.pretrained = str(ENCODER_CKPT)
    return cfg
