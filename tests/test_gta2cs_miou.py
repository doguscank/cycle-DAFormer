"""GTA2CS Cityscapes val mIoU cross-check (slow, needs data + GPU)."""

from __future__ import annotations

import pytest
import torch

from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils.checkpoint import load_checkpoint
from mmseg.utils.legacy_cfg import update_legacy_cfg
from mmseg.utils.mmcv_shim.config import Config
from mmseg.utils.mmcv_shim.parallel import MMDataParallel


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.requires_gpu
def test_gta2cs_val_miou(
    gta2cs_config_path, gta2cs_checkpoint_path, cityscapes_val_available
):
    if not cityscapes_val_available:
        pytest.skip("Cityscapes val not found under DAFORMER_DATA_ROOT")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for full val eval")

    cfg = update_legacy_cfg(Config.fromfile(str(gta2cs_config_path)))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
    )

    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(
        model,
        str(gta2cs_checkpoint_path),
        revise_keys=[(r"^module\.", ""), (r"^model\.", "")],
        strict=False,
    )
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    results = single_gpu_test(model, loader)
    eval_res = dataset.evaluate(results, metric="mIoU", logger="silent")
    miou = eval_res["mIoU"]  # fraction in [0, 1]
    # Published run ~68.3%; allow ~1.5pt drift on PyTorch 2.x + mmcv-free stack.
    assert miou >= 0.665, f"mIoU {miou * 100:.2f}% below gate 66.5% (paper ~68.3%)"
