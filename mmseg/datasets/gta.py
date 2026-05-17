# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class GTADataset(CustomDataset):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, **kwargs):
        assert kwargs.get("split") in [None, "train"]
        if "split" in kwargs:
            kwargs.pop("split")
        super(GTADataset, self).__init__(
            img_suffix=".png", seg_map_suffix="_labelTrainIds.png", split=None, **kwargs
        )

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        img_infos = super().load_annotations(
            img_dir, img_suffix, ann_dir, seg_map_suffix, split
        )
        for img_info in img_infos:
            if "ann" in img_info:
                seg_map = img_info["ann"]["seg_map"]
                img_info["ann"]["seg_map"] = seg_map.replace(
                    "_images/images/", "_labels/labels/"
                )
        return img_infos
