# Copyright (c) OpenMMLab. All rights reserved.
import os
from pathlib import Path

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class TestDataset(BaseDataset):
    """A generic test dataset that helps to retrieve images from a given
    directory and regex pattern. Useful for inference.

    Args:
        data_prefix (str): path to the parent directory.
        pattern (str): patter to search for the images using
            `Path(data_prefix).rglob(pattern)`.
    """  # noqa: E501

    CLASSES = ["unknown"]

    def __init__(self, pattern, debug=False, *args, **kwargs):
        self.pattern = pattern
        self.debug = debug
        super().__init__(*args, **kwargs)

    def load_annotations(self):
        data_infos = []
        image_paths = list(Path(self.data_prefix).rglob(self.pattern))
        if self.debug:
            image_paths = image_paths[:50]

        for image_path in image_paths:
            data_infos.append({
            "img_prefix": None,
            "img_info": {"filename": str(image_path)},
            "gt_label": 0,
        })
        return data_infos
