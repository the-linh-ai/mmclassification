# Copyright (c) OpenMMLab. All rights reserved.
import os
from pathlib import Path

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class ACDCDataset(BaseDataset):
    """Adverse Condition Dataset with Correspondences (ACDC) dataset. This
    dataset was originally released to benchmark segmentation models under
    adverse conditions like rain or fog. However, since there is condition
    classification information in the dataset, we can use it to train
    classification models.

    Reference: https://acdc.vision.ee.ethz.ch/overview

    Args:
        data_prefix (str): path to the parent directory `/path/to/rgb_anon/`
    """  # noqa: E501

    CLASSES = ["normal", "fog", "night", "rain", "snow"]

    def __init__(self, split, *args, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        super().__init__(*args, **kwargs)

    def load_annotations(self):
        # Get all classes except `normal`
        classes = ACDCDataset.CLASSES.copy()
        classes.remove("normal")

        data_infos = []
        cls2idx = dict(zip(ACDCDataset.CLASSES, range(len(ACDCDataset.CLASSES))))

        for c in classes:
            # Find all images belonging to the current class
            c_data_dir = os.path.join(self.data_prefix, c, self.split)
            c_image_paths = list(Path(c_data_dir).rglob("**/*.png"))
            for image_path in c_image_paths:
                data_infos.append({
                "img_prefix": None,
                "img_info": {"filename": str(image_path)},
                "gt_label": cls2idx[c],
            })

            # Find all correspondence images belonging to the `normal` class
            normal_data_dir = os.path.join(
                self.data_prefix, c, self.split + "_ref")
            normal_image_paths = list(Path(normal_data_dir).rglob("**/*.png"))
            for image_path in normal_image_paths:
                data_infos.append({
                "img_prefix": None,
                "img_info": {"filename": str(image_path)},
                "gt_label": cls2idx["normal"],
            })
            assert len(c_image_paths) == len(normal_image_paths)

        return data_infos
