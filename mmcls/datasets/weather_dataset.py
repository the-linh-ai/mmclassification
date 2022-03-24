# Copyright (c) OpenMMLab. All rights reserved.
import os

import numpy as np
import pandas as pd
from PIL import Image

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class WeatherDataset(BaseDataset):
    """Weather and lighting condition dataset for autonomous driving.
    Note that this is a multi-task classification dataset (consisting of
    weather classfication and lighting condition classification.)

    Reference: https://arxiv.org/abs/2104.14042
    """  # noqa: E501

    CLASSES = [
        ["day", "night", "twilight"],  # lighting conditions
        ["clear", "rain", "snow"],  # weather conditions
    ]

    def load_annotations(self):
        # Read dataframe
        self.data_df = pd.read_csv(self.ann_file)
        image_names = self.data_df["image name"]
        image_paths = image_names.apply(
            lambda x: os.path.join(self.data_prefix, "images", x + ".jpg")
        )

        # Read images and labels
        data_infos = []
        for i in range(len(image_paths)):
            row = self.data_df.iloc[i]
            lighting = row.loc[["day", "night", "twilight"]].tolist().index(1)
            weather = row.loc[["clear", "rain", "snow"]].tolist().index(1)
            data_infos.append({
                "img_prefix": None,
                "img_info": {"filename": image_paths[i]},
                "gt_label": np.array([lighting, weather], dtype=np.int64),
            })

        return data_infos
