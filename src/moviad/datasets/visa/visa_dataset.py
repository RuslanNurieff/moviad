import glob
import os
import pandas as pd

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from moviad.datasets.vad_dataset import VADDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.configurations import Split, LabelName

CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)

class VISADataset(VADDataset):
    def __init__(
        self,
        dataset_arguments: DatasetArguments,
        category: str,
        split: Split | list[Split]
    ):
        super().__init__(dataset_arguments, category, split)
        self.category = category
        self.dataset_root = self.dataset_arguments.dataset_path 

        self.df = pd.read_csv(os.path.join(self.dataset_root, "split_csv", "1cls.csv"))
        self.df = self.df[(self.df["object"]==category) & (self.df["split"]==split)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = os.path.join(self.dataset_root, self.df.iloc[index]["image"])
        image = self.transform_image(
            Image.open(path).convert("RGB")
        )
        label = LabelName.NORMAL if self.df.iloc[index]["label"] == "normal" else LabelName.ABNORMAL

        if self.split == Split.TRAIN:
            return image
        else:

            mask = None
            if self.df.iloc[index]["label"] == "normal":
                mask = torch.zeros((1, self.resize_shape[1], self.resize_shape[0]))
            else:
                mask_path = os.path.join(self.dataset_root, self.df.iloc[index]["mask"])
                mask = Image.open(mask_path).convert("L")
                mask = self.transform_mask(mask)

                mask = torch.where(
                    mask > 0.0, torch.ones_like(mask), torch.zeros_like(mask)
                )

            return image, label, mask, path

    @staticmethod
    def get_categories() -> list:
        return list(CATEGORIES)