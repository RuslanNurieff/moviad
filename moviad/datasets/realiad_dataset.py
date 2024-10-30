from enum import Enum

import torch
from PIL import Image
import json
from typing import List, Optional
from dataclasses import dataclass
import os
import unittest
from enum import Enum
from torchvision.transforms import transforms
from pandas import DataFrame

from torch.utils.data import Dataset
from pathlib import Path


class RealIadDefectType(Enum):
    AK = "pit"
    BX = "deformation"
    CH = "abrasion"
    HS = "scratch"
    PS = "damage"
    QS = "missing parts"
    YW = "foreign objects"
    ZW = "contamination"


class RealIadAnomalyClass(Enum):
    OK = "OK"
    NG = "NG"


class RealIadClass(Enum):
    AUDIOJACK = "audiojack"
    BOTTLE = "bottle"
    CABLE = "cable"
    CAPSULE = "capsule"
    CARPET = "carpet"
    GRID = "grid"
    HAZELNUT = "hazelnut"
    LEATHER = "leather"
    METAL_NUT = "metal_nut"
    PILL = "pill"
    SCREW = "screw"
    TILE = "tile"
    TOOTHBRUSH = "toothbrush"
    TRANSISTOR = "transistor"
    WOOD = "wood"
    ZIPPER = "zipper"


@dataclass
class MetaData:
    prefix: str
    normal_class: str
    pre_transform: bool


@dataclass
class ImageData:
    category: RealIadClass
    anomaly_class: RealIadAnomalyClass
    image_path: str
    mask_path: Optional[str]  # mask_path may be None


@dataclass
class RealIadData:
    meta: MetaData
    train: List[ImageData]
    images: List[Image] = None
    class_name: RealIadClass = None

    @staticmethod
    def from_json(json_path: str, class_name: RealIadClass) -> 'RealIadData':
        with open(json_path, 'r') as f:
            data = json.load(f)

        meta = MetaData(**data["meta"])
        train_data = [
            ImageData(category=RealIadClass(item["category"]), anomaly_class=RealIadAnomalyClass(item["anomaly_class"]),
                      image_path=item["image_path"], mask_path=item.get("mask_path")) for item in data["train"]]
        return RealIadData(meta=meta,
                           train=train_data,
                           class_name=class_name)

    def load_images(self, img_root_dir: str) -> None:
        self.images = []
        images_not_found = []

        for image_entry in self.train:
            class_image_root_path = os.path.join(img_root_dir, self.class_name.value)
            img_path = Path(os.path.join(class_image_root_path, image_entry.image_path))
            if not os.path.exists(img_path):
                images_not_found.append(img_path)
                continue
            image = Image.open(img_path).convert("RGB")
            self.images.append(image)

        if len(images_not_found) == self.train.__len__():
            raise ValueError("No images found in the dataset. Check root directory or image paths.")

    def __len__(self) -> int:
        return len(self.train)

    def __getitem__(self, item) -> (ImageData, Image):
        return self.train[item], self.images[item]


class RealIadDataset(Dataset):
    def __init__(self, class_name: RealIadClass, img_root_dir: str, json_path: str, transform = None,
                 image_size=(224, 224)) -> None:
        super().__init__()
        if img_root_dir is None:
            raise ValueError("img_dir should not be None")
        if not os.path.exists(img_root_dir):
            raise ValueError(f"img_dir '{img_root_dir}' does not exist")
        if not os.path.isdir(img_root_dir):
            raise ValueError(f"img_dir '{img_root_dir}' is not a directory")
        self.json = json_path
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.class_name = class_name
        self.data: RealIadData = None

    def load_dataset(self) -> None:
        self.data = RealIadData.from_json(self.json, self.class_name)

        if self.data is None:
            raise ValueError("Dataset is None")

        self.__index_images_and_labels__()

    def __len__(self) -> int:
        return self.data.train.__len__()

    def __getitem__(self, item) -> (str, torch.Tensor):
        entry, image = self.data.__getitem__(item)
        if self.transform:
            image = self.transform(image)
        return entry.category.value, image

    def __index_images_and_labels__(self) -> None:
        self.data.load_images(self.img_root_dir)
