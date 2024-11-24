import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import numpy as np
import PIL.Image as Image

from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClass, RealIadAnomalyClass
from moviad.utilities.configurations import Split

import json


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
class DatasetImageEntry:
    image: Image
    mask: Image

@dataclass
class RealIadData:
    meta: MetaData
    data: List[ImageData]
    images: List[DatasetImageEntry] = None
    class_name: RealIadClass = None


    @staticmethod
    def from_json(json_path: str, class_name: RealIadClass, split: Split) -> 'RealIadData':
        with open(json_path, 'r') as f:
            data = json.load(f)

        meta = MetaData(**data["meta"])
        data = [
            ImageData(category=RealIadClass(item["category"]), anomaly_class=RealIadAnomalyClass(item["anomaly_class"]),
                      image_path=item["image_path"], mask_path=item.get("mask_path")) for item in data[split.value]]

        data = [item for item in data if item.category == class_name]

        return RealIadData(meta=meta,
                           data=data,
                           class_name=class_name)
    
    def partition(self, ratio) -> ('RealIadData', 'RealIadData'):
        split_index = int(len(self.data) * ratio)
        split_1 =  RealIadData(meta=self.meta,
                           data=self.data[:split_index],
                           class_name=self.class_name)
        split_2 = RealIadData(meta=self.meta,
                            data=self.data[split_index:],
                            class_name=self.class_name)

        return split_1, split_2

    def partition_diff(self, partition: 'RealIadData') -> 'RealIadData':
        data = [item for item in self.data if item not in partition.data]
        return RealIadData(meta=self.meta,
                           data=data,
                           class_name=self.class_name)


    def load_images(self, img_root_dir: str) -> None:
        self.images = []
        image = None
        mask = None
        images_not_found = []
        masks_not_found = []

        for image_entry in self.data:
            class_image_root_path = os.path.join(img_root_dir, self.class_name.value)
            img_path = Path(os.path.join(class_image_root_path, image_entry.image_path))
            if not os.path.exists(img_path):
                images_not_found.append(img_path)
                continue
            image = Image.open(img_path).convert("RGB")

            if image_entry.mask_path is not None:
                image_mask_path = Path(os.path.join(class_image_root_path, image_entry.mask_path))
                if not os.path.exists(image_mask_path):
                    masks_not_found.append(image_mask_path)
                    continue
                mask = Image.open(image_mask_path).convert("L")

            self.images.append(DatasetImageEntry(image=image, mask=mask))


        if len(images_not_found) == self.data.__len__():
            raise ValueError("No images found in the dataset. Check root directory or image paths.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item) -> (ImageData, DatasetImageEntry):
        return self.data[item], self.images[item]