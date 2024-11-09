import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import PIL.Image as Image
from pandas.core.interchange.dataframe_protocol import DataFrame

from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClass, RealIadAnomalyClass
from moviad.datasets.visa.visa_dataset_configurations import VisaDatasetCategory
from moviad.utilities.configurations import Split

import json

@dataclass
class VisaImageData:
    category: VisaDatasetCategory
    anomaly_class: str
    image_path: str
    mask_path: Optional[str]  # mask_path may be None

@dataclass
class DatasetImageEntry:
    image: Image
    mask: Image

@dataclass
class VisaData:
    meta: DataFrame
    data: List[VisaImageData]
    images: List[DatasetImageEntry] = None
    class_name: VisaDatasetCategory = None

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

    def __getitem__(self, item) -> (VisaImageData, DatasetImageEntry):
        return self.data[item], self.images[item]