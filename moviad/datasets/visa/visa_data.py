import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List

import PIL.Image as Image
import numpy as np
from PIL import ImageEnhance
from pandas.core.interchange.dataframe_protocol import DataFrame

from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClass, RealIadAnomalyClass
from moviad.datasets.visa.visa_dataset_configurations import VisaDatasetCategory
from moviad.utilities.configurations import Split

import json

from moviad.utilities.pil_image_utils import min_max_scale_image


class VisaAnomalyClass(Enum):
    """
    Enum class for anomaly
    detection data sources
    """
    ANOMALY = "anomaly"
    NORMAL = "normal"

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
    label: VisaAnomalyClass
    image_path: Path

@dataclass
class VisaData:
    meta: DataFrame
    data: List[VisaImageData]
    images: List[DatasetImageEntry] = None

    def load_images(self, img_root_dir: str) -> None:
        self.images = []
        image = None
        mask = None
        images_not_found = []
        masks_not_found = []
        for index, row in self.meta.iterrows():
            img_path = Path(os.path.join(img_root_dir, row['image']))
            if not os.path.exists(img_path):
                images_not_found.append(img_path)
                continue
            image = Image.open(img_path).convert("RGB")
            label = VisaAnomalyClass(row['label'])

            if row['label'] != VisaAnomalyClass.NORMAL.value:
                image_mask_path = Path(os.path.join(img_root_dir, row['mask']))
                if not os.path.exists(image_mask_path):
                    masks_not_found.append(image_mask_path)
                    continue
                mask = Image.open(image_mask_path).convert("L")
                mask = min_max_scale_image(mask, output_dtype=np.uint8)

            self.images.append(DatasetImageEntry(image=image, mask=mask, label=label, image_path=img_path))

        if len(images_not_found) == self.data.__len__():
            raise ValueError("No images found in the dataset. Check root directory or image paths.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item) -> (VisaImageData, DatasetImageEntry):
        return self.data[item], self.images[item]