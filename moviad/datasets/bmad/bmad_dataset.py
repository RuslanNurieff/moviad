from enum import Enum
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path

from moviad.utilities.configurations import TaskType, Split, LabelName
from moviad.datasets.iad_dataset import IadDataset

IMG_EXTENSIONS = (".png", ".PNG")

CATEGORIES_MAPPING = {
    "brain": "Brain_AD",
    "chest": "Chest_AD",
    "histopathology": "Histopathology_AD",
    "liver": "Liver_AD",
    "retinaoct": "RetinaOCT2017_AD",
    "retinaresc": "RetinaRESC_AD",
}


CATEGORIES = ("brain", "liver", "retinaoct", "chest", "histopathology", "retinaresc")


class BMAD(IadDataset):
    """
    Dataset class for BMAD (Benchmarks for Medical Anomaly Detection)
    Handles both segmentation mask and image-level annotation categories
    """

    _CATEGORIES_WITH_MASK = {
        "Brain_AD": True,
        "Liver_AD": True,
        "RetinaRESC_AD": True,
        "RetinaOCT2017_AD": False,
        "Histopathology_AD": False,
        "Chest_AD": False,
    }

    def __init__(
        self,
        task_type: TaskType,
        root_dir: str,
        category: str,
        split: Split,
        norm: bool = True,
        image_size=(256, 256),
    ):
        """
        Args:
            task_type (TaskType): Type of task (e.g., classification, segmentation).
            root_dir (str): Root directory of BMAD dataset.
            category (str): Category name (e.g., 'brain', 'liver', etc.).
            split (Split): Dataset split ('train', 'test').
            norm (bool): Whether to normalize images.
            image_size (tuple): Target image size for resizing.
        """
        self.category = CATEGORIES_MAPPING[category]
        self.root_category = Path(root_dir) / Path(self.category)
        self.split = split
        self.has_masks = self._CATEGORIES_WITH_MASK[self.category]
        self.image_size = image_size
        self.preload_imgs = False
        self.data = None

        self.build_root_path = self.root_category / split

        if norm:
            t_list = [
                transforms.ToTensor(),
                transforms.Resize(self.image_size, antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        else:
            t_list = [
                transforms.ToTensor(),
                transforms.Resize(self.image_size, antialias=True),
            ]

        self.transform_image = transforms.Compose(t_list)

        self.transform_mask = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    self.image_size,  # BMAD masks are equal to the original image size
                    antialias=True,
                    interpolation=InterpolationMode.NEAREST,
                ),
            ]
        )

        self.samples = self._load_samples()

    def _load_samples(self):
        """Collect dataset samples as a DataFrame with robust path handling.

        Columns:
        - image_path: absolute file path to the image
        - label: original label folder name (as on disk)
        - label_norm: normalized lowercase label ("good" or "ungood")
        - label_index: 0 for good, 1 for ungood
        - mask_path: absolute path to mask (only for test and categories with masks); empty string otherwise
        """
        split_dir = Path(self.build_root_path)
        rows = []

        for f in split_dir.rglob("*"):
            if f.is_file() and f.suffix in IMG_EXTENSIONS:
                # Expect structure like: split/label/[img or label] if exists (because train doesn't have it)/file.png
                rel = f.relative_to(split_dir)
                parts = list(rel.parts)
                if not parts:
                    continue
                # Determine label and whether there's an intermediate 'img' or 'label' folder
                label_folder = parts[0]
                maybe_sub = parts[1] if len(parts) > 2 else None
                # image files are either directly under label/ or under label/img/
                if maybe_sub and maybe_sub.lower() in ("img", "label"):
                    sub_folder = maybe_sub
                    filename = Path(*parts[2:]) if len(parts) > 2 else Path(parts[-1])
                else:
                    sub_folder = None
                    filename = Path(*parts[1:]) if len(parts) > 1 else Path(parts[-1])

                # Only the actual images
                if sub_folder is None or sub_folder.lower() == "img":
                    rows.append(
                        {
                            "image_path": str(f.resolve()),
                            "label": label_folder,
                            "_sub_folder": sub_folder or "",
                            "_filename": str(filename),
                        }
                    )

        if not rows:
            return pd.DataFrame(
                columns=["image_path", "label", "label_index", "mask_path"]
            )  # empty

        samples = pd.DataFrame(rows)

        samples.loc[samples["label"] == "good", "label_index"] = LabelName.NORMAL
        samples.loc[samples["label"] == "ungood", "label_index"] = LabelName.ABNORMAL
        samples["label_index"] = samples["label_index"].fillna(-1).astype(int)

        samples["mask_path"] = ""

        # For test split and categories with masks, try to get mask paths for ungood images
        if self.split == Split.TEST and self.has_masks:
            mask_candidates = []
            for _, row in samples.iterrows():
                if row["label_index"] != 1:
                    mask_candidates.append("")
                    continue
                img_path = Path(row["image_path"])  # .../split/Ungood/img/file.png
                # Try replacing 'img' segment with 'label'
                parts = list(img_path.parts)
                try:
                    # Find the 'img' segment near the end (after label)
                    idx = (
                        len(parts) - parts[::-1].index("img") - 1
                        if "img" in parts
                        else -1
                    )
                except ValueError:
                    idx = -1
                mask_path = None
                if idx >= 0:
                    parts_copy = parts.copy()
                    parts_copy[idx] = "label"
                    candidate = Path(*parts_copy)
                    if candidate.exists():
                        mask_path = candidate
                # If not found by simple replace, try constructing .../<split>/<label>/label/<same filename>
                if mask_path is None:
                    rel = img_path.relative_to(Path(self.root_category))
                    # rel = <split>/<label>/img/<file> or <split>/<label>/<file>
                    rel_parts = list(rel.parts)
                    # ensure we insert 'label' after label dir
                    if len(rel_parts) >= 3 and rel_parts[2].lower() == "img":
                        rel_parts[2] = "label"
                    else:
                        rel_parts.insert(2, "label")
                    candidate2 = Path(self.root_category, *rel_parts)
                    if candidate2.exists():
                        mask_path = candidate2
                mask_candidates.append(str(mask_path) if mask_path is not None else "")
            samples["mask_path"] = mask_candidates

        # Drop helper columns
        samples = samples.drop(
            columns=[c for c in ["_sub_folder", "_filename"] if c in samples.columns]
        )
        if self.preload_imgs:
            self.data = [
                self.transform_image(
                    Image.open(self.samples.iloc[index].image_path).convert("RGB")
                )
                for index in range(len(self.samples))
            ]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Args:
            index (int) : index of the element to be returned

        Returns:
            image (Tensor) : tensor of shape (C,H,W) with values in [0,1]
            label (int) : label of the image
            mask (Tensor) : tensor of shape (1,H,W) with values in [0,1]
            path (str) : path of the input image
        """

        # open the image and get the tensor
        if self.preload_imgs and self.data is not None:
            image = self.data[index]
        else:
            image = self.transform_image(
                Image.open(self.samples.iloc[index]["image_path"]).convert("RGB")
            )

        if self.split == Split.TRAIN:
            return image
        else:
            # return also the label, the mask and the path
            label = int(self.samples.iloc[index]["label_index"])
            path = self.samples.iloc[index]["image_path"]
            if label == LabelName.ABNORMAL:
                mask_path = self.samples.iloc[index].get("mask_path", "")
                if self.has_masks and mask_path:
                    mask = Image.open(mask_path).convert("L")
                    mask = self.transform_mask(mask)
                else:
                    mask = torch.zeros(1, *self.image_size)
            else:
                mask = torch.zeros(1, *self.image_size)

            return image, label, mask.int(), path
