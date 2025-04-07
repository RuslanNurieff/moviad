from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from PIL.Image import Image
import PIL
from torch.utils.data import Dataset

from moviad.datasets.common import IadDataset
from moviad.utilities.configurations import Split, TaskType


@dataclass
class MiicDatasetEntry:
    """
    Represents a single entry in the Miic dataset.
    Example of entry path: '/root/train_root/<split>_<class_name>_<image_id>.jpg'
    """
    image_path: Path
    mask_path: str
    class_name: str
    image_id: int
    split: Split
    image: Image

    def __init__(self, image_path: Path, mask_path: str = None):
        image_file_name = image_path.name
        image_file_name_split = image_file_name.split('_')
        self.image_path = image_path
        self.split = Split(image_file_name_split[0])
        self.class_name = image_file_name_split[1]
        self.image_id = int(image_file_name_split[2].split('.')[0])
        self.mask_path = mask_path



class MiicDataset(IadDataset):
    def __init__(self, root_dir: str, task: TaskType, split: Split, img_size=(224, 224),
                 gt_mask_size: Optional[tuple] = None,
                 preload_imgs: bool = True, ):
        super(MiicDataset)
        assert Path(root_dir).exists(), f"Dataset path {root_dir} does not exist."
        self.root_dir = Path(root_dir)
        self.task = task
        self.split = split
        self.preload_imgs = preload_imgs
        self.img_size = img_size
        self.gt_mask_size = gt_mask_size
        self.images = []

        if self.split == Split.TEST:
            self.masks = []
            self.bounding_boxes = []

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        return None

    def load_dataset(self):
        if self.split == Split.TRAIN:
            self.__load_training_data(self.root_dir)
            return
        self.__load_test_data(self.root_dir)

    def __load_training_data(self, normal_images_root_path: Path):
        image_file_list = sorted(list(normal_images_root_path.glob('**/*.jpg')))
        for image in image_file_list:
            image_entry = MiicDatasetEntry(image)
            if self.preload_imgs:
                with PIL.Image.open(image_entry.image_path) as img:
                    image_entry.image = img.convert("RGB")
        return

    def __load_test_data(self, normal_images_root_path: Path,
                         abnormal_image_root_path: Path,
                         mask_root_path: Path,
                         bounding_box_root_path: Path):
        pass

    def contaminate(self, dataset: Dataset, contamination_ratio: float):
        """
        Contaminate the dataset by adding abnormal samples from the given dataset.

        Args:
            dataset (Dataset): The dataset to contaminate.
            contamination_ratio (float): The ratio of contamination.

        Returns:
            int: The number of contaminated samples.
        """
        return 0

    def compute_contamination_ratio(self):
        """
        Compute the contamination ratio of the dataset.

        Returns:
            float: The contamination ratio.
        """
        return 0.0
