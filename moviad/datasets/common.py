from abc import abstractmethod
from enum import Enum

from torch.utils.data.dataset import Dataset
from moviad.utilities.configurations import TaskType, Split


class IadDataset(Dataset):
    task : TaskType
    split: Split
    class_name: str
    dataset_path: str

    @abstractmethod
    def set_category(self, category: str):
        self.class_name = category

    @abstractmethod
    def load_dataset(self):
        pass
