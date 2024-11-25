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

    @abstractmethod
    def contaminate(self, source: 'IadDataset', ratio: float, seed: int = 42) -> None:
        pass

    @abstractmethod
    def partition(self, dataset: 'IadDataset', ratio: float) -> 'IadDataset':
        pass

    @abstractmethod
    def contains(self, entry) -> bool:
        pass
