from enum import Enum

from torch.utils.data.dataset import Dataset

from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset
from moviad.utilities.configurations import TaskType, Split


class IadDataset(Dataset):
    task : TaskType
    split: Split
    category: str
    dataset_path: str

    def __init__(self, task: TaskType, split: Split, category: str, dataset_path: str):
        self.task = task
        self.split = split
        self.category = category
        self.dataset_path = dataset_path



class AnomalyDetectionDatasources(Enum):
    """
    Enum class for anomaly
    detection data sources
    """
    MVTEC = type(MVTecDataset)
    REALIAD = type(RealIadDataset)