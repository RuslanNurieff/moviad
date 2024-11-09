import pandas as pd
from torch.utils.data import Dataset

from moviad.datasets.visa.visa_data import VisaData
from moviad.datasets.visa.visa_dataset_configurations import VisaDatasetCategory
from moviad.utilities.configurations import Split


class VisaDataset(Dataset):
    root_path: str
    csv_path: str
    split: Split
    category: VisaDatasetCategory
    data: VisaData
    transform: None

    def __init__(self, root_path: str, csv_path: str, split: Split, category: VisaDatasetCategory):
        self.root_path = root_path
        self.csv_path = csv_path
        self.split = split
        self.category = category
        self.dataframe = pd.read_csv(csv_path)
        self.dataframe = self.dataframe[self.dataframe["split"] == split.value]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        pass
