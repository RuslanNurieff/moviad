from moviad.datasets.builder import DatasetFactory, DatasetConfig, DatasetType
from moviad.datasets.common import IadDataset
from moviad.utilities.configurations import Split


def load_datasets(dataset_config: DatasetConfig, dataset_type: DatasetType, dataset_category: str)\
        -> (IadDataset, IadDataset):
    dataset_factory = DatasetFactory(dataset_config)
    train_dataset = dataset_factory.build(dataset_type, Split.TRAIN, dataset_category)
    test_dataset = dataset_factory.build(dataset_type, Split.TEST, dataset_category)
    train_dataset.load_dataset()
    test_dataset.load_dataset()
    return train_dataset, test_dataset