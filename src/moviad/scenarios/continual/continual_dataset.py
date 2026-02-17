from torch.utils.data.dataset import Subset
from typing import List
from torch.utils.data import DataLoader
import numpy as np
import torch
from typing import Tuple

from dataclasses import replace

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.configurations import Split

class ContinualDataset:

    def __init__(self,
                 dataset_arguments: DatasetArguments,
                 dataset_class: VADDataset,
                 categories: List[str] = None,
                ):

        """
        This class manage a tasks stream
        """

        self.dataset_arguments = dataset_arguments
        self.dataset_class = dataset_class
        
        if categories is not None:
            self.categories = categories
        else:
            self.categories = dataset_class.get_categories()

    def __len__(self):
        return len(self.categories)

    def get_task_data(self, task_index:int) -> Tuple[DataLoader, DataLoader]:

        """
        Get the training and test data for the given task.

        Args:
        ----
        - task_index (int)
            task index in the task sequence
        - dataset_path (str)
            where the dataset is stored
        - batch_size (int)
            batch_size

        Returns:
        -------
        (tuple):
            0: train dataloader
            1: test dataloader
        """

        train_dataset = self.dataset_class(self.dataset_arguments, category=self.categories[task_index], split=Split.TRAIN)
        test_dataset = self.dataset_class(self.dataset_arguments, category=self.categories[task_index], split=Split.TEST)

        return train_dataset, test_dataset

    def get_task_data_evaluation(self, task_index:int) -> torch.utils.data.DataLoader:

        """
        Get the test data for the given task.

        Args:
        ----
        - task_index (int)
            task index in the task sequence

        Returns:
        -------
        - torch.utils.data.DataLoader: test dataloader for the given task
        """

        test_dataset = self.dataset_class(self.dataset_arguments, category=self.categories[task_index], split=Split.TEST)

        return test_dataset 

    def get_all_tasks_data(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:

        """
        Get the data for all tasks.

        Args:
        -----
        - split (str)
            "train" or "test"
        """
        all_datasets_train = []
        all_datasets_test = []
        for category in self.categories:

            train_dataset = self.dataset_class(self.dataset_arguments, category=category, split=Split.TRAIN)
            all_datasets_train.append(train_dataset)

            test_dataset = self.dataset_class(self.dataset_arguments, category=category, split=Split.TEST)
            all_datasets_test.append(test_dataset)

        train_dataset = torch.utils.data.ConcatDataset(all_datasets_train)
        test_dataset = torch.utils.data.ConcatDataset(all_datasets_test)
        return train_dataset, test_dataset

    def get_previous_tasks(self, task_index):
        return range(len(self.categories))[:task_index+1]
    
    def get_task_category(self, task_index):
        return self.categories[task_index]