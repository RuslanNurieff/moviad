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

class ChunkedDataset:

    def __init__(self,
                 dataset_arguments: DatasetArguments,
                 dataset_class: VADDataset,
                 num_chunks: int = 10
                ):

        self.dataset_arguments = dataset_arguments
        self.dataset_class = dataset_class
        self.num_chunks = num_chunks

        self.chunk_counter = 0
        self._load()

    def _load(self):

        all_datasets_train = []
        all_datasets_test = []
        for category in self.dataset_class.get_categories():

            train_dataset = self.dataset_class(self.dataset_arguments, category=category, split=Split.TRAIN)
            all_datasets_train.append(train_dataset)

            test_dataset = self.dataset_class(self.dataset_arguments, category=category, split=Split.TEST)
            all_datasets_test.append(test_dataset)

        train_dataset = torch.utils.data.ConcatDataset(all_datasets_train)
        test_dataset = torch.utils.data.ConcatDataset(all_datasets_test)
        
        total_length = len(train_dataset)
        chunk_size = total_length // self.num_chunks
        lengths = [chunk_size] * self.num_chunks
        
        # Add any leftover samples to the final chunk so the sum perfectly matches total_length
        lengths[-1] += total_length % self.num_chunks 
        
        # (Optional but recommended) Set a seed so your chunks are the same every time you run the script
        generator = torch.Generator().manual_seed(42)
        
        self.chunks_train = torch.utils.data.random_split(train_dataset, lengths, generator=generator)
        
        self.test_dataset = test_dataset

    def __len__(self):
        return self.num_chunks

    def next_chunk(self):

        if self.chunk_counter < self.num_chunks:
            train_dataset = self.chunks_train[self.chunk_counter]
            self.chunk_counter += 1
            return train_dataset
        else:
            raise StopIteration
        
    def get_test_dataset(self):
        return self.test_dataset