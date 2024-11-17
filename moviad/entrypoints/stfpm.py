import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader

from moviad.models.stfpm.stfpm import Stfpm
from moviad.trainers.trainer_stfpm import train_param_grid_search
from moviad.utilities.evaluator import Evaluator, append_results
from tests.main.common import StfpmTrainingParams


def train_stfpm(params: StfpmTrainingParams) -> None:
    ad_model = "stfpm"

    print(f"Training with params: {params}")
    params.epochs = [params.epochs] * len(params.ad_layers)
    trained_models_filepaths = train_param_grid_search(params.__dict__)
