from tqdm import trange
import torch

from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel
from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.evaluation.metrics import Metric
from moviad.trainers.trainer import Trainer

class FineTuning(ContinualModel):

    def __init__(self, model: VADModel):
        super().__init__(model)

    def start_task(self, task_index: int, train_dataset: VADDataset, train_args: TrainingArgs):
        pass

    def train_task(self, 
                   task_index: int, 
                   train_dataset:VADDataset, 
                   eval_dataset:VADDataset,
                   metrics:list[Metric], 
                   device: torch.device, 
                   logger = None,
                   train_args:TrainingArgs = None):

        trainer = Trainer(
            train_args,
            self.vad_model,
            train_dataset,
            eval_dataset,
            metrics=metrics,
            device=device,
            logger=logger,
            logging_prefix=f"Task_T{task_index}/",
        )

        trainer.train()
        

    def end_task(self, task_index: int=None, train_dataset: VADDataset=None, train_args: TrainingArgs=None):
        pass
