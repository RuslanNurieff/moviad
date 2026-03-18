import torch

from moviad.datasets.vad_dataset import VADDataset
from moviad.models.training_args import TrainingArgs
from moviad.scenarios.continual.strategies.replay.replay_model import Replay
from moviad.models.cfa.cfa import CFA
from moviad.utilities.evaluation.metrics import Metric

class CFAContinual(Replay):

    def __init__(self, model:CFA, memory_size: int = 2000, **kwargs):
        super().__init__(model=model, memory_size=memory_size, **kwargs)

    def update_memory_bank(self, task_index: int, train_dataloader: torch.utils.data.DataLoader):
        task_memory_bank = self.vad_model.initialize_memory_bank(train_dataloader)
        new_memory_bank = self.vad_model.memory_bank * (task_index / (task_index + 1)) + task_memory_bank / (task_index + 1)
        self.vad_model.memory_bank = new_memory_bank

    def start_task(self, task_index: int, train_dataset: VADDataset, train_args: TrainingArgs = None):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_args.batch_size,
            shuffle=True,
            num_workers=4   
        )

        # initialize the memory bank for CFA or update it with the new task data
        if task_index == 0:
            self.vad_model.initialize_memory_bank(train_dataloader)
        else: 
            self.update_memory_bank(task_index, train_dataloader)
    

