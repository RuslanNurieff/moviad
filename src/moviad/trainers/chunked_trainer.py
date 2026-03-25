from __future__ import annotations
import torch
from typing import Any, Callable

from moviad.datasets.chunked_dataset import ChunkedDataset
from moviad.utilities.evaluation.evaluator import Evaluator
from moviad.models import VADModel
from moviad.models.training_args import TrainingArgs
from moviad.utilities.evaluation.metrics import Metric
from moviad.trainers.trainer import Trainer

class ChunkedTrainer:

    def __init__(
        self,
        train_args: TrainingArgs,
        model: VADModel,
        chunked_dataset: ChunkedDataset,
        metrics: list[Metric],
        device: torch.device,
        logger: Any | None = None,
        logging_prefix: str = "",
        save_path: str | None = None,
        saving_criteria: Callable | None = None,
    ):
        self.model = model
        self.chunked_dataset = chunked_dataset
        self.device = device
        self.logger = logger
        self.logging_prefix = logging_prefix
        self.save_path = save_path
        self.saving_criteria = saving_criteria
        self.train_args = train_args
        self.metrics = metrics

    def train(self):

        self.train_args.init_train(self.model)

        if self.logger:
            self.logger.config.update(self.train_args.__to_dict__())

        for chunk_index in range(len(self.chunked_dataset)):

            print(f"Training on chunk {chunk_index + 1}/{len(self.chunked_dataset)}")

            train_chunk = self.chunked_dataset.next_chunk()
            train_dataloader = torch.utils.data.DataLoader(train_chunk, batch_size=self.train_args.batch_size, shuffle=True)

            for epoch in range(self.train_args.epochs):

                self.model.train()

                print(f"EPOCH: {epoch}")

                avg_batch_loss = self.model.train_chunk(train_dataloader, self.train_args)

                if self.logger:
                    self.logger.log({
                        f"{self.logging_prefix}epoch" : epoch,
                        f"{self.logging_prefix}train_loss" : avg_batch_loss
                    })

        eval_dataloader = torch.utils.data.DataLoader(self.chunked_dataset.get_test_dataset(), batch_size=self.train_args.batch_size, shuffle=False)
        print("Evaluating model...")
        results = Evaluator.evaluate(self.model, eval_dataloader, self.metrics, self.device)

        print("Training performances:")
        Trainer.print_metrics(results)

        if self.logger is not None:
            if self.logging_prefix is not None:
                self.logger.log({
                    f"{self.logging_prefix}eval/{metric_name}": value for metric_name, value in results.items()
                }, step=epoch+1)
