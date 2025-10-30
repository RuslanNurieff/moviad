from tqdm import *
import copy

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

from moviad.models.draem.draem import DRAEM
from moviad.utilities.evaluator import Evaluator
from moviad.trainers.trainer import TrainerResult, Trainer

class TrainerDRAEM(Trainer):

    """
    This class contains the code for training the STFPM model
    """

    def train(self, epochs: int, evaluation_epoch_interval: int = 10) -> tuple[TrainerResult, TrainerResult]:

        optimizer = torch.optim.Adam([
            {'params': self.model.model.parameters(), "lr": 0.0001},
            {'params': self.model.model_seg.parameters(), "lr": 0.0001}
            ])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[epochs*0.8,epochs*0.9],gamma=0.2, last_epoch=-1)

        best_metrics = {}
        best_metrics["img_roc_auc"] = 0
        best_metrics["pxl_roc_auc"] = 0
        best_metrics["img_f1"] = 0
        best_metrics["pxl_f1"] = 0
        best_metrics["img_pr_auc"] = 0
        best_metrics["pxl_pr_auc"] = 0
        best_metrics["pxl_au_pro"] = 0

        # log the training configurations
        if self.logger:
            self.logger.config.update(
                {
                    "epochs": epochs,
                    "learning_rate": DRAEM.DEFAULT_PARAMETERS["learning_rate"],
                    "optimizer": "Adam",
                },
                allow_val_change=True
            )
            self.logger.watch(self.model.model, log='all', log_freq=10)
            self.logger.watch(self.model.model_seg, log='all', log_freq=10)

        for epoch in trange(epochs):

            self.model.train()

            print(f"EPOCH: {epoch}")

            avg_batch_loss = 0
            #train the model
            for batch in tqdm(self.train_dataloader):

                loss = self.model(batch)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                avg_batch_loss += loss.item()
            
            scheduler.step()

            avg_batch_loss /= len(self.train_dataloader)
            print(f"Average batch loss is: {avg_batch_loss}")
            if self.logger:
                self.logger.log({
                    "current_epoch" : epoch,
                    "avg_batch_loss": avg_batch_loss
                })

            if (epoch + 1) % evaluation_epoch_interval == 0 and epoch != 0:
                print("Evaluating model...")
                metrics = self.evaluator.evaluate(self.model)
                
                if self.saving_criteria and self.save_path is not None: 
                    print("Saving model...")
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"Model saved to {self.save_path}")
                
                # update the best metrics
                best_metrics = Trainer.update_best_metrics(best_metrics, metrics)
            
                print("Trainer training performances:")
                Trainer.print_metrics(metrics)

                if self.logger is not None:
                    self.logger.log(best_metrics)

        print("Best training performances:")
        Trainer.print_metrics(best_metrics)

        if self.logger is not None:
            self.logger.log(
                best_metrics
            )

        best_results = TrainerResult(
            **best_metrics
        )

        results = TrainerResult(
            **metrics
        )


        return results, best_results
