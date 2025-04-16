from typing import List

from tqdm import tqdm 
import torch
from torch import Tensor
from torch import nn

from moviad.utilities.evaluator import Evaluator
from moviad.trainers.trainer import TrainerResult


class FastflowLoss(nn.Module):
    """FastFlow Loss."""

    def forward(self, hidden_variables: List[Tensor], jacobians: List[Tensor]) -> Tensor:
        """Calculate the Fastflow loss.

        Args:
            hidden_variables (List[Tensor]): Hidden variables from the fastflow model. f: X -> Z
            jacobians (List[Tensor]): Log of the jacobian determinants from the fastflow model.

        Returns:
            Tensor: Fastflow loss computed based on the hidden variables and the log of the Jacobians.
        """
        loss = torch.tensor(0.0, device=hidden_variables[0].device)  # pylint: disable=not-callable
        for (hidden_variable, jacobian) in zip(hidden_variables, jacobians):
            loss += torch.mean(0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - jacobian)
        return loss

class TrainerFastFlow():
    def __init__(self, model, train_dataloader, test_dataloader, device, logger=None):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.device = device
        self.evaluator = Evaluator(test_dataloader, device)
        self.logger = logger
    
    def train(self, epochs:int, evaluation_epoch_interval: int = 10) -> (TrainerResult, TrainerResult):

        self.optimizer = torch.optim.Adam(
            self.model.parameters()
        )
        self.loss = FastflowLoss()

        best_img_roc = 0
        best_pxl_roc = 0
        best_img_f1 = 0
        best_pxl_f1 = 0
        best_img_pr = 0
        best_pxl_pr = 0
        best_pxl_pro = 0

        if self.logger is not None:
            pass #TODO: add configuration logging

        for epoch in range(epochs):

            self.model.train()

            avg_batch_loss = 0.0
            print("Epoch: ", epoch)
            for batch in tqdm(self.train_dataloader):
                batch = batch.to(self.device)
                hidden_variables, jacobians = self.model(batch)
                loss = self.loss(hidden_variables, jacobians)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_batch_loss += loss.item()
            
            avg_batch_loss /= len(self.train_dataloader)

            if self.logger is not None:
                self.logger.log({
                    "current_epoch" : epoch,
                    "avg_batch_loss": avg_batch_loss
                })

            if (epoch + 1) % evaluation_epoch_interval == 0 and epoch != 0:
                print("Evaluating model...")
                img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = self.evaluator.evaluate(self.model)

                if self.logger is not None:
                    self.logger.log({
                        "img_roc": img_roc,
                        "pxl_roc": pxl_roc,
                        "f1_img": f1_img,
                        "f1_pxl": f1_pxl,
                        "img_pr": img_pr,
                        "pxl_pr": pxl_pr,
                        "pxl_pro": pxl_pro
                    })

                best_img_roc = img_roc if img_roc > best_img_roc else best_img_roc
                best_pxl_roc = pxl_roc if pxl_roc > best_pxl_roc else best_pxl_roc
                best_img_f1 = f1_img if f1_img > best_img_f1 else best_img_f1
                best_pxl_f1 = f1_pxl if f1_pxl > best_pxl_f1 else best_pxl_f1
                best_img_pr = img_pr if img_pr > best_img_pr else best_img_pr
                best_pxl_pr = pxl_pr if pxl_pr > best_pxl_pr else best_pxl_pr
                best_pxl_pro = pxl_pro if pxl_pro > best_pxl_pro else best_pxl_pro

                print("Mid training performances:")
                print(f"""
                    img_roc: {img_roc} \n
                    pxl_roc: {pxl_roc} \n
                    f1_img: {f1_img} \n
                    f1_pxl: {f1_pxl} \n
                    img_pr: {img_pr} \n
                    pxl_pr: {pxl_pr} \n
                    pxl_pro: {pxl_pro} \n
                """)

        print("Best training performances:")
        print(f"""
                img_roc: {best_img_roc} \n
                pxl_roc: {best_pxl_roc} \n
                f1_img: {best_img_f1} \n
                f1_pxl: {best_pxl_f1} \n
                img_pr: {best_img_pr} \n
                pxl_pr: {best_pxl_pr} \n
                pxl_pro: {best_pxl_pro} \n
        """)

        if self.logger is not None:
            self.logger.log({
                "best_img_roc": img_roc,
                "best_pxl_roc": pxl_roc,
                "best_f1_img": f1_img,
                "best_f1_pxl": f1_pxl,
                "best_img_pr": img_pr,
                "best_pxl_pr": pxl_pr,
                "best_pxl_pro": pxl_pro
            })

        best_results = TrainerResult(
            img_roc=best_img_roc,
            pxl_roc=best_pxl_roc,
            f1_img=best_img_f1,
            f1_pxl=best_pxl_f1,
            img_pr=best_img_pr,
            pxl_pr=best_pxl_pr,
            pxl_pro=best_pxl_pro
        )

        results = TrainerResult(
            img_roc=img_roc,
            pxl_roc=pxl_roc,
            f1_img=f1_img,
            f1_pxl=f1_pxl,
            img_pr=img_pr,
            pxl_pr=pxl_pr,
            pxl_pro=pxl_pro
        )

        return results, best_results

        

        