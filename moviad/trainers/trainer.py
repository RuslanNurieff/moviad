import torch

from moviad.datasets.iad_dataset import IadDataset


class Trainer:
    train_dataloader: IadDataset
    test_dataloader: IadDataset
    model: torch.nn.Module
    device: torch.device

class TrainerResult:
    img_roc: float
    pxl_roc: float
    f1_img: float
    f1_pxl: float
    img_pr: float
    pxl_pr: float
    pxl_pro: float

    def __init__(self, img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro):
        self.img_roc = img_roc
        self.pxl_roc = pxl_roc
        self.f1_img = f1_img
        self.f1_pxl = f1_pxl
        self.img_pr = img_pr
        self.pxl_pr = pxl_pr
        self.pxl_pro = pxl_pro