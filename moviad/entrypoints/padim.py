import os, random
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from moviad.models.padim.padim import Padim
from moviad.trainers.trainer_padim import PadimTrainer
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.evaluator import Evaluator, append_results
from moviad.utilities.configurations import TaskType, Split

BATCH_SIZE = 2
IMAGE_INPUT_SIZE = (224, 224)
OUTPUT_SIZE = (224, 224)


def train_padim(train_dataset: Dataset, test_dataset: Dataset, category: str, backbone: str, ad_layers: list,
                device: torch.device, model_checkpoint_save_path: str, diagonal_convergence: bool = False,
                results_dirpath: str = None, logger = None) -> None:
    padim = Padim(
        backbone,
        category,
        device=device,
        diag_cov=diagonal_convergence,
        layers_idxs=ad_layers,
    )
    padim.to(device)
    trainer = PadimTrainer(
        model=padim,
        device=device,
        save_path=model_checkpoint_save_path,
        data_path=None,
        class_name=category,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, pin_memory=True, drop_last=True
    )

    trainer.train(train_dataloader, logger)

    # evaluate the model
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    evaluator = Evaluator(test_dataloader=test_dataloader, device=device)
    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(padim)
    if logger is not None:
        logger.log({
            "img_roc": img_roc,
            "pxl_roc": pxl_roc,
            "f1_img": f1_img,
            "f1_pxl": f1_pxl,
            "img_pr": img_pr,
            "pxl_pr": pxl_pr,
            "pxl_pro": pxl_pro,
        })

    print("Evaluation performances:")
    print(f"""
            img_roc: {img_roc}
            pxl_roc: {pxl_roc}
            f1_img: {f1_img}
            f1_pxl: {f1_pxl}
            img_pr: {img_pr}
            pxl_pr: {pxl_pr}
            pxl_pro: {pxl_pro}
            """)


def test_padim(test_dataset: Dataset, category: str, backbone: str, ad_layers: tuple, device: torch.device,
               model_checkpoint_path: str, results_dirpath: str = None):
    padim = Padim(
        backbone,
        category,
        device=device,
        layers_idxs=ad_layers,
    )
    path = padim.get_model_savepath(model_checkpoint_path)
    padim.load_state_dict(
        torch.load(path, map_location=device), strict=False
    )
    padim.to(device)
    print(f"Loaded model from path: {path}")

    # Evaluator
    padim.eval()

    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # evaluate the model
    evaluator = Evaluator(test_dataloader=test_dataloader, device=device)
    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(padim)

    print("Evaluation performances:")
    print(f"""
        img_roc: {img_roc}
        pxl_roc: {pxl_roc}
        f1_img: {f1_img}
        f1_pxl: {f1_pxl}
        img_pr: {img_pr}
        pxl_pr: {pxl_pr}
        pxl_pro: {pxl_pro}
        """)



def save_results(results_dirpath: str, category: str, seed: int, scores: tuple, backbone: str, ad_layers: tuple,
                 img_input_size: tuple, output_size: tuple):
    metrics_savefile = Path(
        results_dirpath, f"metrics_{backbone}.csv"
    )
    # check if the metrics path exists
    dirpath = os.path.dirname(metrics_savefile)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # save the scores
    append_results(
        metrics_savefile,
        category,
        seed,
        *scores,
        "padim",  # ad_model
        ad_layers,
        backbone,
        "IMAGENET1K_V2",  # NOTE: hardcoded, should be changed
        None,  # bootstrap_layer
        -1,  # epochs (not used)
        img_input_size,
        output_size,
    )