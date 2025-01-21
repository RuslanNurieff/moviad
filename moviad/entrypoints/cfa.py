import random
import argparse
import pathlib
import tempfile
from dataclasses import dataclass

import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from moviad.datasets.common import IadDataset
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.cfa.cfa import CFA
from moviad.trainers.trainer_cfa import TrainerCFA
from moviad.utilities.configurations import TaskType
from moviad.utilities.evaluator import Evaluator


@dataclass
class CFAArguments:
    train_dataset: IadDataset = None
    test_dataset: IadDataset = None
    batch_size: int = 2 # default value
    category: str = None
    backbone: str = None
    ad_layers: list = None
    epochs: int = 10
    save_path: str = "./temp.pt"
    model_checkpoint_path: str = f"./patch.pt"
    visual_test_path: str = None
    device: torch.device = None
    contamination_ratio: float = 0.0
    seed: int = 4

def train_cfa(args: CFAArguments, logger=None):
    gamma_c = 1
    gamma_d = 1
    print(f"Training CFA for category: {args.category} \n")
    print(f"Length train dataset: {len(args.train_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(args.train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   drop_last=True)

    print(f"Length test dataset: {len(args.test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(args.test_dataset, batch_size=args.batch_size, shuffle=True,
                                                  drop_last=True)

    feature_extractor = CustomFeatureExtractor(args.backbone, args.ad_layers, args.device)

    cfa_model = CFA(feature_extractor, args.backbone, args.device)
    cfa_model.initialize_memory_bank(train_dataloader)
    cfa_model = cfa_model.to(args.device)

    trainer = TrainerCFA(cfa_model, args.backbone, feature_extractor, train_dataloader, test_dataloader, args.category,
                         args.device, logger)
    results, best_results = trainer.train(args.epochs)

    # save the model
    if args.save_path:
        torch.save(cfa_model.state_dict(), args.save_path)


    return results, best_results


def test_cfa(args: CFAArguments, logger=None):
    gamma_c = 1
    gamma_d = 1

    if logger is not None:
        logger.config.update({
            "ad_model": "cfa",
            "dataset": type(args.test_dataset).__name__,
            "category": args.category,
            "backbone": args.backbone,
            "ad_layers": args.ad_layers,
        }, allow_val_change=True)

    print(f"Length test dataset: {len(args.test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(args.test_dataset, batch_size=args.batch_size, shuffle=True)

    # load the model
    feature_extractor = CustomFeatureExtractor(args.backbone, args.ad_layers, args.device, True, False, None)
    cfa_model = CFA(feature_extractor, args.backbone, args.device)
    cfa_model.load_model(args.model_checkpoint_path)
    cfa_model.to(args.device)
    cfa_model.eval()

    evaluator = Evaluator(test_dataloader, args.device)
    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(cfa_model)

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

    # chek for the visual test
    if args.visual_test_path:

        # Get output directory.
        dirpath = pathlib.Path(args.visual_test_path)
        dirpath.mkdir(parents=True, exist_ok=True)

        for images, labels, masks, paths in tqdm(iter(test_dataloader)):
            anomaly_maps, pred_scores = cfa_model(images.to(args.device))

            anomaly_maps = torch.permute(anomaly_maps, (0, 2, 3, 1))

            for i in range(anomaly_maps.shape[0]):
                cfa_model.save_anomaly_map(dirpath, anomaly_maps[i].cpu().numpy(), pred_scores[i], paths[i], labels[i],
                                           masks[i])
