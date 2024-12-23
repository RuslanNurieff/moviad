import random
import argparse
import gc
import pathlib

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from torchvision.transforms import transforms
from tqdm import tqdm

from moviad.common.common_utils import obsolete
from moviad.datasets.common import IadDataset
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset, RealIadClass
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.patchcore.patchcore import PatchCore
from moviad.trainers.trainer_patchcore import TrainerPatchCore
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.evaluator import Evaluator

REAL_IAD_DATASET_PATH = 'E:\\VisualAnomalyDetection\\datasets\\Real-IAD\\realiad_256'
AUDIO_JACK_DATASET_JSON = 'E:/VisualAnomalyDetection/datasets/Real-IAD/realiad_jsons/audiojack.json'
IMAGE_SIZE = (224, 224)

@dataclass
class PatchCoreArgs:
    contamination_ratio : float = 0.0
    visual_test_path = None
    model_checkpoint_path = "./patch.pt"
    train_dataset: IadDataset = None
    test_dataset: IadDataset = None
    category: str = None
    backbone: str = None
    ad_layers: list = None
    img_input_size: tuple = (224, 224)
    save_path: str = "./temp.pt"
    batch_size: int = 2
    device: torch.device = None


def train_patchcore(args: PatchCoreArgs, logger=None) -> None:
    # initialize the feature extractor
    feature_extractor = CustomFeatureExtractor(args.backbone, args.ad_layers, args.device, True, False, None)
    print(f"Training Pathcore for category: {args.category} \n")
    print(f"Length train dataset: {len(args.train_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(args.train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   drop_last=True)

    print(f"Length test dataset: {len(args.test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(args.test_dataset, batch_size=args.batch_size, shuffle=True,
                                                  drop_last=True)

    # define the model
    patchcore = PatchCore(args.device, input_size=args.img_input_size, feature_extractor=feature_extractor)
    patchcore.to(args.device)
    patchcore.train()

    trainer = TrainerPatchCore(patchcore, train_dataloader, test_dataloader, args.device, logger)
    trainer.train()

    # save the model
    if args.save_path:
        torch.save(patchcore.state_dict(), args.save_path)

    # force garbage collector in case
    del patchcore
    del args.test_dataset
    del args.train_dataset
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()

def test_patchcore(args: PatchCoreArgs, logger=None) -> None:
    print(f"Length test dataset: {len(args.test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(args.test_dataset, batch_size=32, shuffle=True)

    # load the model
    feature_extractor = CustomFeatureExtractor(args.backbone, args.ad_layers, args.device, True, False, None)
    patchcore = PatchCore(args.device, input_size=args.img_input_size, feature_extractor=feature_extractor)
    patchcore.load_model(args.model_checkpoint_path)
    patchcore.to(args.device)
    patchcore.eval()

    evaluator = Evaluator(test_dataloader, args.device)
    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(patchcore)

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

    # chek for the visual test
    if args.visual_test_path:

        # Get output directory.
        dirpath = pathlib.Path(args.visual_test_path)
        dirpath.mkdir(parents=True, exist_ok=True)

        for images, labels, masks, paths in tqdm(iter(test_dataloader)):
            anomaly_maps, pred_scores = patchcore(images.to(args.device))

            anomaly_maps = torch.permute(anomaly_maps, (0, 2, 3, 1))

            for i in range(anomaly_maps.shape[0]):
                patchcore.save_anomaly_map(args.visual_test_path, anomaly_maps[i].cpu().numpy(), pred_scores[i], paths[i],
                                           labels[i], masks[i])
